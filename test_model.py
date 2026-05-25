
import os
import time
import argparse
import numpy as np
import scipy.io
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn

mlp.use('TkAgg')


# ==========================================
# Model Architectures
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class SRResNet(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(SRResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_residual_blocks)]
        )
        self.upconv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(256, 1, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        res = self.residual_blocks(x)
        x = x + res
        x = self.relu(self.upconv1(x))
        x = self.upconv2(x)
        return x


# ==========================================
# Main Evaluation Execution
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Test trained Super-Resolution model on magnetic field maps.")
    parser.add_argument('--label_dir', type=str, default='./example_data/labels/',
                        help='Path to theoretical/ground-truth mat files')
    parser.add_argument('--data_dir', type=str, default='./example_data/low_res/',
                        help='Path to low-resolution simulation mat files')
    parser.add_argument('--model_path', type=str, default='./example_model.pth',
                        help='Path to the trained model (.pth)')
    parser.add_argument('--output_dir', type=str, default='./results/example_run/',
                        help='Directory to save output plots')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Max number of test samples to evaluate (e.g., 100)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_pic_size = 300
    theory_pic_size = 300

    all_simulations_test = []
    all_theory_test = []

    # 1. Load Data
    loaded_count = 0
    for j in range(1, args.num_samples + 1):
        try:
            mat3 = scipy.io.loadmat(os.path.join(args.label_dir, f'Btheor_tot_{j}.mat'))
            Btot_theory_test = mat3['B_theor']
            all_theory_test.append(Btot_theory_test)

            mat4 = scipy.io.loadmat(os.path.join(args.data_dir, f'Bsim_tot_{j}.mat'))
            Btot_sim_test = mat4['B_sim']
            all_simulations_test.append(Btot_sim_test)
            loaded_count += 1
        except FileNotFoundError:
            print(f"File index {j} not found. Stopping data load at {loaded_count} samples.")
            break

    if loaded_count == 0:
        print("No data loaded. Please check your data directories.")
        return

    # 2. Setup Scalers and reshape exactly as original script
    Scaler_x = MinMaxScaler()
    Scaler_y = MinMaxScaler()

    x_test = np.array(all_simulations_test).reshape(loaded_count, 1, train_pic_size, train_pic_size)
    y_test = np.array(all_theory_test).reshape(-1, 1)

    x_test_check = Scaler_x.fit_transform(x_test.reshape(-1, 1))
    y_test_check = Scaler_y.fit_transform(y_test.reshape(-1, 1))

    x_test = x_test_check.reshape(loaded_count, 1, 300, 300)
    y_test = y_test_check.reshape(loaded_count, 1, 300, 300)

    X_test = torch.tensor(x_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Load Model
    print(f"Loading model from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Model file not found at {args.model_path}. Exiting.")
        return

    generator = SRResNet().to(device)
    generator.load_state_dict(checkpoint)

    low_res_image = X_test.reshape(loaded_count, train_pic_size, train_pic_size)
    low_res_image = torch.tensor(low_res_image, dtype=torch.float).unsqueeze(0).to(device)

    all_pred, all_pred_unnormalized, all_test, all_theory = [], [], [], []

    start_time = time.time()

    # 4. Inference loop
    for k in range(0, loaded_count):
        with torch.no_grad():
            high_res_image = generator(low_res_image[0, k].unsqueeze(0).unsqueeze(0))
            all_pred.append(high_res_image.squeeze(0).squeeze(0).cpu().numpy())

    all_pred = np.array(all_pred)

    x = np.linspace(0, 30, 300)
    y = np.linspace(0, 30, 300)
    X, Y = np.meshgrid(x, y)

    # 5. Output generation exactly mirroring the original logic
    print(f"Saving outputs to {args.output_dir}...")
    for run in range(0, loaded_count):
        all_pred_unnormalized.append(Scaler_x.inverse_transform(all_pred[run]))

        # Prediction
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, Scaler_x.inverse_transform(all_pred[run]), levels=100, cmap='jet')
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel(r'$B[\mu T]$', size=17)
        plt.savefig(os.path.join(args.output_dir, f'test_case_{run + 1}_prediction.png'), bbox_inches='tight',
                    pad_inches=0, dpi=300)
        plt.close()

        # Simulation
        fig, ax = plt.subplots()
        all_test.append(Scaler_x.inverse_transform(x_test[run].reshape(-1, 1)))
        cs = ax.contourf(X, Y,
                         Scaler_x.inverse_transform(x_test[run].reshape(-1, 1)).reshape(train_pic_size, train_pic_size),
                         levels=100, cmap='jet')
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel(r'$B[\mu T]$', size=17)
        plt.savefig(os.path.join(args.output_dir, f'test_case_{run + 1}_sim.png'), bbox_inches='tight', pad_inches=0,
                    dpi=300)
        plt.close()

        # Theory
        fig, ax = plt.subplots()
        all_theory.append(Scaler_y.inverse_transform(y_test[run].reshape(-1, 1)))
        cs = ax.contourf(X, Y, Scaler_y.inverse_transform(y_test[run].reshape(-1, 1)).reshape(theory_pic_size,
                                                                                              theory_pic_size),
                         levels=100, cmap='jet')
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel(r'$B[\mu T]$', size=17)
        plt.savefig(os.path.join(args.output_dir, f'test_case_{run + 1}_theory.png'), bbox_inches='tight', pad_inches=0,
                    dpi=300)
        plt.close()

        print(f"Case: {run + 1:.0f} is done!")

    mdic = {"Xtest": all_test, "Ytest": all_theory, "predictions": all_pred_unnormalized}
    savemat(os.path.join(args.output_dir, 'SRResNet_new_for_analysis.mat'), mdic)
    print('Testing complete.')


if __name__ == '__main__':
    main()