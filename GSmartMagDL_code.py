import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import matplotlib as mlp
import torch
from torch import nn
from scipy.io import savemat
mlp.use('TkAgg')




class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1):
        super().__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)
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

if __name__ == '__main__':
    data_set = 25

    path_label_test = '/'
    path_data_test = '/'

    all_simulations = []
    all_theory = []
    all_simulations_test = []
    all_theory_test = []
    all_interpolation = []
    train_pic_size = 300
    theory_pic_size = 300


    for j in range(1, 101):
        mat3 = scipy.io.loadmat(path_label_test  +'Btheor_tot_'+ str(int(j)) + '.mat') #Btheor_tot_
        Btot_theory_test = mat3['B_theor']
        all_theory_test.append(Btot_theory_test)
        mat4 = scipy.io.loadmat(path_data_test  +'Bsim_tot_'+ str(int(j)) + '.mat')#Bsim_tot_

        Btot_sim_test = mat4['B_sim']
        all_simulations_test.append(Btot_sim_test)
    Scaler_x = MinMaxScaler()
    Scaler_y = MinMaxScaler()

    x_test = np.array(all_simulations_test).reshape(100, 1,train_pic_size,train_pic_size)
    y_test = np.array(all_theory_test).reshape(-1, 1)
    x_test_check = Scaler_x.fit_transform(x_test.reshape(-1, 1))
    y_test_check = Scaler_y.fit_transform(y_test.reshape(-1, 1))

    x_test = x_test_check.reshape(100, 1,300, 300)
    y_test = y_test_check.reshape(100, 1,300, 300)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(
        os.path.join(os.getcwd() + '/trained_model.pth'),
        map_location=torch.device('cpu'))
    generator = SRResNet().to(device)
    generator.load_state_dict(checkpoint)
    low_res_image = X_test.reshape(100,train_pic_size,train_pic_size) # for CNN
    low_res_image = torch.tensor(low_res_image, dtype=torch.float).unsqueeze(0).to(device)  # Shape: (1, 1, 33, 33)
    all_pred, all_interp, all_pred_unnormalized,all_test, all_theory = [], [], [], [], []
    start_time = time.time()
    for k in range(0, 100):
        with torch.no_grad():
            high_res_image = generator(low_res_image[0, k].unsqueeze(0).unsqueeze(0))
            print("The run time of a single prediction is: " + str(time.time() - start_time))
            all_pred.append(high_res_image.squeeze(0).squeeze(0).cpu().numpy())
    all_pred = np.array(all_pred)

    x = np.linspace(0, 30, 300)
    y = np.linspace(0, 30, 300)
    X, Y = np.meshgrid(x, y)

    dir_name = ''
    for run in range(0, 100):
        all_pred_unnormalized.append(Scaler_x.inverse_transform(all_pred[run]))
        fig, ax = plt.subplots()
        # plt.title('test case:' + str(run+1) + ' prediction')
        cs = ax.contourf(X, Y, Scaler_x.inverse_transform(all_pred[run]), levels=100, cmap='jet')  # high_res_image[0]
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel('$B[\mu T]$', size=17)
        plt.savefig(
            os.path.join(os.getcwd() + ' prediction.png'),
            bbox_inches='tight', pad_inches=0, dpi=300)
        fig, ax = plt.subplots()
        all_test.append(Scaler_x.inverse_transform(x_test[run].reshape(-1, 1)))
        cs = ax.contourf(X, Y,
                         Scaler_x.inverse_transform(x_test[run].reshape(-1, 1)).reshape(train_pic_size, train_pic_size),
                         levels=100, cmap='jet')
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel('$B[\mu T]$', size=17)
        plt.savefig(
            os.path.join(os.getcwd() + ' sim.png'),
            bbox_inches='tight', pad_inches=0, dpi=300)
        fig, ax = plt.subplots()
        all_theory.append(Scaler_y.inverse_transform(y_test[run].reshape(-1, 1)))
        cs = ax.contourf(X, Y,
                         Scaler_y.inverse_transform(y_test[run].reshape(-1, 1)).reshape(theory_pic_size, theory_pic_size),
                         levels=100, cmap='jet')
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel('$B[\mu T]$', size=17)

        plt.savefig(
            os.path.join(os.getcwd() +' theory.png'),
            bbox_inches='tight', pad_inches=0, dpi=300)
        mdic = {"Xtest":all_test, "Ytest":all_theory,"predictions":all_pred_unnormalized}
        savemat(os.path.join(os.getcwd() + 'data.mat'),mdic)
