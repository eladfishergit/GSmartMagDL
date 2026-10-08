import time
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import scipy.io
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset,random_split
from torchvision import models
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import matplotlib as mlp
mlp.use('TkAgg')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.Norm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

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
class UNetMagneticSR(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, out_channels=1):
        super().__init__()
        # Encoder
        self.conv1 = ConvBlock(in_channels, base_channels)
        self.conv2 = ConvBlock(base_channels, base_channels * 2)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder
        self.upconv1 = UpConvBlock(base_channels * 8, base_channels * 4)
        self.conv5 = ConvBlock(base_channels * 8, base_channels * 4)

        self.upconv2 = UpConvBlock(base_channels * 4, base_channels * 2)
        self.conv6 = ConvBlock(base_channels * 4, base_channels * 2)

        self.upconv3 = UpConvBlock(base_channels * 2, base_channels)
        self.conv7 = ConvBlock(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)           # -> [B, C, H, W]
        x2 = self.conv2(x1)          # -> [B, 2C, H, W]
        x3 = self.conv3(x2)          # -> [B, 4C, H, W]
        x4 = self.conv4(x3)          # -> [B, 8C, H, W]

        u1 = self.upconv1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv5(u1)

        u2 = self.upconv2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv6(u2)

        u3 = self.upconv3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv7(u3)

        out = self.out(u3)
        return out
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
        self.upconv3 = nn.PixelShuffle(upscale_factor=1)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        res = self.residual_blocks(x)
        x = x + res
        x = self.relu(self.upconv1(x))
        x = self.upconv2(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._conv_block(3, 64, stride=1, normalize=False),
            self._conv_block(64, 64, stride=2),
            self._conv_block(64, 128, stride=1),
            self._conv_block(128, 128, stride=2),
            self._conv_block(128, 256, stride=1),
            self._conv_block(256, 256, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 13 * 13, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def _conv_block(self, in_channels, out_channels, stride, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg[:18])).eval()  # Use first few layers
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.features(input)
        target_features = self.features(target)
        return F.mse_loss(input_features, target_features)
def content_loss(output, target):
    return F.mse_loss(output, target)

def local_mutual_information_map_torch(x, y, patch_radius=5, bins=16, eps=1e-8):
    assert x.shape == y.shape
    B, C, H, W = x.shape
    device = x.device
    k = 2 * patch_radius + 1

    centers = torch.linspace(0, 1, bins, device=device).view(1, bins, 1, 1)

    def soft_assign(v):
        dist = torch.abs(v - centers)
        w = torch.clamp(1 - dist * (bins - 1), min=0.0)
        return w

    wx = soft_assign(x)
    wy = soft_assign(y)

    kernel = torch.ones(1, 1, k, k, device=device) / (k * k)

    px = F.conv2d(wx, kernel.expand(bins, 1, k, k),
                  padding=patch_radius, groups=bins)
    py = F.conv2d(wy, kernel.expand(bins, 1, k, k),
                  padding=patch_radius, groups=bins)
    mi = torch.zeros((B, 1, H, W), device=device)
    log2 = math.log(2.0)
    for i in range(bins):
        wx_i = wx[:, i:i+1]
        px_i = px[:, i:i+1]
        for j in range(bins):
            wy_j = wy[:, j:j+1]
            py_j = py[:, j:j+1]

            wxy = wx_i * wy_j
            pxy = F.conv2d(wxy, kernel, padding=patch_radius)

            denom = px_i * py_j + eps
            mi += pxy * (torch.log(pxy + eps) - torch.log(denom)) / log2

    return mi




def gaussian_smooth_torch(x, sigma=1.0, kernel_size=None):
    """
    x: [B,1,H,W]
    """
    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)

    coords = torch.arange(kernel_size, device=x.device) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    kernel_1d = g.view(1, 1, -1)
    kernel_2d = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

    padding = kernel_size // 2
    return F.conv2d(x, kernel_2d, padding=padding)

def mi_weight_from_map_torch(mi_maps, sigma=1.0, gamma=1.2, eps=1e-8):
    """
    mi_maps: [B,1,H,W] torch tensor
    Returns: weight map [B,1,H,W]
    """
    mi_smooth = gaussian_smooth_torch(mi_maps, sigma=sigma)
    mi_min = mi_smooth.amin(dim=(1,2,3), keepdim=True)
    mi_max = mi_smooth.amax(dim=(1,2,3), keepdim=True)
    mi_norm = (mi_smooth - mi_min) / (mi_max - mi_min + eps)
    weight = mi_norm.pow(gamma)
    weight = torch.clamp(weight, 1e-3, 1.0)

    return weight

def train():
    path_labels ='/home/share_folder/SmartMagDl-2/Labels_new_algo/'
    path_data = '/home/share_folder/SmartMagDl-2/Low_res_new_algo/'

    all_simulations = []
    all_theory = []
    train_pic_size = 300
    theory_pic_size = 300
    batch_size = 4

    for i in range(1, 1001):
        mat1 = scipy.io.loadmat(path_labels +'Btheor_tot_' + str(int(i)) + '.mat')
        Btot_theory_i =mat1['B_theor']
        all_theory.append(Btot_theory_i)
        mat2 = scipy.io.loadmat(path_data +'Bsim_tot_'+ str(int(i)) + '.mat')
        Btot_sim_i =mat2['B_sim']
        all_simulations.append(Btot_sim_i)

    X_train = np.array(all_simulations)
    y_train = np.array(all_theory)
    X_train = X_train.reshape(1000, 1 ,train_pic_size,train_pic_size)
    Scaler_x = MinMaxScaler()
    Scaler_y = MinMaxScaler()
    y_train = y_train.reshape(1000, 1 , theory_pic_size,theory_pic_size)
    X_train_check = Scaler_x.fit_transform(X_train.reshape(-1, 1))
    y_train_check = Scaler_y.fit_transform(y_train.reshape(-1, 1))
    X_train = X_train_check.reshape(1000, 1, 300, 300)
    y_train = y_train_check.reshape(1000, 1, 300, 300)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # all_mi_maps = np.zeros((1000,300,300))
    # for you in range(0, 1000):
    #     all_mi_maps[you] = local_mutual_information_map(X_train.reshape(1000, 300, 300)[you], y_train.reshape(1000, 300, 300)[you], patch_radius=4, bins=16)
    #
    # mi_maps_tensor = torch.tensor(all_mi_maps, dtype=torch.float32)
    #
    # from scipy.ndimage import gaussian_filter
    # mi_smooth = gaussian_filter(mi_maps_tensor, sigma=1.0)
    #
    # # normalize (dataset-wise min/max can be precomputed)
    # mi_norm = (torch.tensor(mi_smooth, dtype=torch.float32)- mi_maps_tensor.min()) / (mi_maps_tensor.max() - mi_maps_tensor.min() + 1e-8)
    # # sharpen
    # gamma = 1.2
    # weight = np.clip(mi_norm ** gamma, 1e-3, 1.0)  # avoid zero


    train_ratio = 0.8
    # train_size = int(train_ratio * X_train.shape[0])
    # full_dataset = TensorDataset(X_tensor, y_tensor, mi_maps_tensor)
    # full_dataset = TensorDataset(X_tensor, y_tensor, weight)
    full_dataset = TensorDataset(X_tensor, y_tensor)


    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)  # 80%
    val_size = (total_size - train_size)//2  # 10%
    test_size = val_size

    train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    # train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size], mi_maps_tensor[:train_size])
    # validation_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:], mi_maps_tensor[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SRResNet().to(device)
    # generator = UNetMagneticSR(in_channels=1, base_channels=64, out_channels=1).to(device)
    train_losses = []
    validation_losses = []
    optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    beta_w = 0.05

    def train_step(engine, batch):
        low_res_images, high_res_images = batch

        low_res_images = low_res_images.to(device)
        high_res_images = high_res_images.to(device)
        optimizer.zero_grad()
        generated_images = generator(low_res_images)
        loss_pixel = content_loss(generated_images, high_res_images)
        mi_maps = local_mutual_information_map_torch(generated_images, high_res_images, patch_radius=1, bins=16)
        weight = mi_weight_from_map_torch(mi_maps, sigma=1.0, gamma=1.2)
        loss = loss_pixel - beta_w * weight.mean()
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_step(engine, batch):
        low_res_images, high_res_images= batch
        low_res_images = low_res_images.to(device)
        high_res_images = high_res_images.to(device)
        with torch.no_grad():
            generated_images = generator(low_res_images)
        return generated_images, high_res_images

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)
    mse_metric = MeanSquaredError()
    mse_metric.attach(evaluator, 'mse')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        train_loss = metrics['mse']
        train_losses.append(train_loss)
        print(f"Epoch {engine.state.epoch} - Avg loss: {engine.state.output:.4f}, train mse: {train_loss:.4f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        validation_loss = metrics['mse']
        validation_losses.append(validation_loss)
        print(f"validation mse: {validation_loss:.4f}")


    num_of_epochs=200
    trainer.run(train_loader, max_epochs=num_of_epochs)
    torch.save(generator.state_dict(), os.path.join(
        os.getcwd() + '/' + f'{time.time():.4}' + '_SmartMagDL2_n_epochs_'+str(num_of_epochs)+'_'+'batch_size_'+f'{batch_size}'+'_weight_'+f'{beta_w:.3f}'+'_128_bins_fused_maps_new_MI_learning' + '.pth'))
    # torch.save(generator.state_dict(), os.path.join(
    #     os.getcwd() + '/' + f'{time.time():.3}' + '_UNet_'+str(num_of_epochs)+'_epochs_simA_theoryB' + '.pth'))
    path_results = os.path.join(os.getcwd() + "/results/")

    plt.figure()
    plt.semilogy(np.arange(num_of_epochs), train_losses,label="train")
    plt.semilogy(np.arange(num_of_epochs), validation_losses,label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(
        os.path.join(path_results + f'/MSE_Loss_31122025_SmartMagDL2_batch_size_{batch_size}_2.png'),
        bbox_inches='tight', pad_inches=0, dpi=300)
if __name__=='__main__':
    train()
    print('done!')
