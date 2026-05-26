import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels,
                      3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.3))
        layers += [
            nn.Conv2d(out_channels, out_channels,
                      3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1      = DoubleConv(3, 64)
        self.pool1      = nn.MaxPool2d(2)
        self.down2      = DoubleConv(64, 128)
        self.pool2      = nn.MaxPool2d(2)
        self.down3      = DoubleConv(128, 256)
        self.pool3      = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512,
                          use_dropout=True)
        self.up1        = nn.ConvTranspose2d(
                          512, 256, 2, stride=2)
        self.conv1      = DoubleConv(512, 256)
        self.up2        = nn.ConvTranspose2d(
                          256, 128, 2, stride=2)
        self.conv2      = DoubleConv(256, 128)
        self.up3        = nn.ConvTranspose2d(
                          128, 64, 2, stride=2)
        self.conv3      = DoubleConv(128, 64)
        self.final      = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        x  = self.conv1(torch.cat([self.up1(x4), x3], 1))
        x  = self.conv2(torch.cat([self.up2(x),  x2], 1))
        x  = self.conv3(torch.cat([self.up3(x),  x1], 1))
        return self.final(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(
            self.relu(self.W_g(g) + self.W_x(x)))


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1      = DoubleConv(3, 64)
        self.pool1      = nn.MaxPool2d(2)
        self.down2      = DoubleConv(64, 128)
        self.pool2      = nn.MaxPool2d(2)
        self.down3      = DoubleConv(128, 256)
        self.pool3      = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512,
                          use_dropout=True)
        self.up1        = nn.ConvTranspose2d(
                          512, 256, 2, stride=2)
        self.att1       = AttentionGate(256, 256, 128)
        self.conv1      = DoubleConv(512, 256)
        self.up2        = nn.ConvTranspose2d(
                          256, 128, 2, stride=2)
        self.att2       = AttentionGate(128, 128, 64)
        self.conv2      = DoubleConv(256, 128)
        self.up3        = nn.ConvTranspose2d(
                          128, 64, 2, stride=2)
        self.att3       = AttentionGate(64, 64, 32)
        self.conv3      = DoubleConv(128, 64)
        self.final      = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        g  = self.up1(x4)
        x  = self.conv1(torch.cat([g, self.att1(g, x3)], 1))
        g  = self.up2(x)
        x  = self.conv2(torch.cat([g, self.att2(g, x2)], 1))
        g  = self.up3(x)
        x  = self.conv3(torch.cat([g, self.att3(g, x1)], 1))
        return self.final(x)