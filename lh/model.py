import torch
import torch.nn as nn


class FeatureExtracter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.mp3 = nn.MaxPool2d(2, 2)
        self.mp4 = nn.MaxPool2d(2, 2)
        self.mp5 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.mp1(self.bn1(self.conv1(x))))
        x = self.relu2(self.mp2(self.bn2(self.conv2(x))))
        x = self.relu3(self.mp3(self.bn3(self.conv3(x))))
        x = self.relu4(self.mp4(self.bn4(self.conv4(x))))
        x = self.relu5(self.mp5(self.bn5(self.conv5(x))))
        return x.view(-1, 8192)


AEFeatureExtracter = FeatureExtracter


class NewFE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2048)
        self.fc2 = nn.Linear(2048, out_dim)
        self.lrelu1 = nn.LeakyReLU()
        self.lrelu2 = nn.LeakyReLU()

    def _project(self, x):
        x = self.lrelu1(self.fc1(x))
        return self.lrelu2(self.fc2(x))

    def forward(self, x):
        return self._project(x)


class QuantizableNewFE(NewFE):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self._project(self.quant(x)))


class Inversion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.deconv1 = nn.ConvTranspose2d(in_channels, 512, 4, 1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.sigmod = nn.Sigmoid()

    def _decode(self, x):
        x = x.view(-1, self.in_channels, 1, 1)
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.relu4(self.bn4(self.deconv4(x)))
        x = self.relu5(self.bn5(self.deconv5(x)))
        return self.sigmod(self.deconv6(x))

    def forward(self, x):
        return self._decode(x)


class QuantizableInversion(Inversion):
    def __init__(self, in_channels):
        super().__init__(in_channels)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self._decode(self.quant(x)))


AEInversion = Inversion
AEInversionForStatic = QuantizableInversion


class CLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)
