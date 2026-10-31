import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Subset
from torchvision.datasets import *
from torchvision.transforms.v2 import *
from torchvision.transforms.v2.functional import *
from tqdm import tqdm
from torchplus.utils import Init, class_split, save_excel, save_image2
from torchplus.datasets import PreProcessFolder
from piq import SSIMLoss

# 主训练入口：使用 FaceScrub 数据，训练一个“特征编码 -> 图像重建”的自编码器网络。
if __name__ == "__main__":
    batch_size = 256
    train_epoches = 100
    log_epoch = 4
    h = 128
    w = 128


    seed = 9970
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset_dir = "/data1/dataset/jssnu005/sys/lh/FaceScrub"
    output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "./checkpoints/aeface"
    log_dir = "./checkpoints/aeface"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "output"), exist_ok=True)

    data_workers = 2

    transform = Compose(
        [
            Resize((h, w)),
            RandomHorizontalFlip(),
            ToImage(),
            ToDtype(torch.float, scale=True),
        ]
    )
    
    ds = ImageFolder(root=dataset_dir, transform=transform)

    ds_len = len(ds)
    test_size = ds_len // 10
    train_size = ds_len - test_size
    third = train_size // 3
    idx_part1 = list(range(0, third))                 # resnetface.py
    idx_part2 = list(range(third, 2 * third))         # 当前aeface.py
    idx_part3 = list(range(2 * third, train_size))    # nattackface.py
    idx_test = list(range(train_size, ds_len))        # 全局测试集

    train_ds=Subset(ds, idx_part2)
    test_ds=Subset(ds, idx_test)
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
        pin_memory=True,
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)
    print(train_ds_len)
    print(test_ds_len)

    train_dl1 = DataLoader(         # 用训练集评估的时候使用
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    train_dl_len = len(train_dl)
    test_dl_len = len(test_dl)

    # 特征提取器：通过一系列卷积和池化，把输入图像压缩成 8192 维隐特征，
    # 后续再由反演器还原出近似图像。这里的设计目标是保留图像结构信息。
    class FeatureExtracter(nn.Module):
        def __init__(self):
            super(FeatureExtracter, self).__init__()
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
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.mp1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.mp2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.mp3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.mp4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.mp5(x)
            x = self.relu5(x)
            x = x.view(-1, 8192)
            return x

    class Inversion(nn.Module):
        def __init__(self, in_channels):
            super(Inversion, self).__init__()
            self.in_channels = in_channels
            self.deconv1 = nn.ConvTranspose2d(self.in_channels, 512, 4, 1)
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

        def forward(self, x):
            x = x.view(-1, self.in_channels, 1, 1)
            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.deconv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.deconv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            x = self.deconv6(x)
            x = self.sigmod(x)
            return x

    feature_extractor = FeatureExtracter().train(True).to(output_device)
    myinversion = Inversion(8192).train(True).to(output_device)

    optimizer_fe = optim.Adam(
        feature_extractor.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    optimizer = optim.Adam(
        myinversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    # 训练主循环：先由特征提取器得到隐表示，再由反演器重建图像；
    # 使用 SSIM 作为重建质量衡量指标，并定期保存中间结果和模型权重。
    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        for i, (im, label) in enumerate(tqdm(train_dl, desc=f"epoch {epoch_id} ")):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            optimizer.zero_grad()
            optimizer_fe.zero_grad()
            feature8192 = feature_extractor.forward(im)
            feature8192 = F.normalize(feature8192)
            rim = myinversion.forward(feature8192)
            ssim = SSIMLoss()(rim, im)
            loss = ssim
            loss.backward()
            optimizer_fe.step()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            print("loss", loss.item(), epoch_id)
            print("ssim", ssim.item(), epoch_id)
            save_image2(im.detach(), f"{log_dir}/input/{epoch_id}.png")
            save_image2(rim.detach(), f"{log_dir}/output/{epoch_id}.png")
            with open(
                os.path.join(model_dir, f"feature_extractor_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(feature_extractor.state_dict(), f)
            with open(
                os.path.join(model_dir, f"myinversion_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(myinversion.state_dict(), f)

            with torch.no_grad():
                feature_extractor.eval()
                myinversion.eval()
                r = 0
                ssimloss = 0
                for i, (im, label) in enumerate(tqdm(train_dl1, desc=f"train")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    feature8192 = F.normalize(feature8192)
                    rim = myinversion.forward(feature8192)
                    ssim = SSIMLoss()(rim, im)
                    ssimloss += ssim

                ssimlossavg = ssimloss / r
                print("train ssim", ssimlossavg.item(), epoch_id)

                r = 0
                ssimloss = 0
                for i, (im, label) in enumerate(tqdm(test_dl, desc=f"test")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    feature8192 = F.normalize(feature8192)
                    rim = myinversion.forward(feature8192)
                    ssim = SSIMLoss()(rim, im)
                    ssimloss += ssim

                ssimlossavg = ssimloss / r
                print("test ssim", ssimlossavg.item(), epoch_id)
