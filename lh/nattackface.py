import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import *
from tqdm import tqdm
from torchplus.utils import Init, save_image2
from torchplus.models import ResNet50FE_Dim, resnet50fe
from piq import SSIMLoss

if __name__ == "__main__":
    batch_size = 256
    train_epoches = 100
    log_epoch = 4
    seed = 9970
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    root_dir = "./checkpoints/nattackface"
    target_feature_pkl = "./checkpoints/resnetface/feature_extractor_48.pkl"
    ae_feature_pkl = "./checkpoints/aeface/feature_extractor_100.pkl"
    ae_inv_pkl = "./checkpoints/aeface/myinversion_100.pkl"
    dataset_dir = "/data1/dataset/jssnu005/sys/lh/FaceScrub"
    h = 128
    w = 128

    output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = root_dir
    model_dir = root_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "output"), exist_ok=True)
    data_workers = 2

    from datasplit import(
            part2_ds,
            test_ds
    )
    train_ds=part2_ds
    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)
    print(train_ds_len)
    print(test_ds_len)

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

    train_dl1 = DataLoader(      # 训练集评估时使用
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    train_dl_len = len(train_dl)
    test_dl_len = len(test_dl)

    # 旧编码器结构：用卷积堆叠提取图像特征，最终压成 8192 维表示，
    # 和后续反演器中的隐空间保持一致，以便进行对齐训练。
    class AEFeatureExtracter(nn.Module):
        def __init__(self):
            super(AEFeatureExtracter, self).__init__()
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

    class NewFE(nn.Module):
        def __init__(self, in_dim, out_dim) -> None:
            super(NewFE, self).__init__()
            self.fc1 = nn.Linear(in_dim, 2048)
            self.fc2 = nn.Linear(2048, out_dim)
            self.lrelu1 = nn.LeakyReLU()
            self.lrelu2 = nn.LeakyReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.lrelu1(x)
            x = self.fc2(x)
            x = self.lrelu2(x)
            # x = self.fc3(x)
            return x

    class AEInversion(nn.Module):
        def __init__(self, in_channels):
            super(AEInversion, self).__init__()
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

    target_feature_extractor = resnet50fe().train(False).to(output_device)
    ae_feature_extractor = AEFeatureExtracter().train(False).to(output_device)
    newfe = NewFE(ResNet50FE_Dim, 8192).train(True).to(output_device)
    aeinversion = AEInversion(8192).train(False).to(output_device)

    optimizer = optim.Adam(
        newfe.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    assert os.path.exists(target_feature_pkl)
    target_feature_extractor.load_state_dict(
        torch.load(open(target_feature_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(ae_feature_pkl)
    ae_feature_extractor.load_state_dict(
        torch.load(open(ae_feature_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(ae_inv_pkl)
    aeinversion.load_state_dict(
        torch.load(open(ae_inv_pkl, "rb"), map_location=output_device)
    )

    target_feature_extractor.requires_grad_(False)
    ae_feature_extractor.requires_grad_(False)
    aeinversion.requires_grad_(False)

    # 训练主循环：将目标模型特征投影到新空间，并与 AE 隐表示做对齐；
    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch", mininterval=1.0):
        newfe.train()
        epoch_loss = 0.0
        epoch_ssim = 0.0
        epoch_cossim = 0.0
        epoch_samples = 0
        for i, (im, label) in enumerate(tqdm(train_dl, desc=f"epoch {epoch_id} ", mininterval=1.0)):
            im = im.to(output_device)
            im2 = im
            label = label.to(output_device)
            bs, c, h, w = im.shape
            optimizer.zero_grad()
            target_feature = target_feature_extractor.forward(im)
            ae_feature8192 = ae_feature_extractor.forward(im2)
            target_feature = newfe.forward(target_feature)
            target_feature = F.normalize(target_feature)
            ae_feature8192 = F.normalize(ae_feature8192)
            cossim = torch.mean(F.cosine_similarity(target_feature, ae_feature8192))
            rim = aeinversion.forward(target_feature)
            ssim = SSIMLoss()(rim, im2)
            loss = 1 - cossim
            loss.backward()
            optimizer.step()
            batch_samples = label.size(0)
            epoch_loss += loss.detach().item() * batch_samples
            epoch_ssim += ssim.detach().item() * batch_samples
            epoch_cossim += cossim.detach().item() * batch_samples
            epoch_samples += batch_samples

        if epoch_id % log_epoch == 0:
            print("loss", epoch_loss / epoch_samples, epoch_id)
            print("ssim", epoch_ssim / epoch_samples, epoch_id)
            print("cossim", epoch_cossim / epoch_samples, epoch_id)
            save_image2(im.detach(), f"{log_dir}/input/{epoch_id}.png")
            save_image2(rim.detach(), f"{log_dir}/output/{epoch_id}.png")
            with open(os.path.join(model_dir, f"newfe_{epoch_id}.pkl"), "wb") as f:
                torch.save(newfe.state_dict(), f)

            with torch.no_grad():
                newfe.eval()
                train_samples = 0
                ssimloss = 0.0
                for i, (im, label) in enumerate(tqdm(train_dl1, desc=f"train", mininterval=1.0)):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    im2 = resize(im, 128, antialias=False)
                    bs, c, h, w = im.shape
                    target_feature = target_feature_extractor.forward(im)
                    target_feature = newfe.forward(target_feature)
                    target_feature = F.normalize(target_feature)
                    rim = aeinversion.forward(target_feature)
                    ssim = SSIMLoss()(rim, im2)
                    batch_samples = label.size(0)
                    ssimloss += ssim.item() * batch_samples
                    train_samples += batch_samples

                ssimlossavg = ssimloss / train_samples
                print("train ssim", ssimlossavg, epoch_id)

                test_samples = 0
                ssimloss = 0.0
                for i, (im, label) in enumerate(tqdm(test_dl, desc=f"test", mininterval=1.0)):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    im2 = resize(im, 128, antialias=False)
                    bs, c, h, w = im.shape
                    target_feature = target_feature_extractor.forward(im)
                    target_feature = newfe.forward(target_feature)
                    target_feature = F.normalize(target_feature)
                    rim = aeinversion.forward(target_feature)
                    ssim = SSIMLoss()(rim, im2)
                    batch_samples = label.size(0)
                    ssimloss += ssim.item() * batch_samples
                    test_samples += batch_samples

                ssimlossavg = ssimloss / test_samples
                print("test ssim", ssimlossavg, epoch_id)
