import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchplus.utils import save_image2
from piq import SSIMLoss
from utils.model import FeatureExtracter, Inversion

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

    from utils.datasplit import (
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
    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch",mininterval=1.0,):
        feature_extractor.train()
        myinversion.train()
        epoch_loss = 0.0
        epoch_ssim = 0.0
        epoch_samples = 0
        for i, (im, label) in enumerate(tqdm(train_dl, desc=f"epoch {epoch_id} ", mininterval=1.0)):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            optimizer.zero_grad()
            optimizer_fe.zero_grad()
            feature8192 = feature_extractor.forward(im)
            feature8192 = F.normalize(feature8192)
            rim = myinversion.forward(feature8192)
            ssim = SSIMLoss()(
                rim,
                F.interpolate(im, size=(128, 128), mode="bilinear", align_corners=False),
            )
            loss = ssim
            loss.backward()
            optimizer_fe.step()
            optimizer.step()
            batch_samples = label.size(0)
            epoch_loss += loss.detach().item() * batch_samples
            epoch_ssim += ssim.detach().item() * batch_samples
            epoch_samples += batch_samples

        if epoch_id % log_epoch == 0:
            print("loss", epoch_loss / epoch_samples, epoch_id)
            print("ssim", 1 - epoch_ssim / epoch_samples, epoch_id)
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
                train_samples = 0
                ssimloss = 0.0
                for i, (im, label) in enumerate(tqdm(train_dl1, desc=f"train", mininterval=1.0)):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    feature8192 = F.normalize(feature8192)
                    rim = myinversion.forward(feature8192)
                    ssim = SSIMLoss()(
                        rim,
                        F.interpolate(im, size=(128, 128), mode="bilinear", align_corners=False),
                    )
                    batch_samples = label.size(0)
                    ssimloss += ssim.item() * batch_samples
                    train_samples += batch_samples

                ssimlossavg = ssimloss / train_samples
                print("train ssim", 1 - ssimlossavg, epoch_id)

                test_samples = 0
                ssimloss = 0.0
                for i, (im, label) in enumerate(tqdm(test_dl, desc=f"test", mininterval=1.0)):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    feature8192 = F.normalize(feature8192)
                    rim = myinversion.forward(feature8192)
                    ssim = SSIMLoss()(
                        rim,
                        F.interpolate(im, size=(128, 128), mode="bilinear", align_corners=False),
                    )
                    batch_samples = label.size(0)
                    ssimloss += ssim.item() * batch_samples
                    test_samples += batch_samples

                ssimlossavg = ssimloss / test_samples
                print("test ssim", 1 - ssimlossavg, epoch_id)
