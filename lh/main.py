# -*- coding: utf-8 -*-
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.datasets import ImageFolder
from torchplus.models import resnet50fe
from PIL import Image
import numpy as np


class QuantizableNewFE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.fc1 = nn.Linear(in_dim, 2048)
        self.fc2 = nn.Linear(2048, out_dim)
        self.lrelu1 = nn.LeakyReLU()
        self.lrelu2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.lrelu1(x)
        x = self.fc2(x)
        x = self.lrelu2(x)
        x = self.dequant(x)
        return x


class AEInversionForDynamic(nn.Module):
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
        self.sigmoid = nn.Sigmoid()

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
        x = self.sigmoid(x)
        return x


def save_compressed_image(tensor_img, save_path, quality=50):
    img_np = (tensor_img.cpu().numpy() * 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))
    pil_img = Image.fromarray(img_np)
    pil_img.save(save_path, format='JPEG', quality=quality, optimize=True)
    return os.path.getsize(save_path)


def static_quantize_newfe(model, calib_dataloader):
    print("  Static quantizing NewFE...")
    model = copy.deepcopy(model).eval().cpu()
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    torch.ao.quantization.prepare(model, inplace=True)
    print("    Calibrating...")
    with torch.no_grad():
        for batch in calib_dataloader:
            x = batch[0]
            model(x)
    torch.ao.quantization.convert(model, inplace=True)
    print("    NewFE static quantization done")
    return model


def quantize_aeinversion_fp16(model, device):
    print("  Converting AEInversion to FP16...")
    if device.type == "cuda":
        return copy.deepcopy(model).eval().to(device=device, dtype=torch.float16)
    print("  CUDA unavailable; keeping AEInversion in FP32.")
    return copy.deepcopy(model).eval().to(device=device)


# 该脚本用于加载训练好的特征提取器和反演器，
# 通过对 NewFE / AEInversion 做量化并比较模型体积与重建效果。
if __name__ == "__main__":
    seed = 9970
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch_size = 16
    root_dir = "./checkpoints/mixed_quantized"
    target_feature_pkl = "./checkpoints/resnetface/feature_extractor_48.pkl"
    ae_inv_pkl = "./checkpoints/aeface/myinversion_100.pkl"
    newfe_pkl = "./checkpoints/nattackface/newfe_100.pkl"
    dataset_dir = "/data1/dataset/jssnu005/sys/lh/FaceScrub/"
    h, w = 128,128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = root_dir
    os.makedirs(log_dir, exist_ok=True)
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
    temp_dl = DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=False,
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    print("\n" + "="*50)
    print("Loading models...")
    print("="*50)

    # 加载目标模型、投影模块和解码器；其中目标特征提取器用于生成参考特征，
    # NewFE 与 AEInversion 则分别负责映射和重建。
    target_feature_extractor = resnet50fe().train(False).to(device)
    target_feature_extractor.load_state_dict(
        torch.load(open(target_feature_pkl, "rb"), map_location=device)
    )
    print("  Loaded target model")

    test_im = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        actual_dim = target_feature_extractor(test_im).shape[-1]
    print(f"  Actual feature dimension: {actual_dim}")

    newfe = QuantizableNewFE(actual_dim, 8192).train(False).to(device)
    newfe.load_state_dict(torch.load(open(newfe_pkl, "rb"), map_location=device))
    print("  Loaded newfe model")

    aeinversion = AEInversionForDynamic(8192).train(False).to(device)
    aeinversion.load_state_dict(
        torch.load(open(ae_inv_pkl, "rb"), map_location=device)
    )
    print("  Loaded aeinversion model")

    target_feature_extractor.requires_grad_(False)
    newfe.requires_grad_(False)
    aeinversion.requires_grad_(False)

    # 采样少量训练图像，提取其目标特征，用于后续量化校准过程。
    print("\n" + "="*50)
    print("Extracting features for calibration...")
    print("="*50)
    features_list = []
    target_feature_extractor.eval()
    with torch.no_grad():
        for i, (im, _) in enumerate(temp_dl):
            if i >= 30:
                break
            im = im.to(device)
            feat = target_feature_extractor(im)
            features_list.append(feat.cpu())
    features = torch.cat(features_list, dim=0)
    print(f"  Collected {features.shape[0]} features, dim={features.shape[1]}")

    calib_dl = DataLoader(TensorDataset(features), batch_size=32, shuffle=False)

    # 针对 NewFE 做静态量化：使用校准集统计量化参数，压缩模型权重。
    print("\n" + "="*50)
    print("Quantizing NewFE (Static)...")
    print("="*50)
    newfe_quantized = static_quantize_newfe(newfe, calib_dl)

    # 将 AEInversion 转为 FP16，降低权重和推理激活的存储占用。
    print("\n" + "="*50)
    print("Quantizing AEInversion (FP16)...")
    print("="*50)
    aeinversion_quantized = quantize_aeinversion_fp16(aeinversion, device)

    # 对比量化前后模型体积，观察压缩效果和总体节省比例。
    print("\n" + "="*50)
    print("Model size comparison...")
    print("="*50)

    def get_model_size_mb(model):
        torch.save(model.state_dict(), "/tmp/temp_model.pkl")
        size = os.path.getsize("/tmp/temp_model.pkl") / 1024 / 1024
        os.remove("/tmp/temp_model.pkl")
        return size

    fp32_size_newfe = get_model_size_mb(newfe.cpu())
    int8_size_newfe = get_model_size_mb(newfe_quantized.cpu())
    print(f"NewFE: FP32={fp32_size_newfe:.2f}MB, INT8={int8_size_newfe:.2f}MB, ratio={fp32_size_newfe/int8_size_newfe:.1f}x")

    fp32_size_ae = get_model_size_mb(aeinversion.cpu())
    reduced_size_ae = get_model_size_mb(aeinversion_quantized.cpu())
    ae_precision = "FP16" if device.type == "cuda" else "FP32"
    print(f"AEInversion: FP32={fp32_size_ae:.2f}MB, {ae_precision}={reduced_size_ae:.2f}MB, ratio={fp32_size_ae/reduced_size_ae:.1f}x")

    total_fp32 = fp32_size_newfe + fp32_size_ae
    total_reduced = int8_size_newfe + reduced_size_ae
    print(f"Total: FP32={total_fp32:.2f}MB, NewFE INT8 + AEInversion {ae_precision}={total_reduced:.2f}MB, ratio={total_fp32/total_reduced:.1f}x")

    # 导出重建结果图片并压缩保存。
    print("\n" + "="*50)
    print("Exporting results (all images, compressed)...")
    print("="*50)

    JPEG_QUALITY = 50  # JPEG 图片质量

    target_feature_extractor = target_feature_extractor.to(device)
    newfe_quantized = newfe_quantized.cpu()
    aeinversion_quantized = aeinversion_quantized.to(device)

    @torch.no_grad()
    def process_and_save(dataloader, dataset_name):
        print(f"\n  Processing {dataset_name}...")

        # 创建输入和输出目录。
        input_dir = os.path.join(log_dir, dataset_name, "input")
        output_dir = os.path.join(log_dir, dataset_name, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        batch_idx = 0
        total_images = 0

        for i, (im, label) in enumerate(dataloader):
            im = im.to(device)
            im2 = im
            target_features = target_feature_extractor(im).cpu()
            target_feature = newfe_quantized(target_features)
            target_feature = F.normalize(target_feature, dim=1)
            decoder_dtype = next(aeinversion_quantized.parameters()).dtype
            rim = aeinversion_quantized(target_feature.to(device=device, dtype=decoder_dtype))

            bs = rim.shape[0]
            for j in range(bs):
                # 保存输入图片为 PNG 格式。
                input_path = os.path.join(input_dir, f"input_{batch_idx}_{j}.png")
                input_np = (im2[j].cpu().numpy() * 255).astype(np.uint8)
                input_np = np.transpose(input_np, (1, 2, 0))
                Image.fromarray(input_np).save(input_path)

                # 保存重建图片为 JPEG 格式。
                output_path = os.path.join(output_dir, f"output_{batch_idx}_{j}.jpg")
                save_compressed_image(rim[j], output_path, quality=JPEG_QUALITY)

                total_images += 1

            batch_idx += 1

            # 每处理 10 个 batch 打印一次进度。
            if batch_idx % 10 == 0:
                print(f"    Processed {batch_idx} batches, {total_images} images")

        print(f"    Completed {dataset_name}: {total_images} images saved")
        return total_images

    # 分别处理训练集和测试集。
    test_count = process_and_save(test_dl, "test")

    print(f"\n  Total images saved: {test_count}")

    # 保存量化模型。
    torch.save(newfe_quantized.cpu().state_dict(), f"{log_dir}/newfe_static_quantized.pkl")
    torch.save(aeinversion_quantized.cpu().state_dict(), f"{log_dir}/aeinversion_dynamic_quantized.pkl")

    print("\n" + "="*50)
    print("Mixed Quantization + Full Output Completed!")
    print(f"Results saved to: {log_dir}")
    print(f"  - NewFE: Static Quantization (INT8)")
    print(f"  - AEInversion: {ae_precision}")
    print(f"  - Output images: JPEG quality={JPEG_QUALITY}")
    print("")
    print("="*50)
