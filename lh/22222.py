# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torchvision.transforms.v2.functional import resize
from torchplus.models import ResNet50FE_Dim, resnet50fe
from torchplus.utils import Init
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
    model.eval()
    model.cpu()
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


def dynamic_quantize_aeinversion(model):
    print("  Dynamic quantizing AEInversion...")
    model.eval()
    model.cpu()
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.ConvTranspose2d, nn.Linear}, dtype=torch.qint8
    )
    print("    AEInversion dynamic quantization done")
    return model_quantized


if __name__ == "__main__":
    batch_size = 16
    root_dir = "./log/paper5w/logZZPMAIN.22222mixed_quantized"
    
    target_feature_pkl = "/data/code/jssnu005/two/Mapping-MIA/log/paper5w/logZZPMAIN.resnetface/Log_2026Apr11_19-10-14_supervisor-G5500-V7_main FaceScrub resnet50/feature_extractor_48.pkl"
    ae_inv_pkl = "/data/code/jssnu005/two/Mapping-MIA/log/paper5w/logZZPMAIN.aeface/Model_2026Apr11_22-41-52_supervisor-G5500-V7_FaceScrub ae 8192 norm 128pix ds2/myinversion_100.pkl"
    newfe_pkl = "/data/code/jssnu005/two/Mapping-MIA/log/paper5w/logZZPMAIN.nattackface.resnet/Model_2026Apr12_12-54-29_supervisor-G5500-V7_FaceScrub nattack norm resnet50 newfe 128pix cossim ds2/newfe_100.pkl"
    
    dataset_dir = "/data/code/jssnu005/two/Mapping-MIA/data/FaceScrub/"
    h, w = 224, 224

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        sep=False,
        backup_filename=__file__,
        tensorboard=True,
        comment="mixed quantization + individual compressed images",
    )
    
    device = torch.device("cpu")
    print(f"Using device: {device}")

    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([
        Resize((h, w)),
        ToImage(),
        ToDtype(torch.float, scale=True),
    ])

    ds = ImageFolder(root=dataset_dir, transform=transform)
    ds_len = len(ds)
    priv_ds, aux_ds = random_split(ds, [ds_len * 1 // 2, ds_len - ds_len * 1 // 2])
    aux_ds_len = len(aux_ds)
    aux_train_ds, aux_test_ds = random_split(
        aux_ds, [aux_ds_len * 6 // 7, aux_ds_len - aux_ds_len * 6 // 7]
    )
    train_ds = aux_train_ds
    test_ds = aux_test_ds

    temp_dl = DataLoader(
        dataset=train_ds, batch_size=32, shuffle=True,
        num_workers=data_workers, drop_last=False, pin_memory=False,
    )
    
    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_workers, drop_last=False, pin_memory=False,
    )
    test_dl = DataLoader(
        dataset=test_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_workers, drop_last=False, pin_memory=False,
    )
    priv_dl = DataLoader(
        dataset=priv_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_workers, drop_last=False, pin_memory=False,
    )

    print("\n" + "="*50)
    print("Loading models...")
    print("="*50)
    
    target_feature_extractor = resnet50fe().train(False).to(device)
    target_feature_extractor.load_state_dict(
        torch.load(open(target_feature_pkl, "rb"), map_location=device)
    )
    print("  Loaded target model")
    
    test_im = torch.randn(1, 3, 224, 224).to(device)
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

    # 提取特征用于校准
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

    # 静态量化 NewFE
    print("\n" + "="*50)
    print("Quantizing NewFE (Static)...")
    print("="*50)
    newfe_quantized = static_quantize_newfe(newfe, calib_dl)

    # 动态量化 AEInversion
    print("\n" + "="*50)
    print("Quantizing AEInversion (Dynamic)...")
    print("="*50)
    aeinversion_quantized = dynamic_quantize_aeinversion(aeinversion)

    # 计算模型大小
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
    int8_size_ae = get_model_size_mb(aeinversion_quantized.cpu())
    print(f"AEInversion: FP32={fp32_size_ae:.2f}MB, INT8={int8_size_ae:.2f}MB, ratio={fp32_size_ae/int8_size_ae:.1f}x")
    
    total_fp32 = fp32_size_newfe + fp32_size_ae
    total_int8 = int8_size_newfe + int8_size_ae
    print(f"Total: FP32={total_fp32:.2f}MB, INT8={total_int8:.2f}MB, ratio={total_fp32/total_int8:.1f}x")

    # 导出结果（保持文件夹结构，输出所有图片）
    print("\n" + "="*50)
    print("Exporting results (all images, compressed)...")
    print("="*50)
    
    JPEG_QUALITY = 50  # JPEG质量
    
    target_feature_extractor = target_feature_extractor.to(device)
    newfe_quantized = newfe_quantized.to(device)
    aeinversion_quantized = aeinversion_quantized.to(device)
    
    def process_and_save(dataloader, dataset_name):
        print(f"\n  Processing {dataset_name}...")
        
        # 创建目录
        input_dir = os.path.join(log_dir, dataset_name, "input")
        output_dir = os.path.join(log_dir, dataset_name, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        batch_idx = 0
        total_images = 0
        
        for i, (im, label) in enumerate(dataloader):
            im = im.to(device)
            im2 = resize(im, 128, antialias=False)
            target_feature = target_feature_extractor(im)
            target_feature = newfe_quantized(target_feature)
            target_feature = F.normalize(target_feature, dim=1)
            rim = aeinversion_quantized(target_feature)
            
            bs = rim.shape[0]
            for j in range(bs):
                # 保存输入图片（PNG格式，保持质量）
                input_path = os.path.join(input_dir, f"input_{batch_idx}_{j}.png")
                input_np = (im2[j].cpu().numpy() * 255).astype(np.uint8)
                input_np = np.transpose(input_np, (1, 2, 0))
                Image.fromarray(input_np).save(input_path)
                
                # 保存输出图片（JPEG压缩）
                output_path = os.path.join(output_dir, f"output_{batch_idx}_{j}.jpg")
                save_compressed_image(rim[j], output_path, quality=JPEG_QUALITY)
                
                total_images += 1
            
            batch_idx += 1
            
            # 每处理10个batch打印一次进度
            if batch_idx % 10 == 0:
                print(f"    Processed {batch_idx} batches, {total_images} images")
        
        print(f"    Completed {dataset_name}: {total_images} images saved")
        return total_images
    
    # 处理所有三个数据集
    train_count = process_and_save(train_dl, "train")
    test_count = process_and_save(test_dl, "test")
    priv_count = process_and_save(priv_dl, "priv")
    
    print(f"\n  Total images saved: {train_count + test_count + priv_count}")

    # 保存量化后的模型
    torch.save(newfe_quantized.cpu().state_dict(), f"{log_dir}/newfe_static_quantized.pkl")
    torch.save(aeinversion_quantized.cpu().state_dict(), f"{log_dir}/aeinversion_dynamic_quantized.pkl")
    
    writer.close()
    
    print("\n" + "="*50)
    print("Mixed Quantization + Full Output Completed!")
    print(f"Results saved to: {log_dir}")
    print(f"  - NewFE: Static Quantization (INT8)")
    print(f"  - AEInversion: Dynamic Quantization (INT8)")
    print(f"  - Output images: JPEG quality={JPEG_QUALITY}")
    print("")
    print("="*50)
EOF