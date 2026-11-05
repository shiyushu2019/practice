import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import *
from tqdm import tqdm
from piq import SSIMLoss
from torchplus.models import resnet50fe
from torchplus.utils import (
    Init,
    save_image2,
)

if __name__ == "__main__":
    batch_size = 16
    seed = 9970
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    root_dir = "./checkpoints/testmain"
    target_feature_pkl = "./checkpoints/resnetface/feature_extractor_48.pkl"
    newfe_pkl = "./checkpoints/mixed_quantized/newfe_dynamic_quantized.pkl"
    ae_inv_pkl = "./checkpoints/mixed_quantized/aeinversion_static_quantized.pkl"
    dataset_dir = "/data1/dataset/jssnu005/sys/lh/FaceScrub/"
    h = 128
    w = 128

    output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = root_dir
    os.makedirs(log_dir, exist_ok=True)
    data_workers = 2

    from datasplit import(
            test_ds
    )
    test_ds_len = len(test_ds)
    print(test_ds_len)
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

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

    class AEInversion(nn.Module):
        def __init__(self, in_channels):
            super(AEInversion, self).__init__()
            self.in_channels = in_channels
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()
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
            x = self.quant(x)
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
            return self.dequant(x)

    target_feature_extractor = resnet50fe().train(False).to(output_device)
    assert os.path.exists(target_feature_pkl)
    target_feature_extractor.load_state_dict(
        torch.load(open(target_feature_pkl, "rb"), map_location=output_device)
    )

    test_im = torch.randn(1, 3, h, w).to(output_device)
    with torch.no_grad():
        actual_dim = target_feature_extractor(test_im).shape[-1]

    newfe = QuantizableNewFE(actual_dim, 8192).train(False).cpu()
    newfe = torch.ao.quantization.quantize_dynamic(
        newfe, {nn.Linear}, dtype=torch.qint8
    )
    assert os.path.exists(newfe_pkl)
    newfe.load_state_dict(
        torch.load(open(newfe_pkl, "rb"), map_location="cpu")
    )

    aeinversion = AEInversion(8192).train(False).cpu()
    aeinversion.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    for module in aeinversion.modules():
        if isinstance(module, nn.ConvTranspose2d):
            module.qconfig = torch.ao.quantization.QConfig(
                activation=aeinversion.qconfig.activation,
                weight=torch.ao.quantization.default_weight_observer,
            )
    torch.ao.quantization.prepare(aeinversion, inplace=True)
    torch.ao.quantization.convert(aeinversion, inplace=True)
    assert os.path.exists(ae_inv_pkl)
    aeinversion.load_state_dict(
        torch.load(open(ae_inv_pkl, "rb"), map_location="cpu")
    )

    target_feature_extractor.requires_grad_(False)
    newfe.requires_grad_(False)
    aeinversion.requires_grad_(False)

    # Evaluate cosine similarity on the full shared test set.
    cosine_sum = 0.0
    ssim_sum = 0.0
    evaluated_samples = 0
    with torch.no_grad():
        for im, _ in tqdm(test_dl, desc="evaluate test", mininterval=1.0):
            im = im.to(output_device)
            im2 = im
            im2=im2.cpu()
            target_feature8192 = target_feature_extractor.forward(im).cpu()
            target_feature8192 = newfe.forward(target_feature8192)
            target_feature8192 = F.normalize(target_feature8192)
            rim = aeinversion.forward(target_feature8192)
            cosine_sum += F.cosine_similarity(
                im2.flatten(start_dim=1), rim.flatten(start_dim=1), dim=1
            ).sum().item()
            ssim_sum += (1 - SSIMLoss()(rim, im2)).item() * im.size(0)
            evaluated_samples += im.size(0)

    print("test average cosine similarity", cosine_sum / evaluated_samples)
    print("test average SSIM", ssim_sum / evaluated_samples)

    sample_count = min(100, test_ds_len)
    sample_indices = torch.randperm(
        test_ds_len, generator=torch.Generator().manual_seed(seed + 1)
    )[:sample_count].tolist()
    sample_dl = DataLoader(
        dataset=Subset(test_ds, sample_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )
    pair_dir = os.path.join(log_dir, "test_pairs")
    os.makedirs(pair_dir, exist_ok=True)

    # Save original/reconstruction pairs side by side for visual inspection.
    with torch.no_grad():
        image_index = 0
        for im, _ in tqdm(sample_dl, desc="save test pairs", mininterval=1.0):
            im = im.to(output_device)
            im2 =im
            im2=im2.cpu()
            target_feature8192 = target_feature_extractor.forward(im).cpu()
            target_feature8192 = newfe.forward(target_feature8192)
            target_feature8192 = F.normalize(target_feature8192)
            rim = aeinversion.forward(target_feature8192)
            pairs = torch.cat((im2, rim), dim=3)
            for pair in pairs:
                save_image2(pair.detach(), os.path.join(pair_dir, f"{image_index:03d}.png"))
                image_index += 1
