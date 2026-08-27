"""统一训练脚本 —— CBAM / Focal Loss 真正生效的版本。

用法:
    python train.py --variant baseline          # 纯 YOLO11n (对照)
    python train.py --variant cbam              # YOLO11n + CBAM 注意力 (真正注入)
    python train.py --variant cf                # YOLO11n + CBAM + Focal Loss (真正接入)
    python train.py --variant cf --batch 32 --epochs 50 --device 4

训练完成后 best 权重统一保存到 weights/<variant>.pt。

说明: 之前版本的 CBAM 在原训练脚本里对 model.model 的原地修改会被
ultralytics 在 model.train() 内部按 yaml 重建模型时丢弃, 导致 CBAM 从未生效。
本脚本通过自定义 DetectionTrainer.get_model(), 在模型构建阶段就把 CBAM 注入
backbone、把 Focal Loss 绑定到模型 criterion, 从而保证二者真正参与前向/反向。
"""
import argparse
import os
import shutil
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from cbam import CBAM  # noqa: E402
from focal_loss import FocalDetectionModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")
PRETRAINED = os.path.join(WEIGHTS_DIR, "yolo11n.pt")


def inject_cbam(model):
    """在 backbone 的 SPPF 层前插入 CBAM, 返回被替换的层索引。"""
    layers = list(model.model)
    sppf_idx = next(i for i, l in enumerate(layers) if "SPPF" in str(l))
    sppf = layers[sppf_idx]
    # SPPF 的输入通道是 cv1 (c1), 不是 cv2 (cv2 是 4 条池化分支拼接后的 512 通道)
    in_channels = sppf.cv1.conv.in_channels
    new_layer = nn.Sequential(CBAM(in_channels, reduction=16), sppf)
    # 复制原 SPPF 层的 f/i 属性, 否则 ultralytics 的 _predict_once 访问 m.f/m.i 会报错
    new_layer.f = getattr(sppf, "f", -1)
    new_layer.i = getattr(sppf, "i", sppf_idx)
    model.model[sppf_idx] = new_layer
    return sppf_idx


class CBAMFocalTrainer(DetectionTrainer):
    """自定义 trainer: 在 get_model 阶段注入 CBAM / Focal Loss, 绕过模型重建。"""

    def get_model(self, cfg=None, weights=None, verbose=True):
        variant = getattr(self, "variant", "baseline")
        if variant == "cf":
            gamma = float(getattr(self, "focal_gamma", 2.0))
            alpha = float(getattr(self, "focal_alpha", 0.25))
            model = self.set_model_names_for_load(
                FocalDetectionModel(cfg, ch=self.data["channels"], nc=self.data["nc"],
                                    verbose=verbose, gamma=gamma, alpha=alpha)
            )
            print(f"  [注入] Focal Loss 已接入 criterion (gamma={gamma}, alpha={alpha})")
        else:
            model = self.set_model_names_for_load(
                DetectionModel(cfg, ch=self.data["channels"], nc=self.data["nc"], verbose=verbose)
            )
        if weights:
            model.load(weights)

        if variant in ("cbam", "cf"):
            idx = inject_cbam(model)
            print(f"  [注入] CBAM 已插入 backbone 第 {idx} 层 (SPPF 前)")
        return model


def verify(path):
    """加载保存的权重, 检查 CBAM 是否真的在模型里。"""
    m = YOLO(path)
    types = {type(x).__name__ for x in m.model.modules()}
    has_cbam = any("CBAM" in t for t in types)
    n_params = sum(p.numel() for p in m.model.parameters())
    print(f"  [校验] {os.path.basename(path)}: 含 CBAM 模块={has_cbam}, 参数量={n_params}")
    return has_cbam


def main():
    ap = argparse.ArgumentParser(description="统一训练脚本 (CBAM/Focal 真正生效)")
    ap.add_argument("--variant", default="baseline", choices=["baseline", "cbam", "cf"])
    ap.add_argument("--device", default="4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.25)
    args = ap.parse_args()

    overrides = {
        "model": PRETRAINED,
        "data": "bccd.yaml",
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": 640,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "patience": 100,
        "seed": 0,
        "deterministic": True,
        "amp": True,
        "close_mosaic": 10,
        "device": args.device,
        "project": WEIGHTS_DIR,
        "name": f"_{args.variant}_train",
        "exist_ok": True,
        "plots": True,
    }

    print("=" * 70)
    print(f"训练变体: {args.variant}   batch={args.batch}  epochs={args.epochs}  device={args.device}")
    print("=" * 70)

    trainer = CBAMFocalTrainer(overrides=overrides, _callbacks=None)
    trainer.variant = args.variant
    trainer.focal_gamma = args.gamma
    trainer.focal_alpha = args.alpha
    trainer.train()

    # 统一保存 best 权重
    out_dir = os.path.join(WEIGHTS_DIR, f"_{args.variant}_train")
    best = os.path.join(out_dir, "weights", "best.pt")
    dst = os.path.join(WEIGHTS_DIR, f"{args.variant}.pt")
    shutil.copy2(best, dst)
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"✅ 训练完成, 权重已保存: {dst}")

    verify(dst)

    # 在测试集上评估
    metrics = YOLO(dst).val(data="bccd.yaml", split="test", device=args.device,
                           plots=False, save=False, verbose=False)
    print(f"  mAP@0.5={metrics.box.map50:.4f}  mAP@0.5:0.95={metrics.box.map:.4f}  "
          f"P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}")

    shutil.rmtree(os.path.join(HERE, "runs"), ignore_errors=True)


if __name__ == "__main__":
    main()
