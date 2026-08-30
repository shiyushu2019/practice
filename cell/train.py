"""统一训练脚本 —— 严格按实验文档实现 CBAM + Focal Loss。

用法:
    python train.py --variant baseline          # 纯 YOLOv8n (对照)
    python train.py --variant cbam              # YOLOv8n + CBAM (嵌入 backbone ConvModule)
    python train.py --variant cf                # YOLOv8n + CBAM + Focal Loss (alpha=0.25, gamma=2.0, 归一化)
    python train.py --variant cf --batch 32 --epochs 100
    python train.py --variant cf --freeze-epochs 10 --freeze-layers 10 --unfreeze-lr 0.005  # 先冻结后解冻
    python train.py --variant cf --no-normalize   # 关闭 Focal Loss 按 BCE 总量归一化 (默认开启)
    python train.py --variant cf --gamma 2.0     # 调 Focal Loss gamma (文档要求 2.0)
    python train.py --variant cf --imgsz 640 --optimizer SGD --lr0 0.01 --mosaic 0 --device 0
    python train.py --dataset bccd                     # 仅 BCCD 数据集
    python train.py --dataset txl                      # 仅 TXL-PBC 数据集
    python train.py --dataset mixed                    # 混合两个数据集 (默认)

训练完成后 best 权重统一保存到 weights/<variant>.pt。

对应文档技术路线:
1. Backbone 的 ConvModule 中嵌入 CBAM 注意力模块 (通道 + 空间双重注意力)。
2. 替换分类损失函数为 Focal Loss, 配置 alpha=0.25、gamma=2.0、按 BCE 总量归一化
   (消融实验最优: gamma=2.0、归一化开启; p_t 用 exp(-ce_loss) 几何插值)。
3. SGD 优化器 + 余弦退火, 训练 100 轮。
4. 迁移学习流程: 先冻结主干训练检测头, 再解冻全网低学习率精调
   (--freeze-epochs 10; 实测在 BCCD 上单阶段 --freeze-epochs 0 略优, 故默认 0)。

说明: 之前版本的 CBAM 在原训练脚本里对 model.model 的原地修改会被 ultralytics 在
model.train() 内部按 yaml 重建模型时丢弃。本脚本通过自定义 DetectionTrainer.get_model()
在模型构建阶段注入 CBAM / Focal Loss, 保证二者真正参与前向/反向。
"""
import argparse
import os
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.nn.modules import Conv  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from components.cbam import CBAMConv  # noqa: E402
from components.focal_loss import FocalDetectionModel  # noqa: E402
from components.dataset_selection import resolve_yaml, ensure_dataset_ready  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")
PRETRAINED = os.path.join(WEIGHTS_DIR, "yolov8n.pt")


def inject_cbam(model):
    """在 Backbone 的 ConvModule 中嵌入 CBAM, 返回被替换的层索引。"""
    replaced = []
    for index, layer in enumerate(model.model):
        # Detection head also contains Conv layers; only alter the backbone.
        if index >= 10 or not isinstance(layer, Conv):
            continue
        if isinstance(layer, CBAMConv):
            replaced.append(index)
            continue

        conv = CBAMConv(
            layer.conv.in_channels,
            layer.conv.out_channels,
            layer.conv.kernel_size[0],
            layer.conv.stride[0],
            layer.conv.padding[0],
            layer.conv.groups,
            layer.conv.dilation[0],
            layer.act,
        )
        conv.load_state_dict(layer.state_dict(), strict=False)
        conv.f = getattr(layer, "f", -1)
        conv.i = getattr(layer, "i", index)
        model.model[index] = conv
        replaced.append(index)

    if not replaced:
        raise RuntimeError("未找到可嵌入 CBAM 的 Backbone Conv 层")
    return replaced


class CBAMFocalTrainer(DetectionTrainer):
    """自定义 trainer: 在 get_model 阶段注入 CBAM / Focal Loss, 绕过模型重建。"""

    def get_model(self, cfg=None, weights=None, verbose=True):
        variant = getattr(self, "variant", "baseline")
        if variant == "cf":
            gamma = float(getattr(self, "focal_gamma", 2.0))
            alpha = float(getattr(self, "focal_alpha", 0.25))
            normalize = bool(getattr(self, "focal_normalize", False))
            model = self.set_model_names_for_load(
                FocalDetectionModel(cfg, ch=self.data["channels"], nc=self.data["nc"],
                                    verbose=verbose, gamma=gamma, alpha=alpha,
                                    normalize=normalize)
            )
            print(f"  [注入] Focal Loss 已接入 criterion (gamma={gamma}, alpha={alpha}, normalize={normalize})")
        else:
            model = self.set_model_names_for_load(
                DetectionModel(cfg, ch=self.data["channels"], nc=self.data["nc"], verbose=verbose)
            )
        if variant in ("cbam", "cf"):
            indices = inject_cbam(model)
            print(f"  [注入] CBAM 已嵌入 backbone Conv 层 {indices}")
        if weights:
            model.load(weights)
        return model


def verify(path):
    """加载保存的权重, 检查 CBAM 是否真的在模型里。"""
    m = YOLO(path)
    types = {type(x).__name__ for x in m.model.modules()}
    has_cbam = any("CBAM" in t for t in types)
    n_params = sum(p.numel() for p in m.model.parameters())
    print(f"  [校验] {os.path.basename(path)}: 含 CBAM 模块={has_cbam}, 参数量={n_params}")
    return has_cbam


def build_overrides(args, name, model, epochs, freeze, lr0):
    """按文档技术路线构造 ultralytics 训练配置 (超参数均可由命令行覆盖)。"""
    return {
        "model": model,
        "data": args.data,
        "epochs": epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "lr0": lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": not args.no_cos_lr,
        "mosaic": args.mosaic,
        "degrees": args.degrees,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": not args.no_deterministic,
        "amp": not args.no_amp,
        "close_mosaic": args.close_mosaic,
        "device": args.device,
        "project": WEIGHTS_DIR,
        "name": name,
        "exist_ok": True,
        "plots": not args.no_plots,
        "freeze": freeze,
    }


def run_phase(overrides, args, variant):
    """执行一个训练阶段 (冻结 or 解冻)。"""
    trainer = CBAMFocalTrainer(overrides=overrides, _callbacks=None)
    trainer.variant = variant
    trainer.focal_gamma = args.gamma
    trainer.focal_alpha = args.alpha
    trainer.focal_normalize = args.normalize
    trainer.train()


def main():
    ap = argparse.ArgumentParser(description="统一训练脚本 (CBAM/Focal 按文档实现)")
    ap.add_argument("--variant", default="cf", choices=["baseline", "cbam", "cf"])
    ap.add_argument("--device", default="0",
                    help="GPU 编号 (0-7 或 cpu; 默认 0)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=2.0, help="Focal Loss γ (文档要求 2.0, 消融实测最优)")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                    help="Focal Loss 按 BCE 总量归一化, 保持与 BCE 相近的量纲 (默认开启)")
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="冻结主干训练检测头的轮数 (0 表示不冻结; 文档迁移学习: 10)")
    ap.add_argument("--freeze-layers", type=int, default=10,
                    help="阶段1冻结的 backbone 层数 (yolov8n 主干共 10 层, 默认冻结 10)")
    ap.add_argument("--unfreeze-lr", type=float, default=0.005,
                    help="解冻后精调阶段的学习率 lr0 (文档: 低学习率精调)")
    ap.add_argument("--out", default=None, help="输出权重文件名 (默认 weights/<variant>.pt)")
    ap.add_argument("--dataset", default="mixed", choices=["bccd", "txl", "mixed"],
                    help="数据集选择: bccd=仅BCCD, txl=仅TXL-PBC, mixed=混合 (默认 mixed)")
    ap.add_argument("--data", default=None,
                    help="数据集 yaml 路径 (指定后覆盖 --dataset)")
    ap.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    ap.add_argument("--optimizer", default="SGD", help="优化器 (SGD/Adam/AdamW/... )")
    ap.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    ap.add_argument("--lrf", type=float, default=0.01, help="最终学习率 (lr0 的倍数)")
    ap.add_argument("--weight-decay", type=float, default=0.0005, help="权重衰减")
    ap.add_argument("--warmup-epochs", type=float, default=3, help="warmup 轮数")
    ap.add_argument("--no-cos-lr", action="store_true", help="关闭余弦学习率退火")
    ap.add_argument("--mosaic", type=float, default=1.0, help="Mosaic 增强概率 (0 关闭)")
    ap.add_argument("--close-mosaic", type=int, default=10, help="最后 N 轮关闭 Mosaic")
    ap.add_argument("--degrees", type=float, default=180, help="随机旋转角度范围")
    ap.add_argument("--flipud", type=float, default=0.5, help="上下翻转概率")
    ap.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率")
    ap.add_argument("--hsv-h", type=float, default=0.015, help="HSV-H 扰动")
    ap.add_argument("--hsv-s", type=float, default=0.7, help="HSV-S 扰动")
    ap.add_argument("--hsv-v", type=float, default=0.4, help="HSV-V 扰动")
    ap.add_argument("--patience", type=int, default=100, help="早停耐心轮数")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--no-deterministic", action="store_true", help="关闭确定性训练")
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度 (AMP)")
    ap.add_argument("--no-plots", action="store_true", help="不保存训练曲线图")
    args = ap.parse_args()

    # 解析数据集 yaml (--data 优先, 否则按 --dataset 选择, 默认 mixed)
    if args.data:
        data_yaml = args.data
    else:
        data_yaml = ensure_dataset_ready(args.dataset)
    args.data = data_yaml

    out_stem = os.path.splitext(os.path.basename(args.out))[0] if args.out else args.variant
    train_name = f"_{out_stem}_train"
    freeze_name = f"_{out_stem}_freeze"

    print("=" * 70)
    print(f"训练变体: {args.variant}   dataset={args.dataset} ({args.data})  "
          f"batch={args.batch}  epochs={args.epochs}  "
          f"freeze_epochs={args.freeze_epochs}  device={args.device}")
    print("=" * 70)

    # 阶段1: 冻结 backbone, 只训练检测头 (文档: 先冻结主干训练检测头)
    if args.freeze_epochs > 0:
        print(f"\n[阶段1] 冻结 backbone, 训练检测头 {args.freeze_epochs} 轮 ...")
        overrides1 = build_overrides(args, freeze_name, PRETRAINED,
                                     args.freeze_epochs, freeze=args.freeze_layers, lr0=args.lr0)
        run_phase(overrides1, args, args.variant)
        phase2_model = os.path.join(WEIGHTS_DIR, freeze_name, "weights", "best.pt")
        phase2_epochs = args.epochs - args.freeze_epochs
        phase2_lr0 = args.unfreeze_lr  # 文档: 解冻后低学习率精调
    else:
        phase2_model = PRETRAINED
        phase2_epochs = args.epochs
        phase2_lr0 = args.lr0

    # 阶段2: 解冻全网, 低学习率精调 (文档: 再解冻全网低学习率精调)
    print(f"\n[阶段2] 解冻全网, 低学习率精调 {phase2_epochs} 轮 (lr0={phase2_lr0}) ...")
    overrides2 = build_overrides(args, train_name, phase2_model,
                                 phase2_epochs, freeze=None, lr0=phase2_lr0)
    run_phase(overrides2, args, args.variant)

    # 统一保存 best 权重
    out_dir = os.path.join(WEIGHTS_DIR, train_name)
    best = os.path.join(out_dir, "weights", "best.pt")
    if args.out:
        out_name = args.out if args.out.endswith(".pt") else args.out + ".pt"
        dst = os.path.join(WEIGHTS_DIR, out_name)
    else:
        dst = os.path.join(WEIGHTS_DIR, f"{args.variant}.pt")
    shutil.copy2(best, dst)
    shutil.rmtree(out_dir, ignore_errors=True)
    if args.freeze_epochs > 0:
        shutil.rmtree(os.path.join(WEIGHTS_DIR, freeze_name), ignore_errors=True)
    print(f"✅ 训练完成, 权重已保存: {dst}")

    verify(dst)

    # 在测试集上评估
    metrics = YOLO(dst).val(data=args.data, split="test", device=args.device,
                           plots=False, save=False, verbose=False)
    print(f"  mAP@0.5={metrics.box.map50:.4f}  mAP@0.5:0.95={metrics.box.map:.4f}  "
          f"P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}")

    shutil.rmtree(os.path.join(HERE, "runs"), ignore_errors=True)


if __name__ == "__main__":
    main()
