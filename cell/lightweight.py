# -*- coding: utf-8 -*-
"""模型轻量化脚本 (doc.txt: INT8 量化 + 通道剪枝 + 知识蒸馏恢复精度)。

用法:
    # INT8 量化导出 (默认 TensorRT engine; 可用 --format openvino/onnx 等)
    python lightweight.py quantize --weights weights/cf.pt --data dataset1/bccd_preprocessed.yaml
    python lightweight.py quantize --weights weights/cf.pt --format openvino --device 0

    # 结构化 L1 通道剪枝 (按比例置零 conv 通道, 输出 pruned 权重)
    python lightweight.py prune --weights weights/cf.pt --ratio 0.3 --out weights/cf_pruned.pt

    # 知识蒸馏 (教师 CF-YOLOv8 -> 剪枝后学生, 恢复精度)
    python lightweight.py distill --teacher weights/cf.pt --student weights/cf_pruned.pt --out weights/cf_distilled.pt

说明:
    1. 量化: 借助 ultralytics 导出接口完成 INT8 量化
       (TensorRT engine 的 INT8 会用 --data 数据集做校准; OpenVINO 默认即为 INT8)。
    2. 剪枝: 使用 torch 结构化 L1 范数剪枝按比例把 conv 通道权重置零, 属于"软剪枝"演示;
       真正的通道裁剪/加速需进一步重构网络结构。
    3. 知识蒸馏: 复用 ultralytics 内置蒸馏 (distill_model 参数, 特征级 score-weighted L2),
       教师为完整 CF-YOLOv8, 学生为剪枝后的 CF-YOLOv8, 训练结束后剥离包装只保存学生权重。
"""
import argparse
import os
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))


def quantize(args):
    from ultralytics import YOLO  # noqa: E402

    model = YOLO(args.weights)
    kwargs = {"imgsz": args.imgsz, "int8": args.int8, "half": args.half, "device": args.device}
    if args.data:
        kwargs["data"] = args.data  # engine int8 校准 / openvino 量化需要数据
    out = model.export(format=args.format, **kwargs)
    print(f"✅ INT8 量化导出完成: {out}")


def prune(args):
    import torch  # noqa: E402
    import torch.nn.utils.prune as prune  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    model = YOLO(args.weights)
    n = 0
    for m in model.model.modules():
        if isinstance(m, torch.nn.Conv2d):
            prune.ln_structured(m, name="weight", amount=args.ratio, n=1, dim=0)
            n += 1
    for m in model.model.modules():
        if isinstance(m, torch.nn.Conv2d) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")
    model.save(args.out)
    print(f"✅ 结构化 L1 通道剪枝完成 (置零 {n} 个 conv 层, ratio={args.ratio}) -> {args.out}")


def distill(args):
    """知识蒸馏: 完整 CF-YOLOv8 教师指导剪枝后学生微调, 恢复精度。

    复用 train.py 的 CBAMFocalTrainer 以保证学生结构与 CF-YOLOv8 一致
    (CBAM + Focal Loss + DAMS-Net), 通过 ultralytics 内置 distill_model 参数
    注入蒸馏损失 (特征级 score-weighted L2, 权重 dis), 不改动现有训练逻辑。
    """
    import torch

    from components.dataset_selection import ensure_dataset_ready
    from train import CBAMFocalTrainer  # 复用现有训练器的模型注入逻辑

    if not os.path.exists(args.teacher):
        print(f"❌ 教师权重不存在: {args.teacher}")
        return
    if not os.path.exists(args.student):
        print(f"❌ 学生权重不存在: {args.student} (可先运行 python lightweight.py prune 生成)")
        return

    data_yaml = args.data or ensure_dataset_ready("mixed")  # 与训练脚本默认口径一致
    out_stem = os.path.splitext(os.path.basename(args.out))[0]
    run_name = f"_{out_stem}_distill"

    overrides = {
        "model": args.student,
        "data": data_yaml,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "optimizer": "Adam",
        "lr0": args.lr0,
        "lrf": 0.01,
        "patience": args.patience,
        "cos_lr": True,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "amp": True,
        "plots": True,
        "project": WEIGHTS_DIR,
        "name": run_name,
        "exist_ok": True,
        # ultralytics 内置知识蒸馏: 教师权重路径 + 蒸馏损失权重
        "distill_model": args.teacher,
        "dis": args.dis,
    }

    print("=" * 70)
    print(f"知识蒸馏: student={args.student}  teacher={args.teacher}")
    print(f"  data={data_yaml}  epochs={args.epochs}  batch={args.batch}  "
          f"lr0={args.lr0}  dis={args.dis}  device={args.device}")
    print("=" * 70)

    trainer = CBAMFocalTrainer(overrides=overrides, _callbacks=None)
    # 学生保持完整 CF-YOLOv8 结构 (与剪枝前的 cf.pt 一致)
    trainer.variant = "cf"
    trainer.dams = True
    trainer.focal_gamma = 2.0
    trainer.focal_alpha = 0.25
    trainer.cbam_reduction = 8
    trainer.train()

    best = os.path.join(WEIGHTS_DIR, run_name, "weights", "best.pt")
    if not os.path.exists(best):
        print(f"❌ 蒸馏训练未产生 best.pt: {best}")
        return

    # 剥离 DistillationModel 包装, 只保留学生权重 (与普通权重同格式, 可直接推理/导出)
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema") or ckpt.get("model")
    if ema is None:
        print("❌ best.pt 中找不到模型")
        return
    student = getattr(ema, "student_model", ema)
    if hasattr(student, "criterion"):
        student.criterion = None  # 去除训练期 criterion, 减少 pickle 依赖
    ckpt["model"] = None
    ckpt["ema"] = student

    out = args.out if args.out.endswith(".pt") else args.out + ".pt"
    out_path = os.path.join(WEIGHTS_DIR, out)
    torch.save(ckpt, out_path)
    shutil.rmtree(os.path.join(WEIGHTS_DIR, run_name), ignore_errors=True)
    print(f"✅ 知识蒸馏完成, 学生权重已保存: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="模型轻量化 (量化/剪枝/蒸馏)")
    sub = ap.add_subparsers(dest="mode", required=True)

    q = sub.add_parser("quantize", help="INT8 量化导出")
    q.add_argument("--weights", default=os.path.join(HERE, "weights", "cf.pt"))
    q.add_argument("--format", default="engine", choices=["engine", "openvino", "onnx", "tflite"])
    q.add_argument("--data", default=os.path.join(HERE, "dataset1", "bccd_preprocessed.yaml"),
                   help="校准/量化数据集 yaml")
    q.add_argument("--imgsz", type=int, default=640)
    q.add_argument("--int8", action="store_true", default=True, help="INT8 量化 (默认开启)")
    q.add_argument("--no-int8", dest="int8", action="store_false", help="关闭 INT8")
    q.add_argument("--half", action="store_true", help="FP16 导出")
    q.add_argument("--device", default="0")
    q.set_defaults(func=quantize)

    p = sub.add_parser("prune", help="结构化 L1 通道剪枝")
    p.add_argument("--weights", default=os.path.join(HERE, "weights", "cf.pt"))
    p.add_argument("--ratio", type=float, default=0.3, help="剪枝比例 (0-1)")
    p.add_argument("--out", default=os.path.join(HERE, "weights", "cf_pruned.pt"))
    p.set_defaults(func=prune)

    d = sub.add_parser("distill", help="知识蒸馏 (教师 CF-YOLOv8 -> 剪枝学生, 恢复精度)")
    d.add_argument("--teacher", default=os.path.join(HERE, "weights", "cf.pt"),
                   help="教师权重 (完整 CF-YOLOv8)")
    d.add_argument("--student", default=os.path.join(HERE, "weights", "cf_pruned.pt"),
                   help="学生权重 (剪枝后)")
    d.add_argument("--data", default=None, help="数据集 yaml (默认 mixed)")
    d.add_argument("--epochs", type=int, default=100, help="蒸馏训练轮数")
    d.add_argument("--batch", type=int, default=16)
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--device", default="0")
    d.add_argument("--lr0", type=float, default=0.0001, help="精调学习率")
    d.add_argument("--dis", type=float, default=6.0, help="蒸馏损失权重 (ultralytics 默认 6.0)")
    d.add_argument("--patience", type=int, default=10)
    d.add_argument("--out", default=os.path.join(HERE, "weights", "cf_distilled.pt"))
    d.set_defaults(func=distill)

    args = ap.parse_args()
    args.ratio = max(0.0, min(1.0, getattr(args, "ratio", 0.3)))
    args.func(args)


if __name__ == "__main__":
    main()
