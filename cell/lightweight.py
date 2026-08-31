# -*- coding: utf-8 -*-
"""模型轻量化脚本 (doc.txt: INT8 量化 + 通道剪枝; 知识蒸馏恢复精度)。

用法:
    # INT8 量化导出 (默认 TensorRT engine; 可用 --format openvino/onnx 等)
    python lightweight.py quantize --weights weights/cf.pt --data dataset1/bccd_preprocessed.yaml
    python lightweight.py quantize --weights weights/cf.pt --format openvino --device 0

    # 结构化 L1 通道剪枝 (按比例置零 conv 通道, 输出 pruned 权重)
    python lightweight.py prune --weights weights/cf.pt --ratio 0.3 --out weights/cf_pruned.pt

说明:
    1. 量化: 借助 ultralytics 导出接口完成 INT8 量化
       (TensorRT engine 的 INT8 会用 --data 数据集做校准; OpenVINO 默认即为 INT8)。
    2. 剪枝: 使用 torch 结构化 L1 范数剪枝按比例把 conv 通道权重置零, 属于"软剪枝"演示;
       真正的通道裁剪/加速需进一步重构网络结构。
    3. 知识蒸馏: ultralytics 不原生支持 KD, 需自定义训练循环 (教师软标签 + 蒸馏损失),
       本脚本暂不实现, 作为后续工作。
"""
import argparse
import os

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


def main():
    ap = argparse.ArgumentParser(description="模型轻量化 (量化/剪枝)")
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

    args = ap.parse_args()
    args.ratio = max(0.0, min(1.0, getattr(args, "ratio", 0.3)))
    args.func(args)


if __name__ == "__main__":
    main()
