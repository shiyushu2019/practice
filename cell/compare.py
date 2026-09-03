# -*- coding: utf-8 -*-
"""对比实验脚本 —— 文档 五(2)⑤: YOLOv4/v5/v6/v7/v8 五个基线模型对比。

在测试集/验证集上统一评估各基线权重, 按 RBC / WBC / ALL 三个维度分别输出
Precision / Recall / mAP@0.5, 并与 CF-YOLOv8 (weights/cf.pt) 对比。
混淆矩阵 / PR 曲线留存到 results/。

用法:
    python compare.py                        # 默认 BCCD 测试集
    python compare.py --dataset bccd
    python compare.py --split val --device 1
    python compare.py --weights YOLOv5=weights/yolov5n.pt,YOLOv8=weights/yolov8n.pt

说明:
    - 各基线需先在相同数据集上微调为 2 类 (RBC/WBC) 后再评估, 权重放到 weights/。
      * YOLOv5 / YOLOv6 / YOLOv8 可用 ultralytics 官方训练命令微调;
      * YOLOv4 / YOLOv7 需使用对应官方仓库微调。
    - 本脚本只做统一评估与指标输出, 不修改任何训练 / 数据逻辑。
"""
import argparse
import os
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO  # noqa: E402

from components.dataset_selection import ensure_dataset_ready  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")
RESULTS_DIR = os.path.join(HERE, "results")

# 内置基线表: (显示名, 本地权重路径)。权重需为在相同数据集上微调后的 2 类 checkpoint。
BASELINES = [
    ("YOLOv4", os.path.join(WEIGHTS_DIR, "yolov4.pt")),
    ("YOLOv5", os.path.join(WEIGHTS_DIR, "yolov5n.pt")),
    ("YOLOv6", os.path.join(WEIGHTS_DIR, "yolov6n.pt")),
    ("YOLOv7", os.path.join(WEIGHTS_DIR, "yolov7.pt")),
    ("YOLOv8", os.path.join(WEIGHTS_DIR, "yolov8n.pt")),
    ("CF-YOLOv8", os.path.join(WEIGHTS_DIR, "cf.pt")),
]
NAMES = ["RBC", "WBC"]


def parse_models(spec):
    """解析 --weights 的 名称=路径 列表 (覆盖内置基线表)。"""
    models = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
            name, path = name.strip(), path.strip()
        else:
            path = item
            name = os.path.splitext(os.path.basename(path))[0]
        models.append((name, path))
    return models


def main():
    ap = argparse.ArgumentParser(description="对比实验 (YOLOv4/v5/v6/v7/v8 + CF-YOLOv8)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="0",
                    help="GPU 编号 (0-7 或 cpu; 默认 0)")
    ap.add_argument("--dataset", default="bccd", choices=["bccd", "txl", "mixed"],
                    help="数据集选择 (文档以 BCCD 为评估口径, 默认 bccd)")
    ap.add_argument("--data", default=None,
                    help="数据集 yaml 路径 (指定后覆盖 --dataset)")
    ap.add_argument("--imgsz", type=int, default=640, help="验证输入尺寸")
    ap.add_argument("--iou", type=float, default=0.7, help="验证 NMS 的 IoU 阈值")
    ap.add_argument("--weights", default=None,
                    help="逗号分隔的 名称=路径 列表, 覆盖内置基线表")
    ap.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                    help="保存混淆矩阵/PR 曲线 (默认开启; 用 --no-plots 关闭)")
    ap.add_argument("--project", default=RESULTS_DIR, help="结果输出目录 (混淆矩阵/PR 曲线)")
    args = ap.parse_args()

    # 参数范围保护: IoU 越界时静默夹取到 [0, 1]
    args.iou = max(0.0, min(1.0, args.iou))

    # 解析数据集 yaml (--data 优先, 否则按 --dataset 选择, 默认 bccd)
    if args.data:
        data_yaml = args.data
    else:
        data_yaml = ensure_dataset_ready(args.dataset)
    args.data = data_yaml

    models = parse_models(args.weights) if args.weights else list(BASELINES)

    print(f"对比实验: 数据集={args.dataset} ({args.data}), split={args.split}, 设备={args.device}\n")
    for name, path in models:
        if not os.path.exists(path):
            print(f"⚠️  跳过 {name}: 找不到 {path} (需先在同数据集上微调为 2 类并放入 weights/)")
            continue
        try:
            m = YOLO(path)
            r = m.val(data=args.data, split=args.split, device=args.device,
                      imgsz=args.imgsz, iou=args.iou,
                      project=args.project, name=f"{name}_{args.split}", exist_ok=True,
                      plots=args.plots, save=False, verbose=False)
        except Exception as e:
            print(f"⚠️  评估 {name} 失败: {e}")
            continue

        b = r.box
        names = getattr(r, "names", None) or {i: n for i, n in enumerate(NAMES)}
        print(f"[{name:<10}] mAP@0.5={b.map50:.4f}  mAP@0.5:0.95={b.map:.4f}")
        # 文档要求: 按类别与 ALL 分别输出 Precision / Recall / mAP@0.5
        print(f"           {'类别':<10} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10}")
        for i in range(b.nc):
            cls_name = names.get(i, str(i)) if isinstance(names, dict) else names[i]
            print(f"           {cls_name:<10} {b.p[i]:>10.4f} {b.r[i]:>10.4f} {b.ap50[i]:>10.4f}")
        print(f"           {'ALL':<10} {b.mp:>10.4f} {b.mr:>10.4f} {b.map50:>10.4f}")
        if args.plots:
            print(f"           [图] 混淆矩阵/PR 曲线 -> {os.path.join(args.project, f'{name}_{args.split}')}")
        print()

    # 清理 ultralytics 校验时自动创建的空 runs 目录, 保持工作区干净
    shutil.rmtree(os.path.join(HERE, "runs"), ignore_errors=True)


if __name__ == "__main__":
    main()
