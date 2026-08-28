"""统一验证脚本 —— 在测试集/验证集上评估所有训练权重。

用法:
    python eval.py                    # 默认测试集, GPU 4
    python eval.py --split val        # 用验证集
    python eval.py --device 6         # 换卡
"""
import argparse
import os
import shutil
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")

MODELS = {
    "baseline": os.path.join(WEIGHTS_DIR, "baseline.pt"),
    "cbam": os.path.join(WEIGHTS_DIR, "cbam.pt"),
    "cf": os.path.join(WEIGHTS_DIR, "cf.pt"),
    "cf_norm": os.path.join(WEIGHTS_DIR, "cf_norm.pt"),
}
NAMES = ["RBC", "WBC", "Platelets"]


def main():
    ap = argparse.ArgumentParser(description="统一验证脚本")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="7")
    ap.add_argument("--data", default=os.path.join(HERE, "dataset", "bccd.yaml"), help="数据集 yaml 路径")
    ap.add_argument("--imgsz", type=int, default=640, help="验证输入尺寸")
    ap.add_argument("--iou", type=float, default=0.7, help="验证 NMS 的 IoU 阈值")
    ap.add_argument("--weights", default=None,
                    help="逗号分隔的权重路径; 不指定则评估内置 MODELS")
    args = ap.parse_args()

    if args.weights:
        models = []
        for p in args.weights.split(","):
            p = p.strip()
            if p:
                models.append((os.path.basename(p), p))
    else:
        models = list(MODELS.items())

    print(f"评估集: {args.split}, 设备: GPU {args.device}\n")
    for name, path in models:
        if not os.path.exists(path):
            print(f"⚠️  跳过 {name}: 找不到 {path}")
            continue
        m = YOLO(path)
        r = m.val(data=args.data, split=args.split, device=args.device,
                  imgsz=args.imgsz, iou=args.iou,
                  plots=False, save=False, verbose=False)
        b = r.box
        print(f"[{name:<9}] mAP@0.5={b.map50:.4f}  mAP@0.5:0.95={b.map:.4f}  "
              f"Precision={b.mp:.4f}  Recall={b.mr:.4f}")
        for i, cls in enumerate(NAMES):
            if i < len(b.ap50):
                print(f"           {cls:<10} AP@0.5={b.ap50[i]:.4f}  AP@0.5:0.95={b.ap[i]:.4f}")
        print()

    # 清理 ultralytics 校验时自动创建的空 runs 目录, 保持工作区干净
    shutil.rmtree(os.path.join(HERE, "runs"), ignore_errors=True)


if __name__ == "__main__":
    main()
