"""统一推理脚本 —— 对单张图片或整个目录做血细胞检测。

用法:
    python inf.py --source path/to/image.jpg
    python inf.py --source dataset/BCCD_Dataset/test/images
    python inf.py --source image.jpg --weights weights/baseline.pt --conf 0.3 --save
    python inf.py --source dataset/BCCD_Dataset/test/images --imgsz 640 --iou 0.7 --save-txt --save-conf
"""
import argparse
import os
import sys
from collections import Counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "cf.pt")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser(description="统一推理脚本")
    ap.add_argument("--source", required=True, help="图片路径或目录")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    ap.add_argument("--max-det", type=int, default=300, help="每张图最大检测框数")
    ap.add_argument("--device", default="7")
    ap.add_argument("--save", action="store_true", help="保存绘制了检测框的结果图")
    ap.add_argument("--save-txt", action="store_true", help="保存 YOLO 标签 txt")
    ap.add_argument("--save-conf", action="store_true", help="标签 txt 附带置信度")
    args = ap.parse_args()

    model = YOLO(args.weights)

    if os.path.isdir(args.source):
        images = sorted(
            os.path.join(args.source, f) for f in os.listdir(args.source)
            if f.lower().endswith(IMG_EXTS)
        )
        if not images:
            print(f"❌ 目录中没有图片: {args.source}")
            return
    elif os.path.isfile(args.source):
        images = [args.source]
    else:
        print(f"❌ 找不到文件或目录: {args.source}")
        return

    for img in images:
        results = model(
            img,
            conf=args.conf,
            imgsz=args.imgsz,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
        )
        r = results[0]
        boxes = r.boxes
        counts = dict(Counter(r.names[int(b.cls)] for b in boxes))
        print(f"{os.path.basename(img)}: 共 {len(boxes)} 个细胞 -> {counts}")

        if args.save:
            out_dir = os.path.join(HERE, "temp", "results")
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, os.path.basename(img))
            r.save(filename=out)
            print(f"  已保存: {out}")


if __name__ == "__main__":
    main()
