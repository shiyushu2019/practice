# -*- coding: utf-8 -*-
"""血细胞检测 —— 摄像头实时检测 UI。

用法:
    python camera.py --device 0 --weights weights/yolo26n_mixed.pt --conf 0.25 --camera 0
    按 q 退出
"""
import argparse
import os
from collections import Counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 文档补充要求：避免多线程数值库重复加载冲突，同时降低摄像头模式的线程开销
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "yolo26n_mixed.pt")


def parse_args():
    ap = argparse.ArgumentParser(description="血细胞实时摄像头检测")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="模型权重路径")
    ap.add_argument("--device", default="0", help="推理设备（0-7 或 cpu）")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号")
    return ap.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.camera)

    try:
        print("=" * 50)
        print("🩸 血细胞检测系统已启动")
        print(f"权重: {args.weights} | device={args.device} | conf={args.conf}")
        print("把血细胞图片放在摄像头前")
        print("按 q 退出")
        print("=" * 50)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 摄像头打开失败")
                break

            # 检测 (verbose=False 关闭逐帧日志, 降低计算开销)
            results = model(frame, conf=args.conf, device=args.device, verbose=False)
            r = results[0]

            counts = Counter(r.names[int(b.cls)] for b in r.boxes)
            summary = " | ".join(f"{name}: {counts.get(name, 0)}" for name in r.names.values())
            print(f"检出 {len(r.boxes)} 个细胞 -> {summary}", end="\r")

            result_frame = r.plot()
            cv2.imshow("Blood Cell Detection", result_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()
        print("👋 已退出")


if __name__ == "__main__":
    main()
