# -*- coding: utf-8 -*-
"""数据预处理脚本 —— 实现文档技术路线 (一)④。

组合中值与高斯滤波去噪, 通过直方图均衡化改善染色不均,
统一调整图像尺寸为 640×640, 适配网络输入。

用法:
    python preprocess.py                     # 处理整个 dataset1/BCCD_Dataset
    python preprocess.py --src dataset1/BCCD_Dataset --dst dataset1/BCCD_Dataset_preprocessed
    python preprocess.py --size 640          # 指定输出尺寸 (默认 640)

说明:
    1. 去噪: 先 3×3 中值滤波 (去椒盐噪声), 再 3×3 高斯滤波 (平滑)。
    2. 均衡化: 在 LAB 色彩空间的 L 通道上做 CLAHE, 只提对比度、不改色相,
       更适合改善染色不均。
    3. resize: 拉伸到 size×size。由于 YOLO 标签是归一化坐标, 拉伸缩放时
       (x, y, w, h) 的归一化值保持不变, 因此标签文件直接复制即可。
"""
import argparse
import os
import shutil

import cv2

from components.preprocessing import preprocess_image  # 实现已下沉至 components/preprocessing.py

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "dataset1", "BCCD_Dataset")
DEFAULT_DST = os.path.join(HERE, "dataset1", "BCCD_Dataset_preprocessed")
SPLITS = ("train", "val", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser(description="BCCD 数据预处理 (去噪 + 均衡化 + 640×640)")
    ap.add_argument("--src", default=DEFAULT_SRC, help="原始 BCCD 数据集目录")
    ap.add_argument("--dst", default=DEFAULT_DST, help="预处理后输出目录")
    ap.add_argument("--size", type=int, default=640, help="输出图像尺寸 (默认 640)")
    args = ap.parse_args()

    for split in SPLITS:
        src_img_dir = os.path.join(args.src, split, "images")
        src_lbl_dir = os.path.join(args.src, split, "labels")
        if not os.path.isdir(src_img_dir):
            print(f"⚠️  跳过 {split}: 找不到 {src_img_dir}")
            continue

        dst_img_dir = os.path.join(args.dst, split, "images")
        dst_lbl_dir = os.path.join(args.dst, split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        images = sorted(f for f in os.listdir(src_img_dir) if f.lower().endswith(IMG_EXTS))
        for name in images:
            img = cv2.imread(os.path.join(src_img_dir, name))
            if img is None:
                print(f"⚠️  无法读取 {name}, 跳过")
                continue

            out = preprocess_image(img, args.size)
            cv2.imwrite(os.path.join(dst_img_dir, name), out)

            label_name = os.path.splitext(name)[0] + ".txt"
            src_label = os.path.join(src_lbl_dir, label_name)
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(dst_lbl_dir, label_name))

        print(f"✅ {split}: {len(images)} 张图片处理完成 -> {dst_img_dir}")

    print("✅ 预处理完成。训练时使用: --data dataset1/bccd_preprocessed.yaml")


if __name__ == "__main__":
    main()
