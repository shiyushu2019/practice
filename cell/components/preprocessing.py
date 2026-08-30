# -*- coding: utf-8 -*-
"""图像预处理纯函数 (去噪 + CLAHE 均衡化 + resize)。

从 preprocess.py 下沉而来, 供 preprocess.py 与 components.dataset_selection 复用。
实现文档技术路线 (一)④:
    组合中值与高斯滤波去噪, 通过直方图均衡化改善染色不均,
    统一调整图像尺寸为 640×640, 适配网络输入。

说明:
    1. 去噪: 先 3×3 中值滤波 (去椒盐噪声), 再 3×3 高斯滤波 (平滑)。
    2. 均衡化: 在 LAB 色彩空间的 L 通道上做 CLAHE, 只提对比度、不改色相,
       更适合改善染色不均。
    3. resize: 拉伸到 size×size。由于 YOLO 标签是归一化坐标, 拉伸缩放时
       (x, y, w, h) 的归一化值保持不变, 因此标签文件直接复制即可。
"""
import cv2


def denoise(img):
    """组合中值 + 高斯滤波去噪。"""
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def equalize(img):
    """CLAHE 直方图均衡化 (LAB 的 L 通道), 改善染色不均。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(img, size=640):
    """对单张图像执行: 去噪 -> 直方图均衡化 -> resize 到 size×size。"""
    img = denoise(img)
    img = equalize(img)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
