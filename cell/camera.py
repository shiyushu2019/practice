# -*- coding: utf-8 -*-
"""血细胞检测 —— 摄像头实时检测 UI。

用法:
    python camera.py
    按 q 退出
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "cf.pt")

# 加载模型
model = YOLO(WEIGHTS)

# 打开摄像头
cap = cv2.VideoCapture(0)

print("=" * 50)
print("🩸 血细胞检测系统已启动")
print("把血细胞图片放在摄像头前")
print("按 q 退出")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 摄像头打开失败")
        break

    # 检测
    results = model(frame, conf=0.25, verbose=False)

    # 绘制检测框
    result_frame = results[0].plot()

    # 显示
    cv2.imshow('Blood Cell Detection', result_frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 已退出")
