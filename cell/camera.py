# -*- coding: utf-8 -*-
"""血细胞检测 —— 摄像头实时检测 UI。

用法:
    python camera.py
    按 q 退出
"""
import os
from collections import Counter

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

    # 检测 (verbose=False 关闭逐帧日志, 降低计算开销)
    results = model(frame, conf=0.25, verbose=False)
    r = results[0]

    # 输出各类细胞的统计计数结果
    counts = Counter(r.names[int(b.cls)] for b in r.boxes)
    summary = " | ".join(f"{name}: {counts.get(name, 0)}" for name in r.names.values())
    print(f"检出 {len(r.boxes)} 个细胞 -> {summary}", end="\r")

    # 绘制检测框 (检测框 + 类别标签 + 置信度)
    result_frame = r.plot()

    # 显示
    cv2.imshow('Blood Cell Detection', result_frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print()  # 结束计数行
cap.release()
cv2.destroyAllWindows()
print("👋 已退出")
