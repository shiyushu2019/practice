# -*- coding: utf-8 -*-
"""CF-YOLO26 血细胞检测模型插件。

对应改进模型 CF-YOLO26：
    - 在 Backbone 的 ConvModule 中嵌入 CBAM 注意力模块
    - 分类损失替换为 Focal Loss (α=0.25, γ=2.0)
类别: 0=RBC(红细胞) 1=WBC(白细胞)
"""
import os
import base64
import time
from collections import Counter

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CellDetector:
    CLASS_NAMES = ["RBC", "WBC"]
    CLASS_CN = {"RBC": "红细胞", "WBC": "白细胞"}
    CLASS_COLORS = {"RBC": "#e74c3c", "WBC": "#3498db"}

    def __init__(self, weights=None, device="0", conf=0.25):
        self.WEIGHTS = weights or os.path.join(PROJECT_ROOT, "weights", "cf.pt")
        self.DEFAULT_CONF = conf
        self.device = device
        self.model = None

    def _auto_select_device(self):
        """自动选择剩余显存最多的 GPU，避免 OOM；无 CUDA 则退回 CPU。"""
        if not torch.cuda.is_available():
            return "cpu"
        best, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            try:
                free, _total = torch.cuda.mem_get_info(i)
            except Exception:
                continue
            if free > best_free:
                best_free, best = free, i
        return best

    def load_model(self):
        if self.device is None:
            self.device = self._auto_select_device()
        else:
            self.device = str(self.device)
        print(f"🩸 加载模型: {self.WEIGHTS}  (device={self.device})")
        self.model = YOLO(self.WEIGHTS)
        print("✅ CF-YOLO26 加载成功！")
        return self.model

    def get_model_info(self) -> dict:
        return {
            "arch": "CF-YOLO26 (YOLO26n + CBAM + Focal Loss + DAMS-Net)",
            "classes": self.CLASS_NAMES,
            "classes_cn": [self.CLASS_CN[c] for c in self.CLASS_NAMES],
            "weights": os.path.basename(self.WEIGHTS),
            "device": str(self.device),
            "conf": self.DEFAULT_CONF,
        }

    @staticmethod
    def _img_to_b64(img_bgr) -> str:
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return ""
        return base64.b64encode(buf).decode("utf-8")

    def detect(self, data: dict) -> dict:
        filepath = data["image_path"]
        conf = float(data.get("conf", self.DEFAULT_CONF))

        img = cv2.imread(filepath)
        if img is None:
            return None

        t0 = time.time()
        results = self.model(filepath, conf=conf, device=self.device, verbose=False)
        r = results[0]
        cost_ms = (time.time() - t0) * 1000.0

        counts = Counter(self.CLASS_NAMES[int(b.cls)] for b in r.boxes)
        total = len(r.boxes)
        confs = [float(b.conf) for b in r.boxes]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        # 逐框结果 (SDK 输出: 检测框 xyxy + 类别 + 置信度)
        boxes = [
            {
                "cls": int(b.cls),
                "name": self.CLASS_NAMES[int(b.cls)],
                "conf": round(float(b.conf), 4),
                "xyxy": [round(float(v), 2) for v in b.xyxy[0].tolist()],
            }
            for b in r.boxes
        ]

        annotated = r.plot()  # BGR，已绘制检测框与标签

        return {
            "filename": os.path.basename(filepath),
            "original_b64": self._img_to_b64(img),
            "result_b64": self._img_to_b64(annotated),
            "counts": {c: counts.get(c, 0) for c in self.CLASS_NAMES},
            "total": total,
            "boxes": boxes,
            "avg_conf": round(avg_conf, 4),
            "cost_ms": round(cost_ms, 1),
            "report": self._build_report(counts, total, avg_conf, cost_ms),
        }

    def _build_report(self, counts, total, avg_conf, cost_ms) -> str:
        detail = "、".join(
            f"{self.CLASS_CN[c]}({c}) {counts.get(c, 0)} 个" for c in self.CLASS_NAMES
        )
        return (
            f"检测完成：本图共检出 {total} 个血细胞，其中 {detail}。"
            f"平均置信度 {avg_conf:.3f}，单张推理耗时 {cost_ms:.1f} ms。"
        )
