# -*- coding: utf-8 -*-
"""CF-YOLOv8 血细胞检测 SDK (doc.txt: SDK 开发包, 提供 API 接口)。

对外提供简洁的 Python API: 输入血液显微图片, 输出检测框、细胞类别、置信度与
各类细胞统计计数。底层复用 components.detector.CellDetector。

用法 (自动把 cell 目录加入 sys.path, 任意目录均可导入):
    from sdk import CellSDK

    sdk = CellSDK(weights="weights/cf.pt", device="0", conf=0.25)
    res = sdk.detect("blood.jpg")
    print(res.total)          # 总细胞数
    print(res.counts)         # {'RBC': 18, 'WBC': 2}
    for b in res.boxes:
        print(b.name, b.conf, b.xyxy)
"""
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# 保证从任意工作目录都能 import 到 components (cell 目录加入 sys.path)
_CELL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CELL_ROOT not in sys.path:
    sys.path.insert(0, _CELL_ROOT)

from components.detector import CellDetector  # noqa: E402


@dataclass
class DetectionBox:
    """单个检测框: 类别序号、类别名、置信度与 xyxy 像素坐标。"""

    cls: int
    name: str
    conf: float
    xyxy: List[float]


@dataclass
class DetectionResult:
    """单张图片的检测结果。"""

    filename: str
    total: int
    counts: Dict[str, int]
    boxes: List[DetectionBox]
    avg_conf: float
    cost_ms: float


class CellSDK:
    """CF-YOLOv8 血细胞检测 SDK 主入口。

    Args:
        weights: 模型权重路径, 默认 cell/weights/cf.pt。
        device:  推理设备, "0"-"7" 或 "cpu"。
        conf:    默认置信度阈值。
    """

    def __init__(self, weights: Optional[str] = None, device: str = "0", conf: float = 0.25):
        engine_weights = weights or os.path.join(_CELL_ROOT, "weights", "cf.pt")
        self._engine = CellDetector(weights=engine_weights, device=device, conf=conf)
        self._engine.load_model()

    @property
    def model_info(self) -> dict:
        """返回模型信息 (架构/类别/权重/设备/置信度阈值)。"""
        return self._engine.get_model_info()

    def detect(self, image_path: str, conf: Optional[float] = None) -> DetectionResult:
        """检测单张血液显微图片, 返回结构化结果。

        Args:
            image_path: 图片路径。
            conf: 可选置信度阈值, 缺省用构造时的 conf。

        Returns:
            DetectionResult: 含 total / counts / boxes / avg_conf / cost_ms。
        """
        use_conf = self._engine.DEFAULT_CONF if conf is None else float(conf)
        raw = self._engine.detect({"image_path": image_path, "conf": use_conf})
        if raw is None:
            raise ValueError(f"无法读取图片: {image_path}")
        return DetectionResult(
            filename=raw["filename"],
            total=raw["total"],
            counts=raw["counts"],
            boxes=[DetectionBox(**b) for b in raw["boxes"]],
            avg_conf=raw["avg_conf"],
            cost_ms=raw["cost_ms"],
        )

    def detect_many(self, image_paths: List[str], conf: Optional[float] = None) -> List[DetectionResult]:
        """批量检测多张图片, 返回结果列表。"""
        return [self.detect(p, conf=conf) for p in image_paths]
