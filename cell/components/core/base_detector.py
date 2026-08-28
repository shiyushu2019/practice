# core/base_detector.py
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """检测模型插件的抽象基类，与 web_app_example 的 BaseInverter 对齐。"""

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        pass

    @abstractmethod
    def detect(self, input_data: dict) -> dict:
        pass
