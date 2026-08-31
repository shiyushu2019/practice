# -*- coding: utf-8 -*-
"""DAMS-Net —— 动态自适应多尺度特征融合模块 (doc.txt 补充③+)。

对应文档要求:
    - 引入 DAMS-Net 动态自适应多尺度特征融合网络, 替换 Neck 传统特征金字塔的固定权重融合,
      根据输入图像内容自适应调整多尺度特征融合权重, 增强浅层细节特征与深层语义信息。
    - 需单独编码该模块, 并在消融实验中验证其贡献。

实现方式:
    DAMS 从全局内容描述子 (各尺度特征图的 GAP 拼接) 经两层 MLP 预测各尺度权重,
    softmax 归一化后逐层加权 [P3, P4, P5], 再送入原 Detect 头。
    不改变各层通道数与分辨率, 因此可直接与 CBAM / Focal Loss 组合, 也便于消融。

    DAMSDetect 继承 ultralytics 的 Detect, 用"包装"的方式替换 model.model[-1]:
    这样 ultralytics 内部所有 isinstance(m, Detect) 的分支
    (stride 计算 / _apply 设备迁移 / 损失读取 nc·reg_max·stride) 均不受影响,
    checkpoint 也能正常加载 (原 Detect 的 cv2/cv3/dfl 权重原样保留)。
"""
import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect


class DAMS(nn.Module):
    """动态自适应多尺度融合: 内容描述子 -> MLP -> softmax 权重 -> 逐层加权。"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.nl = len(channels)
        total = sum(channels)
        hidden = max(total // reduction, self.nl)
        self.fc = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.nl),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats):
        # feats: list [P3, P4, P5], 每项 shape B*C*H*W
        gap = torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1)  # B, total
        w = self.softmax(self.fc(gap))  # B, nl
        return [f * w[:, i].view(-1, 1, 1, 1) for i, f in enumerate(feats)]


class DAMSDetect(Detect):
    """包装原 Detect 头: forward 先做 DAMS 多尺度动态融合, 再走 Detect 原前向。"""

    def __init__(self, detect, channels):
        # 不调用 Detect.__init__ (避免随机重建 cv2/cv3), 直接继承原头的权重与属性
        nn.Module.__init__(self)
        for name in (
            "nc", "nl", "reg_max", "no", "stride",
            "cv2", "cv3", "dfl",
            "dynamic", "export", "format", "max_det", "agnostic_nms",
            "shape", "anchors", "strides", "legacy", "xyxy", "inplace",
            "np", "i", "f", "type",
        ):
            if hasattr(detect, name):
                setattr(self, name, getattr(detect, name))
        if hasattr(detect, "_end2end"):
            self._end2end = detect._end2end
        self.dams = DAMS(channels)

    def forward(self, x):
        return Detect.forward(self, self.dams(x))


def inject_dams(model):
    """把 model.model[-1] (Detect) 包装为 DAMSDetect, 返回被替换的层索引。

    需在 model.load(weights) 之前调用: 这样 checkpoint 中同名权重 (cv2/cv3/dfl
    及上一阶段已训练的 dams 参数) 能正常加载迁移, DAMS 不丢失。
    """
    detect = model.model[-1]
    if isinstance(detect, DAMSDetect):
        return len(model.model) - 1
    if not isinstance(detect, Detect):
        raise RuntimeError("模型最后一层不是 Detect, 无法注入 DAMS-Net")

    channels = tuple(m[0].conv.in_channels for m in detect.cv2)  # P3/P4/P5 通道数
    model.model[-1] = DAMSDetect(detect, channels)
    return len(model.model) - 1
