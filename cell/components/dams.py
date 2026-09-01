# -*- coding: utf-8 -*-
"""DAMS-Net —— 动态自适应多尺度特征融合模块 (doc.txt 补充③+)。

对应文档要求:
    - 引入 DAMS-Net 动态自适应多尺度特征融合网络, 替换 Neck 传统特征金字塔的
      固定权重融合, 根据输入图像内容自适应调整多尺度特征融合权重, 增强浅层细节
      特征与深层语义信息。
    - 需单独编码该模块, 并在消融实验中验证其贡献。

实现方式:
    将 Neck 中固定权重的 Concat 融合层替换为 DAMSFusion:
    对每个 Concat 的输入特征, 用各输入 GAP 拼接得到的全局内容描述子, 经共享
    MLP (线性 -> ReLU -> 线性 -> sigmoid) 预测每个输入每个通道的门控权重,
    门控加权后再按通道拼接。输出通道数与原 Concat 完全一致, 不改变后续层输入
    维度, 因此可直接与 CBAM / Focal Loss 组合, 也便于消融与 checkpoint 迁移。
"""
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Concat


class DAMSFusion(nn.Module):
    """内容自适应的加权融合, 替换 Neck 中固定权重的 Concat。

    等价于把 Concat 的"等权拼接"替换为"内容自适应加权拼接":
    对输入 [f1, f2, ...] 先 GAP 拼接成内容描述子, 预测各输入各通道的 sigmoid
    门控, 门控加权后再 torch.cat。
    """

    def __init__(self, channels, d=1):
        super().__init__()
        self.channels = list(channels)
        self.d = d
        total = sum(self.channels)
        hidden = max(total // 4, 1)
        self.gate = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total),
            nn.Sigmoid(),
        )
        # 恒等初始化: 门控初始接近 1 (sigmoid 饱和区), 使 DAMS 起始等价于原 Concat,
        # 从无扰动状态开始学习, 减少对训练早期的干扰。
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, 3.0)

    def forward(self, xs):
        desc = torch.cat([x.mean(dim=(2, 3)) for x in xs], dim=1)  # B, total
        g = self.gate(desc)  # B, total
        outs = []
        start = 0
        for x in xs:
            c = x.size(1)
            w = g[:, start:start + c].view(-1, c, 1, 1)
            outs.append(x * w)
            start += c
        return torch.cat(outs, dim=self.d)


def _out_channels(model, idx):
    """推断 model.model[idx] 的输出通道数。"""
    m = model.model[idx]
    if hasattr(m, "conv") and hasattr(m.conv, "out_channels"):
        return m.conv.out_channels
    if hasattr(m, "cv2") and hasattr(m.cv2, "conv") and hasattr(m.cv2.conv, "out_channels"):
        return m.cv2.conv.out_channels
    if isinstance(m, nn.Upsample):
        return _out_channels(model, idx - 1)
    raise RuntimeError(f"无法推断第 {idx} 层 ({type(m).__name__}) 的输出通道数")


def _concat_in_channels(model, index, f):
    """计算 Concat 各输入分支的通道数 (-1 表示前一层)。"""
    chs = []
    for j in f:
        src = index - 1 if j == -1 else j
        chs.append(_out_channels(model, src))
    return chs


def inject_dams(model):
    """把 Neck 中的 Concat 融合层替换为 DAMSFusion, 返回被替换的层索引列表。

    需在 model.load(weights) 之前调用: 新增门控参数不在 COCO 预训练权重中,
    保持随机初始化; 阶段2 加载阶段1 best.pt 时同名参数可正常迁移。
    """
    targets = []
    for index, layer in enumerate(model.model):
        if isinstance(layer, Concat):
            targets.append((index, layer, _concat_in_channels(model, index, layer.f)))

    if not targets:
        raise RuntimeError("未找到 Neck 中可替换的 Concat 融合层")

    replaced = []
    for index, layer, chs in targets:
        fusion = DAMSFusion(chs, d=getattr(layer, "d", 1))
        fusion.f = getattr(layer, "f", -1)
        fusion.i = getattr(layer, "i", index)
        fusion.type = getattr(layer, "type", "DAMSFusion")
        model.model[index] = fusion
        replaced.append(index)

    return replaced
