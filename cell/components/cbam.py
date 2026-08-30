import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import autopad


class ChannelAttention(nn.Module):
    """通道注意力模块: 最大池化 + 平均池化共享 MLP。

    文档要求: 共享 MLP 为 1 输入层 + 3 隐藏层 + 1 输出层 (共 5 层),
    最大池化与平均池化的结果经该 MLP 后元素求和, 再经 sigmoid。
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_channels // reduction, 1)

        # 1 输入层 + 3 隐藏层 + 1 输出层
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),  # 输入层
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),       # 隐藏层 1
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),       # 隐藏层 2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),       # 隐藏层 3
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),  # 输出层
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """CBAM 注意力模块 - 通道注意力 + 空间注意力"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CBAMConv(nn.Module):
    """在 ConvModule (卷积 + BN + SiLU) 中嵌入 CBAM 注意力模块。

    对应文档: "CBAM 模块置于 Conv2d 卷积层之后、BN 归一化层之前",
    即前向顺序为 conv -> cbam -> bn -> act。
    参数命名与 ultralytics Conv 保持一致 (model.N.conv.weight / bn.*),
    因此 checkpoint 加载兼容; 但不继承 Conv, 从而在 model.fuse() 时
    不会被折叠 BN (CBAM 介于 conv 与 BN 之间, 无法折叠)。
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.cbam = CBAM(c2, reduction=reduction)

    def forward(self, x):
        # 文档要求: CBAM 位于 Conv2d 之后、BN 之前
        return self.act(self.bn(self.cbam(self.conv(x))))