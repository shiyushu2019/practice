import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
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


class CBAMConv(Conv):
    """在 ConvModule (卷积 + BN + SiLU) 中嵌入 CBAM 注意力模块。

    对应文档: "在 Backbone 的 ConvModule 中嵌入 CBAM 注意力模块"。
    继承 ultralytics 的 Conv, 参数命名不变 (model.N.conv.weight),
    因此 checkpoint 加载 / 模型融合 (fuse) 均兼容。
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, reduction=16):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.cbam = CBAM(c2, reduction=reduction)

    def forward(self, x):
        return self.cbam(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.cbam(self.act(self.conv(x)))