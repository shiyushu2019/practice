"""Focal Loss 用于 YOLO 检测分类分支。

按文档实现: FL(pt) = -α_t (1 - pt)^γ log(pt), 配置 α=0.25、γ=2.0。
仅替换分类分支 (cls loss), 边界框回归 (box) 与 DFL 损失保持不变。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017)。"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: 预测 logits [N, C]
            target: 目标标签 [N, C] (允许 soft label, 来自 TAL 任务对齐分配)
        """
        # 交叉熵: ce = -log(p_t)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # p_t = exp(-ce_loss); 对二值标签退化为标准 focal, 对 soft label 也正确泛化
        p_t = torch.exp(-ce_loss)

        # 类别平衡因子: 正样本用 alpha, 负样本用 1-alpha (标准 Focal Loss)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
        return alpha_t * (1 - p_t) ** self.gamma * ce_loss


class FocalV8DetectionLoss(v8DetectionLoss):
    """把 v8DetectionLoss 的分类分支从 BCE 换成 Focal Loss。

    其余 (box / dfl / 任务对齐分配) 逻辑完全沿用父类。
    """

    def __init__(self, model, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__(model)
        # 返回逐元素 loss, 与 v8DetectionLoss 中 self.bce 的调用约定一致,
        # 由父类负责最终的 .sum() / target_scores_sum 归一化。
        self.bce = FocalLoss(alpha=alpha, gamma=gamma)


class FocalDetectionModel(DetectionModel):
    """使用 Focal Loss 的 DetectionModel 子类。

    init_criterion 是正规方法 (非 lambda), 因此 checkpoint 可以正常 pickle;
    criterion 仍然在首次 loss() 调用时惰性创建, 设备与模型一致。
    """

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True,
                 gamma: float = 2.0, alpha: float = 0.25):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.focal_gamma = gamma
        self.focal_alpha = alpha

    def init_criterion(self):
        return FocalV8DetectionLoss(self, gamma=self.focal_gamma, alpha=self.focal_alpha)
