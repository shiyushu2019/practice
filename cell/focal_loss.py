"""Focal Loss 用于 YOLO 检测分类分支。

真正替换 v8DetectionLoss 中的 BCE 分类损失, 使 Focal Loss 参与训练/验证的
前向与反向计算 (不再只是"声明未替换")。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    论文: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [N, C] 或 [N, C, ...]
            target: 目标值 [N, ...]
        """
        # 计算交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # 计算 p_t
        pt = torch.exp(-ce_loss)

        # 计算 Focal Loss
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalV8DetectionLoss(v8DetectionLoss):
    """把 v8DetectionLoss 的分类分支从 BCE 换成 Focal Loss。

    其余 (box / dfl / 任务对齐分配) 逻辑完全沿用父类。
    """

    def __init__(self, model, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__(model)
        # reduction='none': 与 v8DetectionLoss 中 self.bce 的调用约定一致,
        # 由父类负责最终的 .sum() / target_scores_sum 归一化。
        self.bce = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')


class FocalDetectionModel(DetectionModel):
    """使用 Focal Loss 的 DetectionModel 子类。

    init_criterion 是正规方法 (非 lambda), 因此 checkpoint 可以正常 pickle;
    criterion 仍然在首次 loss() 调用时惰性创建, 设备与模型一致。
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True,
                 gamma: float = 2.0, alpha: float = 0.25):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.focal_gamma = gamma
        self.focal_alpha = alpha

    def init_criterion(self):
        return FocalV8DetectionLoss(self, gamma=self.focal_gamma, alpha=self.focal_alpha)
