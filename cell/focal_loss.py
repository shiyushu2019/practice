import torch
import torch.nn as nn
import torch.nn.functional as F

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