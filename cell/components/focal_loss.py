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

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', normalize=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.normalize = normalize

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [N, C] 或 [N, C, ...]
            target: 目标值 [N, ...] (允许 soft label, 来自 TAL 任务对齐分配)
        """
        # 计算交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # 计算 p_t: 模型对"真实类别"赋予的概率。
        # 正确做法: p_t = exp(-ce_loss)。对二值标签(t∈{0,1})退化为标准 focal;
        # 对 TAL 的 soft label (IoU 加权, t∈[0,1]) 也能正确泛化, 得
        #   p_t = p^t * (1-p)^(1-t)  (几何插值)
        # 之前的线性插值 p_t = t*p + (1-t)*(1-p) 对 soft label 是错的:
        # 当 t=0.5 时 p_t 恒等于 0.5(与模型预测 p 无关), 使 focal 调制因子退化为常数,
        # 削弱了对易分样本的压制, 是 Focal Loss 精度下降/γ=0.5 崩溃的根因。
        p_t = torch.exp(-ce_loss)

        # 类别平衡因子 alpha_t: 正样本用 alpha, 负样本用 1-alpha
        # (标准 Focal Loss, 论文 Eq.4; 统一 alpha 会整体压低负样本权重导致精度下降)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # 计算 Focal Loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
        modulating = (1 - p_t) ** self.gamma
        loss = alpha_t * modulating * ce_loss

        # 可选: 保持与 BCE 相近的量纲。gamma 较大时 focal loss 的数值会远小于 BCE,
        # 导致 cls loss 相对 box/dfl loss 被削弱、分类头欠拟合。这里把 focal loss
        # 的总量重缩放到与 BCE 相同, 既保留 focal 的相对加权 (压低易分、抬高难分),
        # 又不会让 cls loss 的绝对尺度崩溃。用 BCE/focal 的总量比值做标量缩放,
        # 该比值随训练稳定, 不会像"除以平均调制因子"那样在模型变好后发散。
        if self.normalize:
            total_ce = ce_loss.detach().sum()
            total_focal = loss.detach().sum()
            loss = loss * (total_ce / (total_focal + 1e-8))

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

    def __init__(self, model, gamma: float = 2.0, alpha: float = 0.25, normalize: bool = True):
        super().__init__(model)
        # reduction='none': 与 v8DetectionLoss 中 self.bce 的调用约定一致,
        # 由父类负责最终的 .sum() / target_scores_sum 归一化。
        self.bce = FocalLoss(alpha=alpha, gamma=gamma, reduction='none', normalize=normalize)


class FocalDetectionModel(DetectionModel):
    """使用 Focal Loss 的 DetectionModel 子类。

    init_criterion 是正规方法 (非 lambda), 因此 checkpoint 可以正常 pickle;
    criterion 仍然在首次 loss() 调用时惰性创建, 设备与模型一致。
    """

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True,
                 gamma: float = 2.0, alpha: float = 0.25, normalize: bool = True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.focal_gamma = gamma
        self.focal_alpha = alpha
        self.focal_normalize = normalize

    def init_criterion(self):
        return FocalV8DetectionLoss(self, gamma=self.focal_gamma, alpha=self.focal_alpha,
                                    normalize=self.focal_normalize)
