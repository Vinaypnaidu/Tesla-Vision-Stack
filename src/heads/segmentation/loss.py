import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossSeg(nn.Module):
    """
    Implements Focal Loss for segmentation, from BloodAxe/pytorch-toolbelt.

    Parameters:
    -----------
    alpha (float [0..1]): Weight factor to balance positive and negative samples.
    gamma (float): Focal loss power factor.
    preds (torch.Tensor): (N, C, H, W).
    targets (torch.Tensor): (N, C, H, W).

    Returns:
    --------
    loss (torch.Tensor): Segmentation loss.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLossSeg, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        target = target.type(output.type())
        logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
        pt = torch.exp(-logpt)

        focal_term = (1.0 - pt).pow(self.gamma)
        loss = focal_term * logpt

        if self.alpha is not None:
            loss *= self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = loss.mean()
        return loss