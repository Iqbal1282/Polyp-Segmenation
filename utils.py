import random, numpy as np, torch
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def iou_pytorch(pred, target, num_classes=2, eps=1e-6):
    ious = []
    for cls in range(num_classes):
        intersect = ((pred == cls) & (target == cls)).sum((1,2)).float()
        union     = ((pred == cls) | (target == cls)).sum((1,2)).float()
        iou = (intersect + eps) / (union + eps)
        ious.append(iou.mean().item())
    return np.mean(ious)

class PolyLoss(nn.Module):
    """
    PolyLoss: Polynomial loss expansion of Cross-Entropy
    γ = 2  is the focusing factor (same as Focal Loss).
    For binary / multi-class segmentation (logits input).
    """
    def __init__(self, num_classes=2, gamma=2.0, epsilon=1e-7, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, logits, target):
        # logits: (B, C, H, W)  – raw logits
        # target: (B, H, W)     – class indices (0..C-1)
        probs = F.softmax(logits, dim=1)          # (B, C, H, W)
        tgt_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        # polynomial term (1 - p_t)^γ
        p_t = (probs * tgt_onehot).sum(dim=1)     # (B, H, W)
        poly_term = (1 - p_t + self.epsilon).pow(self.gamma)

        # cross-entropy term  -log(p_t)
        ce_term = F.cross_entropy(logits, target, reduction='none')

        # PolyLoss = CE * (1 - p_t)^γ
        loss = ce_term * poly_term
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss