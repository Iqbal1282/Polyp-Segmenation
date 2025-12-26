import random, numpy as np, torch

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