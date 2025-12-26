import torch
import numpy as np
from scipy.ndimage import convolve

# ---------- binarise ----------
def _binarize(pred, gt):
    """pred / gt : (H,W)  0-1 or 0-C channel-last"""
    pred = (pred > 0.5).astype(np.uint8)
    gt   = (gt   > 0.5).astype(np.uint8)
    return pred, gt

# ---------- DSC ----------
def dice(pred, gt, eps=1e-6):
    pred, gt = _binarize(pred, gt)
    inter = (pred * gt).sum()
    return (2. * inter + eps) / (pred.sum() + gt.sum() + eps)

# ---------- IoU ----------
def iou(pred, gt, eps=1e-6):
    pred, gt = _binarize(pred, gt)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)

# ---------- Fβ (β=1) ----------
def f1_score(pred, gt, eps=1e-6):
    pred, gt = _binarize(pred, gt)
    tp = (pred * gt).sum()
    fp = pred.sum() - tp
    fn = gt.sum() - tp
    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    return (2 * precision * recall + eps) / (precision + recall + eps)

# ---------- weighted F-measure (Fω) ----------
def fw_measure(pred, gt, beta_sq=1.0, eps=1e-6):
    """
    wF = (1+beta²) * weightedPrecision * weightedRecall
         ----------------------------------------------
            beta² * weightedPrecision + weightedRecall
    weighting via union area of connected components
    """
    from skimage import measure, morphology
    pred, gt = _binarize(pred, gt)

    def get_weight_map(mask):
        # simple connectivity-based weight map
        lbl, n = measure.label(mask, connectivity=2, return_num=True)
        wmap   = np.zeros_like(mask, dtype=np.float32)
        for region in measure.regionprops(lbl):
            coord = region.coords
            wmap[coord[:,0], coord[:,1]] = 1.0 / (1.0 + region.area)
        return wmap / (wmap.sum() + eps)

    wpred = get_weight_map(pred)
    wgt   = get_weight_map(gt)

    tp = (wpred * pred * gt).sum()
    fp = (wpred * pred * (1 - gt)).sum()
    fn = (wgt   * gt   * (1 - pred)).sum()

    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    return (1 + beta_sq) * prec * rec / (beta_sq * prec + rec + eps)

# ---------- S-measure (Sα) ----------
def s_measure(pred, gt, alpha=0.5):
    """
    Sα = α * So + (1-α) * Sr
    So: object-based similarity, Sr: region-based similarity
    """
    pred, gt = _binarize(pred, gt)

    # object (foreground) terms
    inter_o = (pred * gt).sum()
    union_o = pred.sum() + gt.sum() - inter_o
    so = (2 * inter_o + 1e-6) / (union_o + 1e-6)

    # region (area) terms
    sr = 1 - abs(pred.sum() - gt.sum()) / (gt.sum() + pred.sum() + 1e-6)

    return alpha * so + (1 - alpha) * sr

# ---------- E-measure (Eξ) ----------
def e_measure(pred, gt):
    """
    Enhanced alignment measure (Eξ) – simplified version
    ξ = 1 - |pred - gt| / max(pred, gt, 1)
    averaged over foreground pixels
    """
    pred, gt = _binarize(pred, gt)
    diff = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    em = 1 - diff / np.maximum(np.maximum(pred, gt), 1)
    return em.mean()

# ---------- MAE ----------
def mae(pred, gt):
    pred, gt = pred.astype(np.float32), gt.astype(np.float32)
    return np.abs(pred - gt).mean()

# ---------- batch wrapper ----------
def compute_metrics_batch(pred_batch, gt_batch):
    """
    pred_batch / gt_batch : (B,1,H,W)  tensor  0-1 probabilities
    returns dict with scalar mean values
    """
    pred_batch = torch.sigmoid(pred_batch).cpu().numpy().squeeze(1)
    gt_batch   = gt_batch.cpu().numpy().squeeze(1)

    dsc, iouv, f1, fw, sa, ex, maev = [], [], [], [], [], [], []
    for p, g in zip(pred_batch, gt_batch):
        dsc.append(dice(p, g))
        iouv.append(iou(p, g))
        f1.append(f1_score(p, g))
        fw.append(fw_measure(p, g))
        sa.append(s_measure(p, g))
        ex.append(e_measure(p, g))
        maev.append(mae(p, g))

    return {
        'DSC':  np.mean(dsc),
        'IoU':  np.mean(iouv),
        'F1':   np.mean(f1),
        'Fω':   np.mean(fw),
        'Sα':   np.mean(sa),
        'Eξ':   np.mean(ex),
        'MAE':  np.mean(maev),
    }