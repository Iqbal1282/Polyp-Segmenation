import os, torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import CFG
from dataset import get_dataloaders
from models import build_model
from utils import set_seed, iou_pytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(CFG['seed'])
os.makedirs('weights', exist_ok=True)

train_loader, val_loader = get_dataloaders(CFG)
model, backbone = build_model(CFG, device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.decode_head.parameters(),
                              lr=CFG['lr'],
                              weight_decay=CFG['weight_decay'])

writer = SummaryWriter('runs/kvasir_ijepa_seg')
best_iou = 0.

for epoch in range(1, CFG['epochs']+1):
    # ---------- train ----------
    model.train(); backbone.eval()
    train_loss, train_iou = [], []
    grad_norm = 0.
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(img)
        logits = torch.nn.functional.interpolate(
            logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion(logits, mask)
        loss.backward()

        total_norm = sum(p.grad.data.norm(2).item()**2
                         for p in model.decode_head.parameters() if p.grad is not None)
        grad_norm = total_norm ** 0.5
        optimizer.step()

        train_loss.append(loss.item())
        train_iou.append(iou_pytorch(logits.argmax(1), mask))

    # ---------- val ----------
    model.eval()
    val_loss, val_iou = [], []
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            logits = torch.nn.functional.interpolate(
                logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
            val_loss.append(criterion(logits, mask).item())
            val_iou.append(iou_pytorch(logits.argmax(1), mask))

    # ---------- logging ----------
    tr_loss, tr_miou = np.mean(train_loss), np.mean(train_iou)
    vl_loss, vl_miou = np.mean(val_loss),   np.mean(val_iou)
    writer.add_scalar('Loss/train', tr_loss, epoch)
    writer.add_scalar('mIoU/train', tr_miou, epoch)
    writer.add_scalar('Loss/val',   vl_loss, epoch)
    writer.add_scalar('mIoU/val',   vl_miou, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('GradNorm/decoder', grad_norm, epoch)

    print(f'E{epoch:02d} | train loss {tr_loss:.4f} mIoU {tr_miou:.4f} | '
          f'val loss {vl_loss:.4f} mIoU {vl_miou:.4f}')
    if vl_miou > best_iou:
        best_iou = vl_miou
        torch.save(model.state_dict(), 'weights/best_jepa_seg_kvasir.pth')
        print('  ↑ best model saved.')

writer.close()