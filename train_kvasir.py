import os, torch, numpy as np
import datetime # Added for unique timestamps
from torch.utils.tensorboard import SummaryWriter
from config import CFG
from dataset import get_dataloaders
from models import build_model
from utils import set_seed, iou_pytorch
from torch.optim.lr_scheduler import OneCycleLR
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn
from utils import PolyLoss

# --- SETUP EXPERIMENT LOGGING ---
# Create a unique name for this run (e.g., 20231027-143005_kvasir_exp)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
exp_name = f"{current_time}_{CFG.get('exp_name', 'civicdb_kvasir_colondbijepa_seg')}"
log_dir = os.path.join('runs', exp_name)
writer = SummaryWriter(log_dir)

# Save hyperparameters to TensorBoard for reference
# This helps if you forget what LR or Seed you used
# Note: CFG needs to be a simple dict for this to work cleanly
# writer.add_text('config', str(CFG)) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(CFG['seed'])
os.makedirs('weights', exist_ok=True)

train_loader, val_loader = get_dataloaders(CFG)
model, _ = build_model(CFG, device)

backbone = model.backbone
total_blocks = len(backbone.blocks)        # 32
cutoff = total_blocks - 4                  # 28
for i in range(cutoff, total_blocks):
    for p in backbone.blocks[i].parameters():
        p.requires_grad = True


criterion_dice = DiceLoss(mode='multiclass', from_logits=True)
criterion_ce   = nn.CrossEntropyLoss()
#criterion = PolyLoss(num_classes=CFG['num_classes'], gamma=2.0, reduction='mean')

#optimizer = torch.optim.AdamW(model.decode_head.parameters(),
#                              lr=CFG['lr'],
#                              weight_decay=CFG['weight_decay'])

optimizer = torch.optim.AdamW([
    {'params': (p for p in backbone.blocks[cutoff:].parameters()
                if p.requires_grad), 'lr': 1e-5},   # last 4 blocks
    {'params': model.decode_head.parameters(),      'lr': 1e-3}    # decode head
    ], weight_decay=CFG['weight_decay'])

max_lr = 3e-3
pct_start = 0.15
div_factor = 25
#scheduler = OneCycleLR(optimizer, max_lr=max_lr,
#                       epochs=CFG['epochs'], steps_per_epoch=len(train_loader),
#                       pct_start=pct_start, div_factor=div_factor)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 1e-3],          # same order as param groups
    epochs=CFG['epochs'],
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    div_factor=10
)

best_iou = 0.

for epoch in range(1, CFG['epochs']+1):
    # ---------- train ----------
    model.train() #; backbone.eval()
    train_loss, train_iou = [], []
    train_dice = []
    grad_norm = 0.
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(img)
        
        loss = 0.5 * criterion_dice(logits, mask) + 0.5 * criterion_ce(logits, mask)
        #loss = criterion(logits, mask)
        loss.backward()

        total_norm = sum(p.grad.data.norm(2).item()**2
                         for p in model.decode_head.parameters() if p.grad is not None)
        grad_norm = total_norm ** 0.5
        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        train_loss.append(loss.item())
        train_iou.append(iou_pytorch(logits.argmax(1), mask))
        train_dice.append(1 - criterion_dice(logits, mask).item()) 

    # ---------- val ----------
    model.eval()
    val_loss, val_iou = [], []
    val_dice = []
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = 0.5 * criterion_dice(logits, mask) + 0.5 * criterion_ce(logits, mask)
            #loss = criterion(logits, mask)
            val_loss.append(loss.item())
            val_iou.append(iou_pytorch(logits.argmax(1), mask))
            val_dice.append(1 - criterion_dice(logits, mask).item())


    # ---------- logging ----------
    tr_loss, tr_miou = np.mean(train_loss), np.mean(train_iou)
    vl_loss, vl_miou = np.mean(val_loss),   np.mean(val_iou)
    
    # Organize logs into groups (Loss, Metrics, TrainVars)
    writer.add_scalar('Loss/Train', tr_loss, epoch)
    writer.add_scalar('Loss/Val',   vl_loss, epoch)
    writer.add_scalar('mIoU/Train', tr_miou, epoch)
    writer.add_scalar('mIoU/Val',   vl_miou, epoch)
    writer.add_scalar('Hyperparams/LR', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('Hyperparams/GradNorm', grad_norm, epoch)
    writer.add_scalar('Metrics/Train_Dice', np.mean(train_dice), epoch)
    writer.add_scalar('Metrics/Val_Dice',   np.mean(val_dice), epoch)


    print(f'E{epoch:02d} | tr_loss {tr_loss:.4f} tr_iou {tr_miou:.4f} | '
          f'vl_loss {vl_loss:.4f} vl_iou {vl_miou:.4f} val_dice {np.mean(val_dice):.4f}')
    
    
    if vl_miou > best_iou:
        best_iou = vl_miou
        # Use the unique experiment name for the weight file as well
        save_path = f'weights/best_{exp_name}.pth'
        torch.save(model.state_dict(), save_path)
        print(f'  ↑ New best mIoU! Saved to {save_path}')

writer.close()