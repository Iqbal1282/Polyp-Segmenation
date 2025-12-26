import math, torch, torch.nn as nn
import timm
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F

class UPerHead(nn.Module):
    """Light 3-stage U-Net decoder built by hand."""
    def __init__(self, in_channels, num_classes, base=256):
        super().__init__()
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base, base//2, 3, padding=1),
            nn.BatchNorm2d(base//2), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(base//2, base//4, 3, padding=1),
            nn.BatchNorm2d(base//4), nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(base//4, num_classes, 1)

    def forward(self, x):
        # x: (B, 1280, 28, 28)
        x = F.interpolate(self.dec1(x), scale_factor=2, mode='bilinear', align_corners=False)  # 56
        x = F.interpolate(self.dec2(x), scale_factor=2, mode='bilinear', align_corners=False)  # 112
        x = F.interpolate(self.dec3(x), scale_factor=2, mode='bilinear', align_corners=False)  # 224
        logits = self.final(x)   # (B, C, 224, 224)
        return logits


class JepaSeg(nn.Module):
    def __init__(self, backbone, decode_head):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x):
        # frozen ViT gives patch tokens
        patches = self.backbone.forward_features(x)        # (B, N, D)
        B, N, D = patches.shape
        h = w = int(math.sqrt(N))                          # 28
        patches = patches.view(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, h, w)
        logits = self.decode_head(patches)                 # (B, num_cls, h, w)
        return logits


def build_model(cfg, device):
    backbone = timm.create_model(
        'vit_huge_patch16_gap_448.in1k_ijepa',
        pretrained=True, num_classes=0
    )
    # keep backbone frozen (optional: unfreeze later)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval().to(device)

    decode_head = UPerHead(
        in_channels=backbone.num_features,   # 1280
        num_classes=cfg['num_classes'],
        base=256
    ).to(device)

    model = JepaSeg(backbone, decode_head)
    return model, backbone


if __name__ == '__main__':
    # simple test
    model, _ = build_model({
        'num_classes': 2
    }, device='cpu')
    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        logits = model(x)
    print(logits.shape)   # (2, 2, 224, 224)
    assert logits.shape == (2, 2, 224, 224)