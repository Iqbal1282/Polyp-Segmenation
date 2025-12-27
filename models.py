import math, torch, torch.nn as nn
import timm
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * (scale ** 2), 3, padding=1)
        self.pix  = nn.PixelShuffle(scale)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pix(self.conv(x))))

class UPerHead(nn.Module):
    def __init__(self, in_channels, num_classes, base=256):
        super().__init__()
        self.up1 = UpsampleBlock(in_channels, base)        # 28→56
        self.up2 = UpsampleBlock(base, base//2)            # 56→112
        self.up3 = UpsampleBlock(base//2, base//4)         # 112→224
        self.up4 = UpsampleBlock(base//4, base//8)         # 224→448
        self.final = nn.Conv2d(base//8, num_classes, 1)

    def forward(self, x):
        x = self.up4(self.up3(self.up2(self.up1(x))))     # 448×448
        return self.final(x)


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


    print(model)
    #print(logits.shape)   # (2, 2, 224, 224)
    #assert logits.shape == (2, 2, 224, 224)