import math, torch, torch.nn as nn
import timm

class Decoder(nn.Module):
    def __init__(self, in_ch, num_cls):
        super().__init__()
        self.head = nn.Conv2d(in_ch, num_cls, 3, padding=1)
    def forward(self, x): return self.head(x)

class JepaSeg(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone, self.decoder = backbone, decoder
    def forward(self, x):
        patches = self.backbone.forward_features(x)        # (B, N, D)
        B, N, D = patches.shape
        h = w = int(math.sqrt(N))
        patches = patches.view(B, h, w, D).permute(0,3,1,2)
        return self.decoder(patches)                       # (B, C, h, w)

def build_model(cfg, device):
    backbone = timm.create_model(
        'vit_huge_patch16_gap_448.in1k_ijepa',
        pretrained=True, num_classes=0)
    for p in backbone.parameters(): p.requires_grad = False
    backbone.eval().to(device)
    model = JepaSeg(backbone, Decoder(1280, cfg['num_classes'])).to(device)
    return model, backbone