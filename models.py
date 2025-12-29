import math, torch, torch.nn as nn
import timm
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

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
    

class SpatialPriorGate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, patches: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        B, D, Hp, Wp = patches.shape          # patch grid
        H, W = image.shape[-2:]

        hsv = K.color.rgb_to_hsv(image)                       # (B, 3, H, W)
        gray = image.mean(dim=1, keepdim=True)                # (B, 1, H, W)
        edge = K.filters.sobel(gray)                          # (B, 1, H, W)

        # 3. concat & down-sample to patch resolution
        prior_in = torch.cat([hsv, edge], dim=1)              # (B, 4, H, W)
        if (H, W) != (Hp * 16, Wp * 16):                      # safety
            prior_in = F.avg_pool2d(prior_in, kernel_size=16, stride=16)
        else:
            prior_in = prior_in.view(B, 4, Hp, 16, Wp, 16).mean(dim=(3, 5))

        # 4. gate
        gate = self.sigmoid(self.cnn(prior_in))               # (B, D, Hp, Wp)
        return patches * gate

class WaveletGate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # The Haar DWT produces 4 sub-bands (LL, LH, HL, HH).
        # For a 3-channel RGB image, this results in 3 * 4 = 12 channels.
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def get_wavelet_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2D Haar Wavelet Transform using fixed-weight slicing.
        Input x: (B, 3, H, W)
        Output: (B, 12, H/2, W/2)
        """
        # Slicing the image into 2x2 blocks for Haar decomposition
        x00 = x[:, :, 0::2, 0::2]
        x10 = x[:, :, 1::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x11 = x[:, :, 1::2, 1::2]

        # Haar Wavelet formulas
        ll = (x00 + x10 + x01 + x11) / 4.0
        lh = (x00 - x10 + x01 - x11) / 4.0
        hl = (x00 + x10 - x01 - x11) / 4.0
        hh = (x00 - x10 - x01 + x11) / 4.0

        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, patches: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        B, D, Hp, Wp = patches.shape
        H, W = image.shape[-2:]

        # 1. Extract Wavelet features (Low/High frequency sub-bands)
        # Returns (B, 12, H/2, W/2)
        wavelet_feats = self.get_wavelet_features(image)

        # 2. Down-sample to patch resolution
        # Note: wavelet_feats is already H/2, W/2. 
        # If patches are 16x16, we need to pool by a factor of 8 more.
        if (H // 2, W // 2) != (Hp, Wp):
            # Calculate required pooling factor
            pool_factor = (H // 2) // Hp
            prior_in = F.avg_pool2d(wavelet_feats, kernel_size=pool_factor, stride=pool_factor)
        else:
            prior_in = wavelet_feats

        # 3. Generate the gate
        gate = self.sigmoid(self.cnn(prior_in))  # (B, D, Hp, Wp)

        # 4. Apply gate to Transformer patches
        return patches * gate
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticUncertaintyGate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        # The CNN now outputs 2 * embed_dim channels 
        # (one for the mean 'mu' and one for the log-variance 'log_var')
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim * 2, 1) 
        )
        self.softplus = nn.Softplus() # Ensures variance is positive

    def forward(self, patches: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        B, D, Hp, Wp = patches.shape
        H, W = image.shape[-2:]

        # 1. Standard Feature Extraction (Using your original HSV + Edge logic)
        gray = image.mean(dim=1, keepdim=True)
        edge = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]) # Simple edge for example
        edge = F.pad(edge, (0, 1)) 
        # ... (Assuming prior_in is processed to size B, 4, Hp, Wp as before)
        prior_in = F.interpolate(torch.cat([image, edge.mean(1, keepdim=True)], dim=1), 
                                 size=(Hp, Wp), mode='bilinear')

        # 2. Predict Distribution Parameters
        stats = self.cnn(prior_in)
        mu, log_var = torch.chunk(stats, 2, dim=1)
        
        # We use Sigmoid on mu to keep it in range [0, 1]
        mu = torch.sigmoid(mu)
        # Variance should be positive; log_var helps with numerical stability
        std = torch.exp(0.5 * log_var)

        # 3. The Reparameterization Trick
        if self.training:
            # Sample noise epsilon ~ N(0, 1)
            epsilon = torch.randn_like(std)
            # gate = mu + epsilon * sigma
            gate = mu + epsilon * std
            # Clamp to keep weights valid
            gate = torch.clamp(gate, 0.0, 1.0)
        else:
            # At inference, use the deterministic mean
            gate = mu

        return patches * gate
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K


# ---------- your existing HSV gate ----------
class SpatialPriorGate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # HSV(3) + edge(1)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, patches, image):
        B, D, Hp, Wp = patches.shape
        # HSV + edge (Sobel)
        hsv = K.color.rgb_to_hsv(image)
        gray = image.mean(1, keepdim=True)
        edge = K.filters.sobel(gray)
        prior_in = torch.cat([hsv, edge], 1)
        # down-sample to patch grid
        if prior_in.shape[-2:] != (Hp, Wp):
            prior_in = F.avg_pool2d(prior_in, kernel_size=16, stride=16)
        gate = self.sigmoid(self.cnn(prior_in))
        return patches * gate


# ---------- stochastic uncertainty gate ----------
class StochasticUncertaintyGate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        # outputs μ and logσ²
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim * 2, 1)  # μ ∈ ℝᴰ, logσ² ∈ ℝᴰ
        )

    def forward(self, patches, image):
        B, D, Hp, Wp = patches.shape
        # same prep as HSV gate
        hsv = K.color.rgb_to_hsv(image)
        gray = image.mean(1, keepdim=True)
        edge = K.filters.sobel(gray)
        prior_in = torch.cat([hsv, edge], 1)
        if prior_in.shape[-2:] != (Hp, Wp):
            prior_in = F.avg_pool2d(prior_in, kernel_size=16, stride=16)

        stats = self.cnn(prior_in)  # (B, 2D, Hp, Wp)
        mu, log_var = torch.chunk(stats, 2, dim=1)
        mu = torch.sigmoid(mu)  # keep μ ∈ (0,1)
        sigma = torch.exp(0.5 * log_var)  # σ > 0

        if self.training:  # stochastic during training
            eps = torch.randn_like(sigma)
            gate = torch.clamp(mu + eps * sigma, 0.0, 1.0)
        else:  # deterministic at test time
            gate = mu

        return patches * gate


# ---------- dual gate with learnable fusion ----------
class DualGate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.spatial_gate = SpatialPriorGate(embed_dim)
        self.bayes_gate = StochasticUncertaintyGate(embed_dim)
        # learnable 1×1 fusion of the two gates
        self.fuse = nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=True)

    def forward(self, patches, image):
        g1 = self.spatial_gate(patches, image)  # deterministic
        g2 = self.bayes_gate(patches, image)    # stochastic (train) / mean (test)
        # fuse softly
        fused = torch.sigmoid(self.fuse(torch.cat([g1, g2], dim=1)))
        return patches * fused
    

class JepaSeg(nn.Module):
    def __init__(self, backbone, decode_head):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        #self.spatial_gate = SpatialPriorGate(backbone.num_features)
        #self.wavelet_gate = WaveletGate(backbone.num_features)
        #self.stochastic_gate = StochasticUncertaintyGate(backbone.num_features)
        self.dual_gate = DualGate(backbone.num_features)

    def forward(self, x):
        # frozen ViT gives patch tokens
        patches = self.backbone.forward_features(x)        # (B, N, D)
        B, N, D = patches.shape
        h = w = int(math.sqrt(N))                          # 28
        patches = patches.view(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, h, w)
        #hsv_patches = self.spatial_gate(patches, x)            # apply spatial prior gating
        #patches = self.wavelet_gate(patches, x)       # apply wavelet gating
        #patches = self.stochastic_gate(patches, x)
        # combine both gated features along with addition
        #patches = hsv_patches + haar_patches
        patches = self.dual_gate(patches, x)

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

    model = JepaSeg(backbone, decode_head).to(device)

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
    print(logits.shape)   # (2, 2, 224, 224)
    #assert logits.shape == (2, 2, 224, 224)