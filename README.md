"# Polyp-Segmenation" 

# Model Outline 
```
INPUT  (448×448×3)
   │
   ▼
Multi-Grid Patch Embed  (strides 8/16/32)
   │
   ▼
FROZEN I-JEPA ViT-Huge  (28×28×1280) ────► [ last-4-blocks unfrozen ]
   │
   ▼
SPATIAL-PRIOR GATE  (novel)
┌─────────────────────────────┐
│ HSV + Sobel edge │ 3×3 CNN │► sigmoid gate
└─────────────────────────────┘
   │
   ▼
GATED PATCH TOKENS  (28×28×1280)
   │
   ▼
LIGHT UPerNet DECODER
   ├─► up×2  (56×56×256)
   ├─► up×2  (112×112×128)
   ├─► up×2  (224×224×64)
   ├─► up×2  (448×448×32)
   └─► 1×1 conv  (448×448×2)
   │
   ▼
RESIDUAL REFINEMENT MODULE  (RRM)
   │
   ▼
LOGITS  (448×448×2)
   │
   ├─► MC-Dropout  (T=5)  ──► entropy map
   └─► Dense-CRF  (5 it) ──► refined logits
   │
   ▼
FINAL MASK  (448×448)
```