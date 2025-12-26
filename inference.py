import cv2, torch, matplotlib.pyplot as plt
from config import CFG
from models import build_model
from dataset import SegDataset   # re-use val_aug
from albumentations.pytorch import ToTensorV2
import albumentations as A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = build_model(CFG, device)
model.load_state_dict(torch.load('weights/best_jepa_seg_kvasir.pth', map_location=device))
model.eval()

val_aug = A.Compose([
    A.Resize(CFG['img_size'], CFG['img_size']),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2()
])

def infer(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    aug = val_aug(image=img)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model(aug).argmax(1).squeeze(0).cpu().numpy()
    return mask

if __name__ == '__main__':
    m = infer(r'kvasir-sessile\sessile-main-Kvasir-SEG\images\cju0qoxqj9q6s0835b43399p4.jpg')
    plt.imshow(m); plt.axis('off'); plt.show()