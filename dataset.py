import cv2, numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        self.files = sorted(
            list(self.img_dir.glob('*.jpg'))  +
            list(self.img_dir.glob('*.jpeg')) +
            list(self.img_dir.glob('*.png'))
        )
        if not self.files:
            raise RuntimeError(f'No images found in {self.img_dir}')

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        name, ext = img_path.stem, img_path.suffix
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        # mask search logic
        for try_ext in (ext, '.png', '.jpg', '.jpeg'):
            mask_path = self.mask_dir / f'{name}{try_ext}'
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                break
        else:
            raise RuntimeError(f'No mask for {img_path}')

        mask = (mask > 200).astype(np.uint8)
        if self.transforms:
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        return img, mask.long()

def get_dataloaders(cfg):
    from torch.utils.data import DataLoader
    mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    train_aug = A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(0.2,0.2,0.2,0.1,p=0.5),
        A.Normalize(mean=mean, std=std), A.pytorch.ToTensorV2()
    ])
    val_aug = A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        A.Normalize(mean=mean, std=std), A.pytorch.ToTensorV2()
    ])

    tr_ds = SegDataset(cfg['root']/"Kvasir-SEG"/'images',
                       cfg['root']/"Kvasir-SEG"/'masks', train_aug)
    vl_ds = SegDataset(cfg['root']/"ETIS-Larib"/'images',
                       cfg['root']/"ETIS-Larib"/'masks', val_aug)

    tr_dl = DataLoader(tr_ds, batch_size=cfg['batch_size'], shuffle=True,
                       num_workers=cfg['num_workers'], pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=cfg['batch_size'], shuffle=False,
                       num_workers=cfg['num_workers'], pin_memory=True)
    return tr_dl, vl_dl