import cv2, numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        self.files = sorted(
            list(self.img_dir.glob('*.jpg'))  +
            list(self.img_dir.glob('*.jpeg')) +
            list(self.img_dir.glob('*.png')) + 
            list(self.img_dir.glob('*.tif'))
        )
        if not self.files:
            raise RuntimeError(f'No images found in {self.img_dir}')

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        name, ext = img_path.stem, img_path.suffix
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        # mask search logic
        for try_ext in (ext, '.png', '.jpg', '.jpeg', ".tif"):
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
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # 1.  Affine replaces ShiftScaleRotate
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.8, 1.2),
            rotate=(-45, 45),
            mode=cv2.BORDER_CONSTANT,
            cval=0,
            p=0.7
        ),

        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),

        # 2.  GaussNoise keeps var_limit
        A.GaussNoise(var_limit=(0, 50), p=0.3),

        # 3.  CoarseDropout new signature
        A.CoarseDropout(
            num_holes_range=(1, 5),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill_value=0,
            p=0.3
        ),

        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    val_aug = A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
        ToTensorV2()
    ])

    tr_ds = SegDataset(cfg['root']/"ClinicDB_Kvasir"/'images',
                       cfg['root']/"ClinicDB_Kvasir"/'masks', train_aug)
    vl_ds = SegDataset(cfg['root']/"CVC-ColonDB"/'images',
                       cfg['root']/"CVC-ColonDB"/'masks', val_aug)

    tr_dl = DataLoader(tr_ds, batch_size=cfg['batch_size'], shuffle=True,
                       num_workers=cfg['num_workers'], pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=cfg['batch_size'], shuffle=False,
                       num_workers=cfg['num_workers'], pin_memory=True)
    return tr_dl, vl_dl


if __name__ == '__main__':
    # simple test
    from config import CFG
    train_loader, val_loader = get_dataloaders(CFG)
    for imgs, masks in train_loader:
        print('Train batch - imgs:', imgs.shape, 'masks:', masks.shape)
        break
    for imgs, masks in val_loader:
        print('Val batch   - imgs:', imgs.shape, 'masks:', masks.shape)
        break
    print('Dataset and DataLoader are working fine.')
    # number of samples in train and val sets
    tr_ds = train_loader.dataset
    vl_ds = val_loader.dataset
    print(f'Number of training samples: {len(tr_ds)}')
    print(f'Number of validation samples: {len(vl_ds)}')
