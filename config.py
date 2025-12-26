from pathlib import Path

CFG = {
    'img_size'   : 448,
    'patch_size' : 16,
    'num_classes': 2,                 # background + tumor
    'batch_size' : 16,
    'lr'         : 1e-3,
    'epochs'     : 30,
    'weight_decay': 1e-4,
    'num_workers': 0,
    'seed'       : 42,
    'root'       : Path('./data'),
}