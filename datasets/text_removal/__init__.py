from pathlib import Path
from torch.utils.data import ConcatDataset
from torchvision.transforms import Compose

from . import transforms as T
from .scut_enstext import SCUTEnsTextDataset


def build(cfg, args):
    root = Path(cfg['DATA_ROOT'])
    dataset_names = cfg['DATASET_NAMES']

    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'scutens_train':
            data_root = root / 'SCUT-ENS' / 'train'; ext = '.jpg'
        elif dataset_name == 'scutens_test':
            data_root = root / 'SCUT-ENS' / 'test'; ext = '.jpg'
        else:
            raise NotImplementedError 
        
        transforms = make_data_transform(cfg['TRANSFORM'])
        dataset = SCUTEnsTextDataset(
            data_root=data_root, 
            transform=transforms, 
            ext=ext)
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset


def make_data_transform(cfg):
    transforms = []
    if 'RANDOM_CROP' in cfg:
        transforms.append(T.RandomCrop(
            min_size_ratio=cfg['RANDOM_CROP']['MIN_SIZE_RATIO'],
            max_size_ratio=cfg['RANDOM_CROP']['MAX_SIZE_RATIO'],
            prob=cfg['RANDOM_CROP']['PROB'],
        ))
    if 'RANDOM_HORIZONTAL_FLIP' in cfg:
        transforms.append(T.RandomHorizontalFlip(
            prob=cfg['RANDOM_HORIZONTAL_FLIP']['PROB'],
        ))
    if 'RANDOM_ROTATE' in cfg:
        transforms.append(T.RandomRotate(
            angle=cfg['RANDOM_ROTATE']['ANGLE'],
            prob=cfg['RANDOM_ROTATE']['PROB'],
        ))
    if 'RESIZE' in cfg:
        transforms.append(T.Resize(
            size=cfg['RESIZE']['SIZE']
        ))
    transforms.append(T.ToTensor())
    
    return Compose(transforms)