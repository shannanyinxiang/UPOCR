from pathlib import Path

from . import transforms as T
from .tampered_ic13 import TamperedIC13Dataset

from torch.utils.data import ConcatDataset
from torchvision.transforms import Compose


def build(cfg, args):
    root = Path(cfg['DATA_ROOT'])
    dataset_names = cfg['DATASET_NAMES']
    transforms = make_data_transform(cfg['TRANSFORM'])

    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'tampered-ic13_train':
            data_root = root / 'Tampered-IC13'
            phase = 'train'
        elif dataset_name == 'tampered-ic13_test':
            data_root = root / 'Tampered-IC13'
            phase = 'test'
        else:
            raise ValueError
        
        dataset = TamperedIC13Dataset(
            data_root=data_root,
            phase=phase,
            transform=transforms,
        )
        datasets.append(dataset)
    
    if len(datasets) > 0:
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
            prob=cfg['RANDOM_HORIZONTAL_FLIP']['PROB']
        ))
    if 'RANDOM_VERTICAL_FLIP' in cfg:
        transforms.append(T.RandomVerticalFlip(
            prob=cfg['RANDOM_VERTICAL_FLIP']['PROB']
        ))
    if 'RANDOM_DISTORTION' in cfg:
        transforms.append(T.RandomDistortion(
            brightness=cfg['RANDOM_DISTORTION']['BRIGHTNESS'],
            contrast=cfg['RANDOM_DISTORTION']['CONTRAST'],
            saturation=cfg['RANDOM_DISTORTION']['SATURATION'],
            hue=cfg['RANDOM_DISTORTION']['HUE'],
            prob=cfg['RANDOM_DISTORTION']['PROB'],
        ))
    if 'RANDOM_ROTATE' in cfg:
        transforms.append(T.RandomRotate(
            angle=cfg['RANDOM_ROTATE']['ANGLE'],
            prob=cfg['RANDOM_ROTATE']['PROB']
        ))
    if 'RANDOM_TRANSPOSE' in cfg:
        transforms.append(T.RandomTranspose(
            prob=cfg['RANDOM_TRANSPOSE']['PROB']
        ))
    if 'RESIZE' in cfg:
        transforms.append(T.Resize(
            size=cfg['RESIZE']['SIZE'],
            phase=cfg['RESIZE']['PHASE']
        ))
    transforms.append(T.ToTensor())

    transforms = Compose(transforms)
    return transforms