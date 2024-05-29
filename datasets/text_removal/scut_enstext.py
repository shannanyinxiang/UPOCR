import os

from PIL import Image
from pathlib import Path
from torchvision.transforms import Compose
from torch.utils.data import Dataset, ConcatDataset

from . import transforms as T

class SCUTEnsTextDataset(Dataset):
    def __init__(self, data_root, transform, ext='.jpg'):
        self._get_paths(data_root, ext)
        self.transform = transform
        self.task = 'text removal'
        
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        label = Image.open(self.label_paths[index]).convert('RGB')
        gt_mask = Image.open(self.mask_paths[index]).convert('1')

        data = {
            'image': image, 
            'label': label, 
            'gt_mask': gt_mask, 
            'filepath': self.image_paths[index],
            'task': self.task
        }

        if not self.transform is None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.image_paths)

    def _get_paths(self, data_root, ext):
        image_folder = os.path.join(data_root, 'image')
        label_folder = os.path.join(data_root, 'label')
        mask_folder = os.path.join(data_root, 'mask')

        image_names = os.listdir(image_folder)
        image_names.sort()

        self.image_paths = [os.path.join(image_folder, image_name) 
                             for image_name in image_names]
        self.label_paths = [os.path.join(label_folder, image_name.split('.')[0] + ext) 
                             for image_name in image_names]
        self.mask_paths = [os.path.join(mask_folder, image_name.split('.')[0] + ext)
                            for image_name in image_names]


def make_erase_transform(image_set, args):
    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomCrop(args.crop_min_ratio, args.crop_max_ratio, args.crop_prob))
        transforms.append(T.RandomHorizontalFlip(args.horizontal_flip_prob))
        transforms.append(T.RandomRotate(args.rotate_max_angle, args.rotate_prob))
    transforms.append(T.Resize((args.pix2pix_size, args.pix2pix_size)))
    transforms.append(T.ToTensor())
    return Compose(transforms)

def build(image_set, args):
    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset
    elif image_set == 'val':
        dataset_names = args.val_dataset
    
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'scutsyn_train':
            data_root = root / 'SCUT-Syn' / 'syn_train'; ext = '.png'
        elif dataset_name == 'scutsyn_test':
            data_root = root / 'SCUT-Syn' / 'syn_test'; ext = '.png'
        elif dataset_name == 'scutens_train':
            data_root = root / 'SCUT-EnsText' / 'train'; ext = '.jpg'
        elif dataset_name == 'scutens_test':
            data_root = root / 'SCUT-EnsText' / 'test'; ext = '.jpg'
        else:
            raise NotImplementedError 
        
        transforms = make_erase_transform(image_set, args)
        dataset = TextEraseDataset(data_root, transforms, ext)
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset