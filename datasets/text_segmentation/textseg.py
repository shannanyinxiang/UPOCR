import os 
import json  
from PIL import Image
from torch.utils.data import Dataset 

class TextSegDataset(Dataset):
    def __init__(self, data_root, phase, transform):
        self.data_root = data_root 
        self.phase = phase 
        self.transform = transform 
        self.task = 'text segmentation'

        self._read_sample_names()


    def __getitem__(self, index):
        sample_name = self.sample_names[index]

        image_path = os.path.join(self.data_root, 'image', sample_name + '.jpg')
        image = Image.open(image_path).convert('RGB')

        label_path = os.path.join(self.data_root, 'semantic_label', sample_name + '_maskfg.png')
        label = Image.open(label_path).convert('RGB')

        data = {
            'image': image,
            'label': label,
            'filepath': image_path,
            'task': self.task
        }

        if not self.transform is None:
            data = self.transform(data)

        return data


    def __len__(self):
        return len(self.sample_names)


    def _read_sample_names(self):
        split_json_path = os.path.join(self.data_root, 'split.json')
        split = json.load(open(split_json_path, 'r'))

        sample_names = []
        if 'train' in self.phase:
            sample_names.extend(split['train'])
        elif 'val' in self.phase:
            sample_names.extend(split['val'])
        elif 'test' in self.phase:
            sample_names.extend(split['test'])
        
        assert len(sample_names) > 0, 'Wrong Phase String'

        self.sample_names = sample_names