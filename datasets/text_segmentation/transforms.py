import random 
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

class ConvertLabel(object):
    def __init__(self, word_effect_value, ignore_value):
        self.word_effect_value = word_effect_value
        self.ignore_value = ignore_value

    def __call__(self, data):
        label = np.asarray(data['label']).copy()
        label[label == 255] = self.ignore_value
        label[label == 200] = self.word_effect_value
        label[label == 100] = 255 
        label = Image.fromarray(label).convert('RGB')
        data['label'] = label 
        data['ignore_value'] = self.ignore_value
        return data 
    

class RandomCrop(object):
    def __init__(self, min_size_ratio, max_size_ratio, prob):
        self.min_size_ratio = min_size_ratio 
        self.max_size_ratio = max_size_ratio 
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            width_crop_ratio = random.uniform(self.min_size_ratio, self.max_size_ratio)
            height_crop_ratio = random.uniform(self.min_size_ratio, self.max_size_ratio)

            W, H = data['image'].size
            crop_W, crop_H = int(W * width_crop_ratio), int(H * height_crop_ratio)
            xmin = random.randint(0, W - crop_W)
            ymin = random.randint(0, H - crop_H)
            xmax = xmin + crop_W 
            ymax = ymin + crop_H

            data = {k: v.crop((xmin, ymin, xmax, ymax)) \
                    if isinstance(v, Image.Image) else v \
                    for k, v in data.items()}
        
        return data 
    

class RandomRotate(object):
    def __init__(self, angle, prob):
        self.angle = angle 
        self.prob = prob 

    def __call__(self, data):
        if random.random() < self.prob:
            angle = random.uniform(-self.angle, self.angle)
            for k in ['label']:
                data[k] = data[k].rotate(
                    angle,
                    resample=Image.Resampling.NEAREST,
                    expand=True,
                    fillcolor=(data['ignore_value'],) * 3
                )
            for k in ['image']:
                data[k] = data[k].rotate(
                    angle,
                    expand=True,
                )
            
        return data 
    

class Resize(object):
    def __init__(self, size, phase):
        self.size = size 
        self.phase = phase

    def __call__(self, data):
        for k in ['label']:
            if self.phase == 'train':
                data[k] = data[k].resize(
                    self.size,
                    resample=Image.Resampling.NEAREST,
                )
            else:
                data[k] = np.asarray(data[k].convert('L'))
        for k in ['image']:
            data[k] = data[k].resize(
                self.size,
                resample=Image.Resampling.BICUBIC,
            )
        return data 
    

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob 

    def __call__(self, data):
        if random.random() < self.prob:
            data = {k: v.transpose(Image.FLIP_LEFT_RIGHT) \
                    if isinstance(v, Image.Image) else v \
                    for k, v in data.items()}
            
        return data 
    

class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, data):
        data = {k: self.to_tensor(v) \
                if isinstance(v, Image.Image) else v \
                for k, v in data.items()}
        
        return data
    

class RandomDistortion(object):
    def __init__(self, brightness, contrast, saturation, hue, prob):
        self.prob = prob 
        self.tfm = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, data):
        if random.random() < self.prob:
            for k in ['image']:
                data[k] = self.tfm(data[k])
        
        return data