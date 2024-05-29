import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torchvision.transforms import Normalize


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, data):
        data = {k: self.to_tensor(v) \
                if isinstance(v, Image.Image) else v \
                for k, v in data.items()}
        
        return data


class NormalizeTensor(object):
    def __init__(self):
        self.normalize = Normalize(
            mean=(0.485, 0.455, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
    
    def __call__(self, data):
        data = {
            k: self.normalize(v) \
            if k == 'image' else v \
            for k, v in data.items()}
        
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
                data[k] = np.asarray(data[k])
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
    

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            data = {k: v.transpose(Image.FLIP_TOP_BOTTOM) \
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
                    fillcolor=(128,) * 3
                )
            for k in ['image']:
                data[k] = data[k].rotate(
                    angle,
                    expand=True,
                )
            
        return data


class RandomTranspose(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            data = {k: v.transpose(Image.TRANSPOSE) \
                    if isinstance(v, Image.Image) else v \
                    for k, v in data.items()}
            
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