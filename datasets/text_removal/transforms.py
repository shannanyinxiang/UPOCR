import PIL
import random 
import torchvision.transforms as transforms

class RandomRotate(object):
    def __init__(self, angle, prob):
        self.angle = angle 
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            angle = random.uniform(-self.angle, self.angle)
            for k in ['image', 'label']:
                data[k] = data[k].rotate(angle, expand=True)
            data['gt_mask'] = data['gt_mask'].rotate(angle, expand=True, fillcolor=1)
        return data 

class Resize(object):
    def __init__(self, size):
        self.size = size 

    def __call__(self, data):
        for k in ['image', 'label', 'gt_mask']:
            data[k] = data[k].resize(self.size)
        return data

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob 
    
    def __call__(self, data):
        if random.random() < self.prob:
            for k in ['image', 'label', 'gt_mask']:
                data[k] = data[k].transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return data

class RandomCrop(object):
    def __init__(self, min_size_ratio=0.7, max_size_ratio=1.0, prob=1.0):
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

            for k in ['image', 'label', 'gt_mask']:
                data[k] = data[k].crop((xmin, ymin, xmax, ymax))
        
        return data

class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, data):
        for k in ['image', 'label', 'gt_mask']:
            data[k] = self.to_tensor(data[k])
        return data 