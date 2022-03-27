import random
from PIL import Image, ImageFilter
import numpy as np


# TODO Transformの引数用のクラスを作成した方がよさそう。
class Transforms:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, target):
        for transform in self.transforms:
            data, target = transform(data=data, target=target)
        return data, target


class SyncRandomHorizontalFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data, target):
        if random.random() < self.p:
            data = data.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        return data, target


class SyncRandomRotation:

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, data, target):
        angle = np.random.uniform(*self.degrees)
        data = data.rotate(angle)
        target = target.rotate(angle)
        return data, target


class SyncRandomScaledCrop:

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, data, target):
        scale = np.random.uniform(*self.scale)
        size = np.array(data.size)
        scaled_size = (size * scale).astype(int)
        data = data.resize(scaled_size, Image.BICUBIC)
        target = target.resize(scaled_size, Image.NEAREST)
        if scale > 1.0:
            diff = scaled_size - size
            if np.any(diff == 0):
                return data, target
            box1 = np.random.randint(0, diff).astype(int)
            box2 = box1 + size
            data = data.crop((*box1, *box2))
            target = target.crop((*box1, *box2))
        else:
            diff = size - scaled_size
            if np.any(diff == 0):
                return data, target
            box1 = np.random.randint(0, diff).astype(int)
            original_data = data.copy()
            data = Image.new(data.mode, tuple(size), 0)
            data.paste(original_data, tuple(box1))
            original_target = target.copy()
            target = Image.new(target.mode, tuple(size), 0)
            target.paste(original_target, tuple(box1))
        return data, target


class Transform:

    def __init__(self, data=lambda x: x, target=lambda x: x):
        self.data_transform = data
        self.target_transform = target

    def __call__(self, data, target):
        return self.data_transform(data), self.target_transform(target)


class SyncResize:

    def __init__(self, size):
        self.size = size

    def __call__(self, data, target):
        data = data.resize(self.size, Image.BICUBIC)
        target = target.resize(self.size, Image.NEAREST)
        return data, target


class RandomGaussianBlur:

    def __init__(self, radius=5, p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img
