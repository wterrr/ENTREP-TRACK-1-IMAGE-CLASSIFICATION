import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random


class AddGaussianNoise:
    """Thêm noise"""
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class SimulateEndoscopeLighting:
    def __init__(self, severity=0.3):
        self.severity = severity

    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]

        # Vignetting effect
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        mask = 1 - (dist / max_dist) * self.severity

        for c in range(3):
            img_array[:, :, c] *= mask

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


class ResizeOrPad:
    """Resize ảnh hoặc pad để handle ảnh nhỏ"""
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img):
        w, h = img.size
        if w < self.min_size or h < self.min_size:
            scale = self.min_size / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return T.functional.resize(img, (new_h, new_w))
        return img