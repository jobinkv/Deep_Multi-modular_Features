import random

import numpy as np
from skimage.filters import gaussian
import torch
from PIL import Image, ImageFilter


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))
class multiscaleImg:
    def __init__(self,rsize,baseHeights):
        assert sum(baseHeights)<=rsize
        self.rsize = rsize
        self.bheights = baseHeights

    def __call__(self,image):
        new_im = Image.new('RGB', (self.rsize,self.rsize))
        i_cords = 0 
        for i in self.bheights :
            scale = (i/float(image.size[1]))
            wsize = int((float(image.size[0])*float(scale)))
            image1 = image.resize((wsize,i), Image.ANTIALIAS)
            for j in range(0, self.rsize, wsize):
                new_im.paste(image1, (j, i_cords))
            i_cords +=i 
        return new_im

class PortionCrop(object):
    def __init__(self, reg_name):
        self.reg_name = reg_name
        if self.reg_name=='top':
            self.x1 = 0
            self.y1 = 0
            self.x2 = 600
            self.y2 = 250  
        elif self.reg_name=='bot':
            self.x1 = 0
            self.y1 = 530
            self.x2 = 600
            self.y2 = 780  
        elif self.reg_name=='left':
            self.x1 = 0
            self.y1 = 190
            self.x2 = 300
            self.y2 = 590  
        elif self.reg_name=='right':
            self.x1 = 300
            self.y1 = 190
            self.x2 = 600
            self.y2 = 590
        else:
            raise (RuntimeError('Found found no image region specified'))
             
    def __call__(self, img):
        return img.crop((self.x1, self.y1, self.x2, self.y2))


class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, channel_axis=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))
