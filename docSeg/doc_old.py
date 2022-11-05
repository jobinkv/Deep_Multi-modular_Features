import os
import ipdb
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import torchvision.transforms as standard_transforms
from utils import transforms as extended_transforms
num_classes = 16
ignore_label = 255
palette = [0,0,0, 255,0,0, 0,255,0, 0,0,255, 255,0,255, 0,255,255, 255,255,0, 0,125,0, 125,0,0]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
def colorize_mask_combine(mask,img_path):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    org_image = Image.open(os.path.join(img_path)).convert('RGB')
    new_mask.putpalette(palette)
    mask_combine = Image.blend(new_mask.convert("RGB"),org_image,0.5)
    return mask_combine


def make_dataset(mode,root):
    assert mode in ['train', 'val', 'test_eva', 'test']
    items = []
    if mode == 'train':
       
        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'labels','train10.txt')).readlines()] 
        for it in data_list:
	    itt = it.split(' ')
            item = (os.path.join(img_path, itt[0]), int(itt[1]))
            items.append(item)
                
    elif mode == 'val':

        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'labels','val2.txt')).readlines()] 
        for it in data_list:
	    itt = it.split(' ')
            item = (os.path.join(img_path, itt[0]), int(itt[1]))
            items.append(item)

    elif mode == 'test_eva':

        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'labels','test2.txt')).readlines()] 
        for it in data_list:
	    itt = it.split(' ')
            item = (os.path.join(img_path, itt[0]), int(itt[1]))
            items.append(item)
          
    else:

        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'labels','test2.txt')).readlines()] 
        for it in data_list:
	    itt = it.split(' ')
            item = (os.path.join(img_path, itt[0]), int(itt[1]))
            items.append(item)
                   
                          
    return items   

#def resize(im,percent):
#    	""" retaille suivant un pourcentage 'percent' """
#    	return im.resize(((percent*w)/100,(percent*h)/100))
 
#UBIN0DCBEDC
class DOC(data.Dataset):
    def __init__(self, mode,root):
        self.imgs = make_dataset(mode,root)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.mode = mode

       
    def __getitem__(self, index):
                 
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        return img, label

    def __len__(self):
        return len(self.imgs)
        
        
        
       
    
