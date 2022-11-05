import os
import ipdb
import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageOps
from torch.utils import data
import random
num_classes = 10
ignore_label = 255
labelNames = ['ARB','BEN','ENG','GUJ','HIN','KAN','ORI','PUN','TAM','TEL']

tablehead='<table><tr>\n'
tablehead+='<th>Iteration</th>\n'
for item in labelNames:
    tablehead+='<th>'+item+'</th>\n'
tablehead+='<th>Mean</th>\n'
tablehead+='</tr>\n'

def make_dataset(mode,root):
    assert mode in ['train', 'val', 'test_eva', 'test']
    labelList = []
    data_list = [l.strip('\n') for l in open(os.path.join(
        root, 'cvsi2015','train.txt')).readlines()] 
    for it in data_list:
        itt = it.split('/')
        item = itt[-1].split('_')
        if item[0] not in labelList:
            labelList.append(item[0])
    labelList.sort()
    items = []
    if mode == 'train':
       
        img_path = os.path.join(root, 'cvsi2015')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'cvsi2015','train.txt')).readlines()] 
        for it in data_list:
            itt = it.split('/')
            item = itt[-1].split('_')
            item = (os.path.join(img_path, it), labelList.index(item[0]))
            items.append(item)
                
    elif mode == 'val':
        img_path = os.path.join(root, 'cvsi2015')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'cvsi2015','val.txt')).readlines()] 
        for it in data_list:
            itt = it.split('/')
            item = itt[-1].split('_')
            item = (os.path.join(img_path, it), labelList.index(item[0]))
            items.append(item)

    elif mode == 'test':
        img_path = os.path.join(root, 'cvsi2015')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'cvsi2015','task4.txt')).readlines()] 
        for it in data_list:
            itt = it.split('/')
            item = itt[-1].split('_')
            item = (os.path.join(img_path, it), labelList.index(item[0]))
            items.append(item)
                   
                          
    return items   

 

class DOC(data.Dataset):
    def __init__(self, mode,root, joint_transform=None, transform=None, target_transform=None, path=False):
        self.imgs = make_dataset(mode,root)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.mode = mode
        self.path = path
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
       
    def __getitem__(self, index):
                 
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('L').convert('RGB')
        if random.random()>0.5:
            img = ImageOps.invert(img)
        if self.transform is not None:
                img = self.transform(img)
                
        if self.path:
            return img, label, img_path
        else:
            return img, label
 

    def __len__(self):
        return len(self.imgs)


class DOCW(data.Dataset):
    def __init__(self, mode,root, joint_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode,root)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

       
    def __getitem__(self, index):
                 
        #img_path, label = self.imgs[index]
        #img = Image.open(img_path).convert('L').convert('RGB')
        #if random.random()>0.5:
        #    img = ImageOps.invert(img)
        #if self.transform is not None:
        #        img = self.transform(img)
                
                
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)
