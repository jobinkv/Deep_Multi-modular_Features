import os
import ipdb
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

num_classes = 16
ignore_label = 255
labelNames = ['letter','form','email','handwritten','advertisement', 'scientific_report', 'scientific_publication', 'spec    ification', 'file_folder', 'news_article', 'budget', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo',]
tablehead="""
<table>
  <tr>
<th>Iteration:</th>
<th>letter </th>
<th>form</th>
<th>email</th>
<th>handwritten</th>
<th>advertisement</th>
<th>scientific report</th>
<th>scientific publication</th>
<th>specification</th>
<th>file folder</th>
<th>news article</th>
<th>budget</th>
<th>invoice</th>
<th>presentation</th>
<th>questionnaire</th>
<th>resume</th>
<th>memo</th>
<th>mean</th>
  </tr>
"""



def make_dataset(mode,root):
    assert mode in ['train', 'val', 'test_eva', 'test']
    items = []
    if mode == 'train':
       
        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'labels','train.txt')).readlines()] 
        for it in data_list:
            itt = it.split(' ')
            item = (os.path.join(img_path, itt[0]), int(itt[1]))
            items.append(item)
                
    elif mode == 'val':

        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'labels','val.txt')).readlines()] 
        for it in data_list:
            itt = it.split(' ')
            item = (os.path.join(img_path, itt[0]), int(itt[1]))
            items.append(item)

    elif mode == 'test_eva':

        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'labels','test.txt')).readlines()] 
        for it in data_list:
            itt = it.split(' ')
            item = (os.path.join(img_path, itt[0]), int(itt[1]))
            items.append(item)
          
    else:

        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'labels','test.txt')).readlines()] 
        for it in data_list:
            itt = it.split(' ')
            item = (os.path.join(img_path, itt[0]), int(itt[1]))
            items.append(item)
                   
                          
    return items   

 

class DOC(data.Dataset):
    def __init__(self, mode,root, joint_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode,root)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

       
    def __getitem__(self, index):
                 
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        

        if self.transform is not None:
                img = self.transform(img)
                
                
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
                 
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)
