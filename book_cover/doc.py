import os
import ipdb
import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageOps
from torch.utils import data

num_classes = 30
ignore_label = 255

labelNames = ['Arts-N-Photography', 'Biographies-N-Memoirs', 'Business-N-Money', 'Calendars', 'Childrens-Books', 'Christian-Books-N-Bibles', 'Comics-N-Graphic-Novels', 'Computers-N-Technology', 'Cookbooks-Food-N-Wine', 'Crafts-Hobbies-N-Home', 'Engineering-N-Transportation', 'Health-Fitness-N-Dieting', 'History', 'Humor-N-Entertainment', 'Law', 'Literature-N-Fiction', 'Medical-Books', 'Mystery-Thriller-N-Suspense', 'Parenting-N-Relationships', 'Politics-N-Social-Sciences', 'Reference', 'Religion-N-Spirituality', 'Romance', 'Science-Fiction-N-Fantasy', 'Science-N-Math', 'Self-Help', 'Sports-N-Outdoors', 'Teen-N-Young-Adult', 'Test-Preparation', 'Travel']

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
        root, 'Task1','train.txt')).readlines()] 
    for it in data_list:
        itt = it.split('/')
        item = itt[0]
        if item not in labelList:
            labelList.append(item)
    labelList.sort()
    items = []
    if mode == 'train':
       
        img_path = os.path.join(root, 'Task1','train')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'Task1','train.txt')).readlines()] 
        for it in data_list:
            itt = it.split('/')
            item = itt[0]
            item = (os.path.join(img_path, it), labelList.index(item))
            items.append(item)
                

    elif mode == 'test':
        img_path = os.path.join(root, 'Task1','test')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'Task1','test.txt')).readlines()] 
        for it in data_list:
            itt = it.split('/')
            item = itt[0]
            item = (os.path.join(img_path, it), labelList.index(item))
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
