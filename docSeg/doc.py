import os
from PIL import Image
from torch.utils import data
ignore_label = 255
num_classes = 28
palette = [0,0,0, 255,0,0, 0,255,0, 0,0,255, 255,0,255, 0,255,255, 255,255,0, 0,125,0, 125,0,0]
labelNames = ['3D objects',
    'Algorithm',       
    'Area chart',      
    'Bar plots',       
    'Block diagram',   
    'Box plot',        
    'Bubble Chart',    
    'Confusion matrix',
    'Contour plot',    
    'Flow chart',      
    'Geographic map',  
    'Graph plots',     
    'Heat map',        
    'Histogram',       
    'Mask',            
    'Medical images',  
    'Natural images',  
    'Pareto charts',   
    'Pie chart',       
    'Polar plot',      
    'Radar chart',     
    'Scatter plot',    
    'Sketches',        
    'Surface plot',    
    'Tables',          
    'Tree Diagram',    
    'Vector plot',     
    'Venn Diagram']   
labelListTable = labelNames.copy()
labelListTable.insert(0,'Iteration')
labelListTable.append('Mean')
tablehead='<table><tr>\n'
for item in labelListTable:
    tablehead+='<th>'+item+'</th>\n'
tablehead+='</tr>\n'

def make_dataset(mode,root):
    assert mode in ['train', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root,'docfig','images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root,'docfig','annotation','train.txt')).readlines()] 
        for it in data_list:
            itt = it.split(', ')
            item = (os.path.join(img_path, itt[0]), labelNames.index(itt[1]))
            items.append(item)
                
    elif mode == 'test':

        img_path = os.path.join(root,'docfig','images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'docfig','annotation','test.txt')).readlines()] 
        for it in data_list:
            itt = it.split(', ')
            item = (os.path.join(img_path, itt[0]), labelNames.index(itt[1]))
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
