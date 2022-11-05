import datetime
from  math import cos, radians
import math
import os
import random
import tensorboardX
import ipdb
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
import matplotlib.ticker as ticker
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from subprocess import call
sys.path.insert(0, '../')
from utils import joint_transforms as simul_transforms
from utils import transforms as extended_transforms
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from htmlCreator import logHtml
cudnn.benchmark = True
import argparse
from rvl_cdip import doc
import cv2



def scale_width(img, target_width):
    ow, oh = img.size
    w = target_width
    target_height = int(target_width * oh / ow)
    h = target_height
    return img.resize((w, h), Image.BICUBIC)
    
    
def transform_onlysize():
    transform_list = []
    transform_list.append(transforms.Resize(448))
    transform_list.append(transforms.CenterCrop((448, 448)))
    transform_list.append(transforms.Pad((42, 42)))
    return transforms.Compose(transform_list)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def read_specific_line(line, path):
    target = int(line)
    with open(path, 'r') as f:
        line = f.readline()
        c = []
        while line:
            currentline = line
            c.append(currentline)
            line = f.readline()
        
    reg =  c[target-1].split(',')[-1]     
    return reg

def path_to_contents(path):
    filename = path.split('/')[-1]
    index_gtline = re.split('_|.jpg', filename)[-2]
    index_image = filename.split('_')[1]
    gt_dir = '/data1/data_sdj/ICDAR2015/end2end/train/gt'
    gt_file = os.path.join(gt_dir, 'gt_img_'+str(index_image)+'.txt')
    # I want to read gt_file of specific line index_gtline
    contents = read_specific_line(int(index_gtline), gt_file)
    #print(index_image, index_gtline, contents)
    return contents

def create_font(fontfile, contents):
    # text and font
    unicode_text = contents
    if isinstance(unicode_text,str) and unicode_text.find('###') != -1 or unicode_text == '':
        print('######################')
        return None
    try:
        font = ImageFont.truetype(fontfile, 36, encoding = 'unic')
    
        # get line size
        # text_width, text_font.getsize(unicode_text)
    
        canvas = Image.new('RGB', (128, 48), "white")
    
        draw = ImageDraw.Draw(canvas)
        draw.text((5,5), unicode_text, 'black', font)

    #canvas.save('unicode-text.png','PNG')
    #canvas.show()
        print(canvas.size)
        return canvas
    except:
        return None

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    #imga = Image.fromarray(imga)
    #imgb = Image.fromarray(imgb)
    w1,h1 = imga.size
    w2,h2 = imgb.size
    img = Image.new("RGB",(256, 48))
    img.paste(imga, (0,0))
    img.paste(imgb, (128, 0))
    return img

def get_transform():
    transform_list = []
    
    transform_list.append(transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, 448)))
    
    #transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    
    transform_list.append(transforms.CenterCrop((448, 448)))
    
    transform_list.append(transforms.ToTensor())
    
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    
    return transforms.Compose(transform_list)


def draw_patch(epoch, model, index2classlist, args):
    """Implement: use model to predict images and draw ten boxes by POOL6
    path to images need to predict is in './dataset/bird'
    result : directory to accept images with ten boxes
    subdirectory is epoch, e,g.0,1,2...
    index2classlist : transform predict label to specific classname
    """
    result = os.path.abspath(args.result)
    if not os.path.isdir(result):
        os.mkdir(result)

    path_img = os.path.join(os.path.abspath('./'), 'vis_img')
    num_imgs = len(os.listdir(path_img))

    dirs = os.path.join(result, str(epoch))
    if not os.path.exists(dirs):
        os.mkdir(dirs)
    
    for original in range(num_imgs):
        img_path = os.path.join(path_img, '{}.jpg'.format(original)) 
        
        transform1 = get_transform()       # transform for predict 
        transform2 = transform_onlysize()  # transform for draw

        img = Image.open(img_path)
        img_pad = transform2(img)
        img_tensor = transform1(img)
        img_tensor = img_tensor.unsqueeze(0)
        out1, out2, out3, indices = model(img_tensor)
        out = out1 + out2 + 0.1 *out3
    
        value, index = torch.max(out.cpu(), 1)
        vrange = np.arange(0, 10)  
        # select from index - index+9 in 2000
        # in test I use 1st class, so I choose indices[0, 9] 
        for i in vrange:
            indice = indices[0, i]
            row, col = indice/56, indice%56
            p_tl = (8*col, 8*row)
            p_br = (col*8+92, row*8+92)
            draw = ImageDraw.Draw(img_pad)
            draw.rectangle((p_tl, p_br), outline='red')
    
        # search corresponding classname
        idx = int(index[0])
        dirname = index2classlist[idx]
    
        filename = 'epoch_'+'{:0>3}'.format(epoch)+'_[org]_'+str(original)+'_[predict]_'+str(dirname)+'.jpg'
        filepath = os.path.join(os.path.join(result,str(epoch)),filename)
        img_pad.save(filepath, "JPEG") 


args = {
    'train_batch_size': 8,
    'alpha':1, # encoding
    'beta':1, # discriminative
    'gama':1, # global
    'lr': 0.001,
    'lr_decay': 0.9,
    'max_iter':1e5,
    'input_size': 384,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot':'',  # empty string denotes learning from scratch
    'print_freq': 200,
    'max_epoch':50,
    'res2Net':False,
    'No_train_images':0,
    'Type_of_train_image':'',
    'Auxilary_loss_contribution':0.4
}

def threed2Img(colorMap,img_colr):
    vis = colorMap.squeeze().detach().cpu().numpy()
    vis = vis/vis.max()
    vis = np.array(vis * 255, dtype = np.uint8)
    vis_colr = cv2.resize(vis,(384,384))
    vis_colr = cv2.applyColorMap(vis_colr, cv2.COLORMAP_JET)
    vis_out = cv2.addWeighted(vis_colr, 0.5, img_colr, 0.5, 0)
    return Image.fromarray(cv2.cvtColor(vis_out, cv2.COLOR_BGR2RGB))
def draw_patch_me(train_args):
    rvl_label = ['letter','form','email','handwritten','advertisement', 'scientific_report', 'scientific_publication', 'specification', 'file_folder', 'news_article', 'budget', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo',]
    Dataroot = '/ssd_scratch/cvit/jobinkv' #location of data
    net =  models.resnext101_32x8d()
    num_features = net.fc.in_features
    fc = list(net.fc.children()) # Remove last layer
    fc.extend([nn.Linear(num_features, doc.num_classes)]) # Add our layer with 4 outputs
    net = DFLTEN_1000(net, k = 20, nclass = doc.num_classes)     # with layer changes
    net.load_state_dict(torch.load('/ssd_scratch/cvit/jobinkv/epoch_9_loss_0.00547_testAcc_0.92463_left.pth'))
    net = net.cuda()
    mean_std =([0.9584, 0.9588, 0.9586], [0.1246, 0.1223, 0.1224])
    transform_test = standard_transforms.Compose([
        standard_transforms.Resize((train_args['input_size'], train_args['input_size']), interpolation=Image.ANTIALIAS),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    transform_draw = standard_transforms.Compose([
        standard_transforms.Resize((train_args['input_size'], train_args['input_size']), interpolation=Image.ANTIALIAS),
        standard_transforms.Pad((1, 1))
    ])
    target_transform = extended_transforms.MaskToTensor()
    tes_set = doc.DOCW('train',Dataroot)
    #tes_loader = DataLoader(tes_set, batch_size=train_args['train_batch_size'], num_workers=1, shuffle=False, drop_last = True)
    k = 0
    font = ImageFont.truetype("arial.ttf", 16)
    if not os.path.exists('./dis'):
        os.makedirs('./dis') 
    if not os.path.exists('./glo'):
        os.makedirs('./glo') 
    if not os.path.exists('./rec'):
        os.makedirs('./rec') 
    if not os.path.exists('./org'):
        os.makedirs('./org') 
    for data in tes_set:
        img_path, gts = data
        img = Image.open(img_path).convert('RGB')
        img_colr = cv2.imread(img_path, 1)
        img_colr = cv2.resize(img_colr,(384,384))
        img_pad = transform_draw(img)
        img_tensor = transform_test(img)
        img_tensor = img_tensor.unsqueeze(0)
        out, indices,colorMap_p ,colorMap_g, colorMap_e, out1 = net(img_tensor.cuda()) #inputs.cuda()
        predictions = out.squeeze().data.max(0)[1].detach().cpu().numpy()
        ploteIt(out1,predictions)
        if predictions == gts:
            continue
        out_p = threed2Img(colorMap_p,img_colr)
        out_g = threed2Img(colorMap_g,img_colr)
        out_e = threed2Img(colorMap_e,img_colr)
        fileName = 'temp_'+str(k)+'_'+rvl_label[gts]+rvl_label[predictions]+'.jpg'
        img_pad.save('./org/'+fileName, "JPEG") 
        out_p.save('./dis/'+fileName, "JPEG") 
        out_g.save('./glo/'+fileName, "JPEG") 
        #out_e.save('temp_'+str(k)+'_'+str(gts)+'.jpg', "JPEG") 
        #cv2.imwrite('temp_Jobin'+str(k)+'_'+str(gts)+'.jpg',vis_out)
        vrange = np.arange(0, 320,10)  
        # select from index - index+9 in 2000
        # in test I use 1st class, so I choose indices[0, 9] 
        for i in vrange:
            indice = indices[0, i]
            row, col = indice//24, indice%24
            p_tl = (16*col, 16*row)
            p_br = (col*16+4, row*16+4)
            draw = ImageDraw.Draw(out_p)
            draw.rectangle((p_tl, p_br), outline='red')
            if i//20 == gts:
                draw.rectangle((p_tl, p_br), outline='green')
                #draw.text(p_tl,str(i//20),(0,0,0),font=font)
        out_p.save('./rec/'+fileName, "JPEG") 
        k = k+1
        if (k>500):
            break
        
def ploteIt(out1,predictions):
    one_feat = out1.squeeze().detach().cpu().numpy()
    x = np.linspace(1, one_feat.shape[0], one_feat.shape[0])
    ipdb.set_trace()
    plt.stem(x, one_feat, use_line_collection=True)
    plt.savefig('stemPlot.png')
    ipdb.set_trace()
    print ('yahoo')
    
if __name__ == '__main__':
    draw_patch_me(args)
