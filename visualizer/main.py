import datetime
from  math import cos, radians
import math
import os
import random
import tensorboardX
from ipdb import set_trace as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
#cudnn.benchmark = True
import argparse
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

#================================================================================

parser = argparse.ArgumentParser(description='Visualize')
parser.add_argument('-e','--exp', type=str, default='exp1',
                    help='name of output folder')
parser.add_argument('-d','--dataset', type=str, default='rvl_cdip',
                    help='choose the dataset: cvpr(9 labels) or dsse(7 labels)')
parser.add_argument('-n','--net', type=str, default='resnext101',
                    help='choose the base network ')
parser.add_argument('-s','--trainedModel', type=str, default='',
                    help='give the trained model for visualizing')
parser.add_argument('-i','--imgsize', type=int, default=384,
                    help='image size')
parser.add_argument('-k','--discrim', type=int, default=20,
                    help='discriminative filter size')
parser.add_argument('-g','--gmms', type=int, default=64,
                    help='number of gmms')
parser.add_argument('-a','--selectedFile', type=str, default='',
                    help='Selected file list for visualization')
parser.add_argument('-c','--compresDim', type=int, default=512,
                    help='number of gmms')

args = parser.parse_args()
print ('The exp arguments are ',args.exp,args.net,args.dataset)




def threed2Img(colorMap,img_colr):
    vis = colorMap.squeeze().detach().cpu().numpy()
    vis = vis/vis.max()
    vis = np.array(vis * 255, dtype = np.uint8)
    vis_colr = cv2.resize(vis,(384,384))
    vis_colr = cv2.applyColorMap(vis_colr, cv2.COLORMAP_JET)
    vis_out = cv2.addWeighted(vis_colr, 0.5, img_colr, 0.5, 0)
    return Image.fromarray(cv2.cvtColor(vis_out, cv2.COLOR_BGR2RGB))
def img_average(im):
    im_grey = im.convert('LA') # convert to grayscale
    width, height = im.size
    total = 0
    for i in range(0, width):
        for j in range(0, height):
            total += im_grey.getpixel((i,j))[0]
    mean = total / (width * height)
    print(mean)
    return mean




def draw_patch_me(train_args):
    if args.dataset=='script':
        from cvsi import doc
    elif args.dataset=='book_cover':
        from book_cover import doc
    elif args.dataset=='docSeg':
        from docSeg import doc
    if len(args.selectedFile)>0:
        selectedfile = open(args.selectedFile)
        selecteDList = list(selectedfile)
    net =  models.resnext101_32x8d()
    num_features = net.fc.in_features
    fc = list(net.fc.children()) # Remove last layer
    fc.extend([nn.Linear(num_features, doc.num_classes)]) # Add our layer with 4 outputs
    #net = DFLTEN_ResNetConC_all_vis(net, k = args.discrim, nclass = doc.num_classes, gmms= args.gmms, dataC= args.compresDim)
    net = DFLTEN_VGG19ConC_all(net, k = args.discrim, nclass = doc.num_classes, gmms= args.gmms, dataC= args.compresDim)
    net.load_state_dict(torch.load(args.trainedModel))
    net.eval()
    net = net.cuda()
    mean_std =([0.9584, 0.9588, 0.9586], [0.1246, 0.1223, 0.1224])
    scrip_multiscale = extended_transforms.multiscaleImg(args.imgsize,[100,40,80,160])

    if args.dataset=='script':
        transform_test = standard_transforms.Compose([
            extended_transforms.multiscaleImg(args.imgsize,[100,40,80,160]),
            standard_transforms.Resize((args.imgsize, args.imgsize), interpolation=Image.ANTIALIAS),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        print ('I am string')
    else:
        transform_test = standard_transforms.Compose([
        standard_transforms.Resize((args.imgsize, args.imgsize), interpolation=Image.ANTIALIAS),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
        ])
    transform_draw = standard_transforms.Compose([
        standard_transforms.Resize((args.imgsize,args.imgsize), interpolation=Image.ANTIALIAS),
        standard_transforms.Pad((1, 1))
    ])
    target_transform = extended_transforms.MaskToTensor()
    Dataroot = '/ssd_scratch/cvit/jobinkv' #location of data
    tes_set = doc.DOCW('test',Dataroot)
    #tes_loader = DataLoader(tes_set, batch_size=train_args['train_batch_size'], num_workers=1, shuffle=False, drop_last = True)
    k = 0
    #font = ImageFont.truetype("arial.ttf", 16)
    if not os.path.exists(args.dataset+'Out'):
        os.makedirs(args.dataset+'Out') 
    if not os.path.exists(args.dataset+'Out/dis'):
        os.makedirs(args.dataset+'Out/dis') 
    if not os.path.exists(args.dataset+'Out/glo'):
        os.makedirs(args.dataset+'Out/glo') 
    if not os.path.exists(args.dataset+'Out/org'):
        os.makedirs(args.dataset+'Out/org')
    if not os.path.exists(args.dataset+'Out/img'):
        os.makedirs(args.dataset+'Out/img')
    LabelCnts = {key: 0 for key in doc.labelNames}  
    for data in tes_set:
        img_path, gts  = data
        img_name = os.path.basename(img_path)
        if len(args.selectedFile)>0:
            temp_name = doc.labelNames[gts]+'_'+doc.labelNames[gts]+'_'+img_name+'\n'
            if temp_name not in selecteDList :
                continue
        print (img_name)
        if args.dataset=='script':
            img_clr = Image.open(img_path).convert('RGB') 
            img = Image.open(img_path).convert('L').convert('RGB')
            img_avg = img_average(img)
            #if random.random()>0.5:
            imgI = ImageOps.invert(img)
            imgI_avg = img_average(imgI)
            if imgI_avg>=img_avg:
                 img = imgI
            img_pad = scrip_multiscale(img)
            img_colr = np.array(img_pad)
        else:
            img = Image.open(img_path).convert('RGB')
            img_temp = img.convert('L').convert('RGB')
            img_colr = np.array(img_temp)
            img_colr = cv2.resize(img_colr,(384,384))
            img_pad = transform_draw(img)
        img_tensor = transform_test(img)
        img_tensor = img_tensor.unsqueeze(0)
        #out, indices,colorMap_p ,colorMap_g, colorMap_e, out1 = net(img_tensor.cuda()) #inputs.cuda()
        out, indices, weight_p, weight_g = net(img_tensor.cuda()) #inputs.cuda()
        predictions = out.squeeze().data.max(0)[1].detach().cpu().numpy()
        #ploteIt(out1,predictions)
        #if predictions != gts:
        #    print ('gone')
        #    continue

        LabelCnts[doc.labelNames[gts]]+=1
        st()
        if LabelCnts[doc.labelNames[gts]]>6:
            continue
        out_p = threed2Img(weight_p,img_colr)
        out_g = threed2Img(weight_g,img_colr)
        #out_e = threed2Img(colorMap_e,img_colr)
        fileName = doc.labelNames[gts]+'_'+doc.labelNames[predictions]+'_'+img_name
        fileName = fileName.replace(' ', '')
        img_clr.save(args.dataset+'Out/img/'+fileName, "JPEG") 
        img_pad.save(args.dataset+'Out/org/'+fileName, "JPEG") 
        out_p.save(args.dataset+'Out/dis/'+fileName, "JPEG") 
        out_g.save(args.dataset+'Out/glo/'+fileName, "JPEG")
        #ipdb.set_trace()
        print (fileName, ' done!') 
        #k = k+1
        #if (k>100):
        #    break
        #out_e.save('temp_'+str(k)+'_'+str(gts)+'.jpg', "JPEG") 
        #cv2.imwrite('temp_Jobin'+str(k)+'_'+str(gts)+'.jpg',vis_out)
        #vrange = np.arange(0, 320,10)  
        # select from index - index+9 in 2000
        # in test I use 1st class, so I choose indices[0, 9] 
        '''
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
        '''
        
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
