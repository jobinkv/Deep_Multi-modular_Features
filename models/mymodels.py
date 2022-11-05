import torch
from torch import nn
import numpy as np
import math
from utils import initialize_weights

def spatial_pyramid_pool(previous_conv, out_pool_size):
    num_sample = previous_conv.size(0)
    previous_conv_size = previous_conv.size()[2:]
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(float(previous_conv_size[0]) // out_pool_size[i]))
        w_wid = int(math.ceil(float(previous_conv_size[1]) // out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)//2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)//2
        #maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        maxpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
        else:
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp



class _Res2Conv(nn.Module):
    def __init__(self, features, stride=1, scale_num=1, cardinality=1):
        super(_Res2Conv, self).__init__()
        assert scale_num>=2 # it will be standard conv when scale_num=1
        assert features%scale_num==0
        self.invi_features = int(features/scale_num)
        self.convs = nn.ModuleList()
        for i in range(scale_num-1):
            self.convs.append(
                nn.Conv2d(self.invi_features, self.invi_features, 3, stride=stride, padding=1, groups=cardinality)
            )

    def forward(self, x):
        feas = x[:,0:self.invi_features,:,:]
        fea = feas
        for i, conv in enumerate(self.convs):
            first = (i+1)*self.invi_features
            invi_x = x[:,first:first+self.invi_features,:,:]
            fea = conv(fea+invi_x)
            feas = torch.cat([feas, fea], dim=1)
            # print('iter {}, invi_X shape {}, fea shape {}, feas shape {}'.format(i, invi_x.shape, fea.shape, feas.shape))
        return feas

class res2netAdd(nn.Module):
    def __init__(self,net_arch,path, num_classes):
        super(res2netAdd, self).__init__()
        net_arch.load_state_dict(torch.load(path))
        # load the resnet 
        self.net = net_arch
        self.net.layer4[2] = nn.Sequential(*(list(net_arch.layer4[2].children())[:-3]))
        self.net = nn.Sequential(*list(self.net.children())[:-2])
        # chope the last few layer
        # initialize the res2net layer
        self.res2conv = _Res2Conv(512, 1, scale_num=4, cardinality=1)
        # get the spatial pyramid pooling scales
        self.sspScale = [4,2,1]
        #define a final layer to fine tune the network
        self.finalLinear = nn.Sequential(
            nn.Linear(512*np.sum(np.square(self.sspScale)),2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,num_classes),
            nn.Sigmoid()
            )
        # initailize the the defined networks
        initialize_weights(self.res2conv,self.finalLinear)
        
    def forward(self, x):
        x = self.net(x)
        x = self.res2conv(x)
        x = spatial_pyramid_pool(x,self.sspScale)
        return self.finalLinear(x)

class resNetFeat(nn.Module):
    def __init__(self,net_arch,path):
        super(resNetFeat, self).__init__()
        net_arch.load_state_dict(torch.load(path))
        self.layer0 = nn.Sequential(net_arch.conv1, net_arch.bn1, net_arch.relu, net_arch.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net_arch.layer1, net_arch.layer2, net_arch.layer3, net_arch.layer4
        self.layer3[35] = nn.Sequential(*(list(self.layer3[35].children())[:-2]))
        '''
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        '''
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        return x

