import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision
import sys
sys.path.insert(0, '../')
from utils import initialize_weights
import encoding
import ipdb

class DFLTEN_ResNet_v1(nn.Module):
        def __init__(self,resnet50, k = 20, nclass = 16):
                super(DFLTEN_ResNet_v1, self).__init__()
                self.k = k
                self.nclass = nclass
                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                #resnet50 = torchvision.models.resnext101_32x8d(pretrained=True)
                # conv1_conv4
                layers_conv1_conv3 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        print ('check:', name)
                        layers_conv1_conv3.append(getattr(resnet50, name))
                conv1_conv3 = torch.nn.Sequential(*layers_conv1_conv3)
                # conv4
                layers_conv4 = []
                layers_conv4.append(getattr(resnet50, 'layer3'))
                conv4 = torch.nn.Sequential(*layers_conv4)
                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(resnet50, 'layer4'))
                conv5 = torch.nn.Sequential(*layers_conv5)

                conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

                # Feature extraction root
                self.conv1_conv3 = conv1_conv3
                self.conv4 = conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(nclass),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = conv6
                self.pool6 = pool6
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                # encoding branch
                n_codes = 64
                self.enco = nn.Sequential(
                  nn.BatchNorm2d(k * nclass),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=k * nclass,K=n_codes),
                  encoding.nn.View(-1, k * nclass * n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(k * nclass * n_codes, nclass),
                )   
                self.enco1 = nn.Sequential(
                  nn.Conv2d(1024, 128, 1), 
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=128,K=n_codes),
                  encoding.nn.View(-1, 128*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(128*n_codes, nclass),
                )   
                initialize_weights(self.enco)
                initialize_weights(self.cls5,self.conv6,self.cls6)

        def forward(self, x):
                batchsize = x.size(0)
                # Stem: Feature extraction
                #x = self.conv1_conv3(x)
                inter4 = self.conv4(x)
                #print('inter4',inter4.shape)

                # G-stream
                x_g = self.conv5(inter4)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                # Encodig stream
                out4 = self.enco(x_p)
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                return out1, out2, out3, out4, indices





class DFLTEN_ResNet(nn.Module):
        def __init__(self,resnet50, k = 20, nclass = 16):
                super(DFLTEN_ResNet, self).__init__()
                self.k = k
                self.nclass = nclass
                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                #resnet50 = torchvision.models.resnext101_32x8d(pretrained=True)
                # conv1_conv4
                layers_conv1_conv4 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        print ('check:', name)
                        layers_conv1_conv4.append(getattr(resnet50, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)
                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(resnet50, 'layer4'))
                conv5 = torch.nn.Sequential(*layers_conv5)

                conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

                # Feature extraction root
                self.conv1_conv4 = conv1_conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(nclass),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = conv6
                self.pool6 = pool6
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                # encoding branch
                n_codes = 64
                self.enco = nn.Sequential(
                  nn.BatchNorm2d(k * nclass),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=k * nclass,K=n_codes),
                  encoding.nn.View(-1, k * nclass * n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(k * nclass * n_codes, nclass),
                )   
                self.enco1 = nn.Sequential(
                  nn.Conv2d(1024, 512, 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=512,K=n_codes),
                  encoding.nn.View(-1, 512*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(512*n_codes, nclass),
                )   
                initialize_weights(self.enco1)
                initialize_weights(self.cls5,self.conv6,self.cls6)

        def forward(self, x):
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                #print('inter4',inter4.shape)

                # G-stream
                x_g = self.conv5(inter4)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                # Encodig stream
                out4 = self.enco1(inter4)
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                return out1, out2, out3, out4, indices


class DFLTEN_ResNetConC(nn.Module):
        def __init__(self,resnet50, k = 20, nclass = 16):
                super(DFLTEN_ResNetConC, self).__init__()
                self.k = k
                self.nclass = nclass
                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                #resnet50 = torchvision.models.resnext101_32x8d(pretrained=True)
                # conv1_conv4
                layers_conv1_conv4 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        print ('check:', name)
                        layers_conv1_conv4.append(getattr(resnet50, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)
                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(resnet50, 'layer4'))
                conv5 = torch.nn.Sequential(*layers_conv5)

                conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

                # Feature extraction root
                self.conv1_conv4 = conv1_conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(nclass),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = conv6
                self.pool6 = pool6
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                # encoding branch
                n_codes = 64
                self.enco = nn.Sequential(
                  nn.BatchNorm2d(k * nclass),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=k * nclass,K=n_codes),
                  encoding.nn.View(-1, k * nclass * n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(k * nclass * n_codes, nclass),
                )  
                self.enco1 = nn.Sequential(
                  nn.Conv2d(1024, 512, 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=512,K=n_codes),
                  encoding.nn.View(-1, 512*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(512*n_codes, nclass),
                )  
                self.finalConc = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(4*nclass, nclass))
                  
                initialize_weights(self.enco1,self.finalConc)
                initialize_weights(self.cls5,self.conv6,self.cls6)

        def forward(self, x):
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                #print('inter4',inter4.shape)
                # G-stream
                x_g = self.conv5(inter4)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)
                # Encodig stream
                out4 = self.enco1(inter4)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                return out, indices

class DFLTEN_Lader(nn.Module):
        def __init__(self,resnet50, k = 20, nclass = 16):
                super(DFLTEN_Lader, self).__init__()
                self.k = k
                self.nclass = nclass
                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                # conv1
                layers_conv1 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                layers_conv1.append(getattr(resnet50, 'layer1'))
                self.conv1 = torch.nn.Sequential(*layers_conv1) # 256 filters
                # conv2
                layers_conv2 = []
                layers_conv2.append(getattr(resnet50, 'layer2'))
                self.conv2 = torch.nn.Sequential(*layers_conv2) # 512 filters
                # conv3
                layers_conv3 = []
                layers_conv3.append(getattr(resnet50, 'layer3'))
                self.conv3 = torch.nn.Sequential(*layers_conv3) # 1024 filters
                # conv4
                layers_conv4 = []
                layers_conv4.append(getattr(resnet50, 'layer4'))
                self.conv4 = torch.nn.Sequential(*layers_conv4) # 2048 filters

                #conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                #pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

                # Feature extraction root
                #self.conv1_conv4 = conv1_conv4

                # G-Stream
                #self.conv5 = conv5
                self.conv5 = nn.Sequential(
                        nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(nclass),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                self.pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                # encoding branch
                n_codes = 64
                self.enco = nn.Sequential(
                  nn.BatchNorm2d(k * nclass),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=k * nclass,K=n_codes),
                  encoding.nn.View(-1, k * nclass * n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(k * nclass * n_codes, nclass),
                )  
                self.enco1 = nn.Sequential(
                  nn.Conv2d(256, 512, 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=512,K=n_codes),
                  encoding.nn.View(-1, 512*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(512*n_codes, nclass),
                )
                self.enco2 = nn.Sequential(
                  nn.Conv2d(512, 512, 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=512,K=n_codes),
                  encoding.nn.View(-1, 512*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(512*n_codes, nclass),
                )
                self.enco3 = nn.Sequential(
                  nn.Conv2d(1024, 512, 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=512,K=n_codes),
                  encoding.nn.View(-1, 512*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(512*n_codes, nclass),
                )
                self.enco4 = nn.Sequential(
                  nn.Conv2d(2048, 512, 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=512,K=n_codes),
                  encoding.nn.View(-1, 512*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(512*n_codes, nclass),
                )

  
                self.finalConc = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(7*nclass, nclass))
                  
                initialize_weights(self.enco1,self.enco2,self.enco3,self.enco4,self.finalConc)
                initialize_weights(self.conv5,self.conv6,self.cls6)

        def forward(self, x):
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter1 = self.conv1(x)
                out1 = self.enco1(inter1)
                #-----------------------
                inter2 = self.conv2(inter1)
                out2 = self.enco2(inter2)
                #-----------------------
                inter3 = self.conv3(inter2)
                out3 = self.enco3(inter3)
                #-----------------------
                inter4 = self.conv4(inter3)
                out4 = self.enco4(inter4)
                #print('inter4',inter4.shape)
                # G-stream
                out5 = self.conv5(inter4)
                out5 = out5.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter3)
                
                x_p, indices = self.pool6(x_p)
                inter5 = x_p
                out6 = self.cls6(x_p)
                out6 = out6.view(batchsize, -1)

                # Side-branch
                inter5 = inter5.view(batchsize, -1, self.k * self.nclass)
                out7 = self.cross_channel_pool(inter5)
                out7 = out7.view(batchsize, -1)
                out = self.finalConc(torch.cat((out1, out2, out3, out4, out5, out6, out7),1))
                return out, indices




class DFLTEN_ResNetEonly(nn.Module):
        def __init__(self,resnet50, k = 20, nclass = 16):
                super(DFLTEN_ResNetEonly, self).__init__()
                self.k = k
                self.nclass = nclass
                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                #resnet50 = torchvision.models.resnext101_32x8d(pretrained=True)
                # conv1_conv4
                layers_conv1_conv4 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                for i in range(2):
                        name = 'layer%d' % (i + 1)
                        print ('check:', name)
                        layers_conv1_conv4.append(getattr(resnet50, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)
                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(resnet50, 'layer3'))
                conv5 = torch.nn.Sequential(*layers_conv5)

                conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

                # Feature extraction root
                self.conv1_conv4 = conv1_conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(nclass),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = conv6
                self.pool6 = pool6
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                # encoding branch
                n_codes = 64
                self.enco = nn.Sequential(
                  nn.BatchNorm2d(k * nclass),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=k * nclass,K=n_codes),
                  encoding.nn.View(-1, k * nclass * n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(k * nclass * n_codes, nclass),
                )  
                self.enco1 = nn.Sequential(
                  nn.Conv2d(512, 512, 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=512,K=n_codes),
                  encoding.nn.View(-1, 512*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(512*n_codes, nclass),
                )  
                self.finalConc = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(4*nclass, nclass))
                  
                initialize_weights(self.enco1,self.finalConc)
                initialize_weights(self.cls5,self.conv6,self.cls6)

        def forward(self, x):
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                #print('inter4',inter4.shape)
                out = self.enco1(inter4)
                # Side-branch
                return out, out*1





class DFL_VGG16(nn.Module):
        def __init__(self, k = 10, nclass = 200):
                super(DFL_VGG16, self).__init__()
                self.k = k
                self.nclass = nclass

                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features
                conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11])
                conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:])
                conv6 = torch.nn.Conv2d(512, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((56, 56), stride = (56, 56), return_indices = True)

                # Feature extraction root
                self.conv1_conv4 = conv1_conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(512, 200, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(200),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = conv6
                self.pool6 = pool6
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                initialize_weights(self.cls5,self.conv6,self.cls6)

        def forward(self, x):
                batchsize = x.size(0)

                # Stem: Feature extractionc
                inter4 = self.conv1_conv4(x)
                #print(inter4.shape)

                # G-stream
                x_g = self.conv5(inter4)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)

                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                # encodedFeatures branch
                return out1, out2, out3, indices


class DFL_ResNet_for_sample(nn.Module):
        def __init__(self, k = 10, nclass = 200):
                super(DFL_ResNet_for_sample, self).__init__()
                self.k = k
                self.nclass = nclass

                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                resnet50 = torchvision.models.resnet50(pretrained=True)
                # conv1_conv4
                layers_conv1_conv4 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        layers_conv1_conv4.append(getattr(resnet50, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)

                self.conv1_conv4 = conv1_conv4

        def forward(self, x):
                batchsize = x.size(0)

                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                center = torch.norm(inter4.norm(2,0),2,0).mean()

                return center


class DFL_ResNet(nn.Module):
        def __init__(self,resnet50, k = 10, nclass = 200):
                super(DFL_ResNet, self).__init__()
                self.k = k
                self.nclass = nclass
                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                #resnet50 = torchvision.models.resnext101_32x8d(pretrained=True)
                # conv1_conv4
                layers_conv1_conv4 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        print (name)
                        layers_conv1_conv4.append(getattr(resnet50, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)

                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(resnet50, 'layer4'))
                conv5 = torch.nn.Sequential(*layers_conv5)

                conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

                # Feature extraction root
                self.conv1_conv4 = conv1_conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(nclass),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = conv6
                self.pool6 = pool6
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                # encoding branch
                initialize_weights(self.cls5,self.conv6,self.cls6)

        def forward(self, x):
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                ipdb.set_trace()
                #print('inter4',inter4.shape)

        # G-stream
                #print('inter4',inter4.shape)
                x_g = self.conv5(inter4)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)
                #print('out1',out1.shape)

                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                #print('conv6',x_p.shape)
                x_p, indices = self.pool6(x_p)
                #print(x_p.shape)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)
                #print('out2',out2.shape)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                #print('out3',out3.shape)


                return out1, out2, out3, indices

class DFLTEN_1000(nn.Module):
        def __init__(self,resnet50, k = 20, nclass = 16):
                super(DFLTEN_1000, self).__init__()
                self.k = k
                self.nclass = nclass
                # k channels for one class, nclass is total classes, therefore k * nclass for conv6
                #resnet50 = torchvision.models.resnext101_32x8d(pretrained=True)
                # conv1_conv4
                layers_conv1_conv4 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        print ('check:', name)
                        layers_conv1_conv4.append(getattr(resnet50, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)
                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(resnet50, 'layer4'))
                conv5 = torch.nn.Sequential(*layers_conv5)
                enco5 = torch.nn.Sequential(*layers_conv5)

                conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

                # Feature extraction root
                self.conv1_conv4 = conv1_conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(nclass),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = conv6
                self.pool6 = pool6
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                # encoding branch
                n_codes = 64
                 
                self.enco = nn.Sequential(
                  nn.BatchNorm2d(k * nclass),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=k * nclass,K=n_codes),
                  encoding.nn.View(-1, k * nclass * n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(k * nclass * n_codes, nclass),
                )
                self.enco5 = enco5  
                self.enco1 = nn.Sequential(
                  nn.Conv2d(2048, 512, 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=512,K=n_codes),
                  encoding.nn.View(-1, 512*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(512*n_codes, nclass),
                )  
                self.finalConc = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(4*nclass, nclass))
                  
                initialize_weights(self.enco1,self.finalConc)
                initialize_weights(self.cls5,self.conv6,self.cls6)

        def forward(self, x):
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                #print('inter4',inter4.shape)
                # G-stream
                x_g = self.conv5(inter4)
                weight_g = torch.norm(x_g,p=2, dim=1)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                weight_p = torch.norm(x_p,p=2, dim=1)
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)
                # Encodig stream
                inter4 =  self.enco5(inter4)
                weight_e = torch.norm(inter4,p=2, dim=1)
                out4 = self.enco1(inter4)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                return out, indices, weight_p, weight_g,weight_e, inter6


if __name__ == '__main__':
        input_test = torch.ones(10,3,448,448)
        net = DFL_ResNet()
        output_test = net(input_test)
        print(output_test)




#===================================================================================

class DFLTEN_ResNetConC_all(nn.Module):
        def __init__(self,resnet50, k = 20, nclass = 16,mode = 'ged', gmms=64, dataC=256  ):
                super(DFLTEN_ResNetConC_all, self).__init__()
                self.k = k
                self.nclass = nclass
                self.mode = mode
                self.gmms = gmms
                self.dataC = dataC
                # conv1_conv4
                layers_conv1_conv4 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool,
                ]
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        print ('check:', name)
                        layers_conv1_conv4.append(getattr(resnet50, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)
                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(resnet50, 'layer4'))
                conv5 = torch.nn.Sequential(*layers_conv5)

                conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

                # Feature extraction root
                self.conv1_conv4 = conv1_conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
                        nn.BatchNorm2d(nclass),
                        nn.ReLU(True),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # P-Stream
                self.conv6 = conv6
                self.pool6 = pool6
                self.cls6 = nn.Sequential(
                        nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
                        nn.AdaptiveAvgPool2d((1,1)),
                        )

                # Side-branch
                self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
                # encoding branch
                n_codes = self.gmms
                compres_dim = self.dataC
                self.enco = nn.Sequential(
                  nn.BatchNorm2d(k * nclass),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=k * nclass,K=n_codes),
                  encoding.nn.View(-1, k * nclass * n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(k * nclass * n_codes, nclass),
                )  
                self.enco1 = nn.Sequential(
                  nn.Conv2d(1024, compres_dim, 1), 
                  nn.BatchNorm2d(compres_dim),
                  nn.ReLU(inplace=True),
                  encoding.nn.Encoding(D=compres_dim,K=n_codes),
                  encoding.nn.View(-1, compres_dim*n_codes),
                  encoding.nn.Normalize(),
                  nn.Linear(compres_dim*n_codes, nclass),
                )  
                self.finalConc = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(4*nclass, nclass))
                self.finalConc3 = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(3*nclass, nclass))
                self.finalConc2 = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(2*nclass, nclass))
                  
                initialize_weights(self.enco1,self.finalConc,self.finalConc3)
                initialize_weights(self.cls5,self.conv6,self.cls6)

        def forward(self, x):
            if (self.mode == 'ged'or 'l' in self.mode):
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                #print('inter4',inter4.shape)
                # G-stream
                x_g = self.conv5(inter4)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                # Encodig stream
                out4 = self.enco1(inter4)
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                if self.mode == 'ged':
                    out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                    return out, indices
                if 'l' in self.mode:
                    return out1, out2, out3, out4, indices
            elif self.mode == 'gd':
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                #print('inter4',inter4.shape)
                # G-stream
                x_g = self.conv5(inter4)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)
                # Encodig stream
                #out4 = self.enco1(inter4)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                out = self.finalConc3(torch.cat((out1, out2, out3),1))
                return out, indices
            elif self.mode == 'ed':
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                #print('inter4',inter4.shape)
                # G-stream
                #x_g = self.conv5(inter4)
                #out1 = self.cls5(x_g)
                #out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)
                # Encodig stream
                out4 = self.enco1(inter4)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                out = self.finalConc3(torch.cat((out4, out2, out3),1))
                return out, indices
            elif self.mode == 'd':
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                #print('inter4',inter4.shape)
                # G-stream
                #x_g = self.conv5(inter4)
                #out1 = self.cls5(x_g)
                #out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)
                # Encodig stream
                #out4 = self.enco1(inter4)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                out = self.finalConc2(torch.cat((out2, out3),1))
                return out, indices


            elif self.mode == 'e':
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                # Encodig stream
                out = self.enco1(inter4)
                return out, out
            elif self.mode == 'g':
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                # G-stream
                x_g = self.conv5(inter4)
                out = self.cls5(x_g)
                out = out.view(batchsize, -1)
                return out, out
            elif self.mode == 'ge':
                batchsize = x.size(0)
                # Stem: Feature extraction
                inter4 = self.conv1_conv4(x)
                # Encodig stream
                out1 = self.enco1(inter4)
                # G-stream
                x_g = self.conv5(inter4)
                out2 = self.cls5(x_g)
                out2 = out2.view(batchsize, -1)
                out = self.finalConc2(torch.cat((out1, out2),1))
                return out, out
