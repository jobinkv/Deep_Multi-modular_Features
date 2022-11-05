import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision
import sys
sys.path.insert(0, '../')
from utils import initialize_weights
#import encoding
import ipdb



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




if __name__ == '__main__':
        input_test = torch.ones(10,3,448,448)
        net = DFL_ResNet()
        output_test = net(input_test)
        print(output_test)




#===================================================================================

class DFLTEN_ResNetConC_all(nn.Module):
        def __init__(self,net, k = 20, nclass = 16,mode = 'ged', gmms=64, dataC=256  ):
                super(DFLTEN_ResNetConC_all, self).__init__()
                self.k = k
                self.nclass = nclass
                self.mode = mode
                self.gmms = gmms
                self.dataC = dataC
                for param in net.parameters():
                    param.requires_grad = False
                # conv1_conv4
                layers_conv1_conv4 = [
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                ]
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        print ('check:', name)
                        layers_conv1_conv4.append(getattr(net, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)
                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(net, 'layer4'))
                conv5 = torch.nn.Sequential(*layers_conv5)

                conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((24, 24), stride = (24, 24), return_indices = True)

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
                #self.enco = nn.Sequential(
                #  nn.BatchNorm2d(k * nclass),
                #  nn.ReLU(inplace=True),
                #  encoding.nn.Encoding(D=k * nclass,K=n_codes),
                #  encoding.nn.View(-1, k * nclass * n_codes),
                #  encoding.nn.Normalize(),
                #  nn.Linear(k * nclass * n_codes, nclass),
                #)  
                #self.enco1 = nn.Sequential(
                #  nn.Conv2d(1024, compres_dim, 1), 
                #  nn.BatchNorm2d(compres_dim),
                #  nn.ReLU(inplace=True),
                #  encoding.nn.Encoding(D=compres_dim,K=n_codes),
                #  encoding.nn.View(-1, compres_dim*n_codes),
                #  encoding.nn.Normalize(),
                #  nn.Linear(compres_dim*n_codes, nclass),
                #)  
                self.finalConc = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(4*nclass, nclass))
                self.finalConc3 = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(3*nclass, nclass))
                self.finalConc2 = nn.Sequential(
                  nn.ReLU(inplace=True),
                  nn.Linear(2*nclass, nclass))
                  
                initialize_weights(self.finalConc,self.finalConc3)
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
                #out4 = self.enco1(inter4)
                out4 = out1.clone()
                x_p, indices = self.pool6(x_p)
                inter6 = x_p
                out2 = self.cls6(x_p)
                out2 = out2.view(batchsize, -1)

                # Side-branch
                inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
                out3 = self.cross_channel_pool(inter6)
                out3 = out3.view(batchsize, -1)
                if not(self.training) and self.mode == 'ged':
                    out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                    return out, indices
                elif self.training and self.mode == 'ged':
                    out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                    return out, indices, out1, out2, out3, out4 
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
                #out4 = self.enco1(inter4)
                out4 = out2.clone()

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


            #elif self.mode == 'e':
            #    batchsize = x.size(0)
            #    # Stem: Feature extraction
            #    inter4 = self.conv1_conv4(x)
            #    # Encodig stream
            #    out = self.enco1(inter4)
            #    return out, out
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
                # G-stream
                x_g = self.conv5(inter4)
                out2 = self.cls5(x_g)
                out2 = out2.view(batchsize, -1)
                out1 = out2.clone()
                out = self.finalConc2(torch.cat((out1, out2),1))
                return out, out
#===================================================================================

class DFLTEN_VGG19ConC_all(nn.Module):
        def __init__(self, net, k = 20, nclass = 16,mode = 'ged', gmms=64, dataC=256  ):
                super(DFLTEN_VGG19ConC_all, self).__init__()
                self.k = k
                self.nclass = nclass
                self.mode = mode
                self.gmms = gmms
                self.dataC = dataC
                # conv1_conv4
                vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features
                conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11])
                conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:])
                conv6 = torch.nn.Conv2d(512, k * nclass, kernel_size = 1, stride = 1, padding = 0)
                pool6 = torch.nn.MaxPool2d((56, 56), stride = (56, 56), return_indices = True)
                '''
                layers_conv1_conv4 = [
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
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
                '''
                # Feature extraction root
                self.conv1_conv4 = conv1_conv4

                # G-Stream
                self.conv5 = conv5
                self.cls5 = nn.Sequential(
                        nn.Conv2d(512, nclass, kernel_size=1, stride = 1, padding = 0),
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
                  nn.Conv2d(512, compres_dim, 1), 
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
                if not(self.training) and self.mode == 'ged':
                    out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                    return out, indices
                elif self.training and self.mode == 'ged':
                    out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                    return out, indices,out2,out3 
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
#===================================================================================


class DFLTEN_ResNetConC_all_vis(nn.Module):
        def __init__(self,net, k = 20, nclass = 16,mode = 'ged', gmms=64, dataC=256  ):
                super(DFLTEN_ResNetConC_all_vis, self).__init__()
                self.k = k
                self.nclass = nclass
                self.mode = mode
                self.gmms = gmms
                self.dataC = dataC
                # conv1_conv4
                layers_conv1_conv4 = [
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                ]
                #ipdb.set_trace()
                for i in range(3):
                        name = 'layer%d' % (i + 1)
                        print ('check:', name)
                        layers_conv1_conv4.append(getattr(net, name))
                conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)
                # conv5
                layers_conv5 = []
                layers_conv5.append(getattr(net, 'layer4'))
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
                weight_g = torch.norm(x_g,p=2, dim=1)
                out1 = self.cls5(x_g)
                out1 = out1.view(batchsize, -1)
                # P-stream ,indices is for visualization
                x_p = self.conv6(inter4)
                weight_p = torch.norm(x_p,p=2, dim=1)
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
                if not(self.training) and self.mode == 'ged':
                    #out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                    out = torch.tanh(out1)+torch.tanh(out2)+torch.tanh(out3)+torch.tanh(out4)
                    return out, indices, weight_p, weight_g
                elif self.training and self.mode == 'ged':
                    #out = self.finalConc(torch.cat((out1, out2, out3, out4),1))
                    out = torch.tanh(out1)+torch.tanh(out2)+torch.tanh(out3)+torch.tanh(out4)
                    return out, indices,out2,out3, out4 
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
#===================================================================================

