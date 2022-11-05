from  math import cos, radians
import os
from PIL import Image
import torchvision.transforms as standard_transforms
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
sys.path.insert(0, '../')
from utils import joint_transforms as simul_transforms
from utils import transforms as extended_transforms
from models import *
from htmlCreator import logHtml, ploteIt

from utils import check_mkdir, evaluate, AverageMeter
cudnn.benchmark = True
import argparse
from pytorch_metric_learning import losses, miners


parser = argparse.ArgumentParser(description='Train deep multimodular features for documents')
parser.add_argument('-e','--exp', type=str, default='exp1',
                    help='name of output folder')
parser.add_argument('-d','--dataset', type=str, default='rvl_cdip',
                    help='choose the dataset: ')
parser.add_argument('-n','--net', type=str, default='resnext101',
                    help='choose the network architecture: psp or mfcn')
parser.add_argument('-s','--snapshot', type=str, default='',
                    help='give the trained model for further training')
parser.add_argument('-l','--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('-f','--feature', type=str, default='ged',
                    help='learning rate')
parser.add_argument('-i','--imgsize', type=int, default=384,
                    help='image size')
parser.add_argument('-k','--discrim', type=int, default=20,
                    help='discriminative filter size')
parser.add_argument('-g','--gmms', type=int, default=64,
                    help='number of gmms')
parser.add_argument('-c','--compresDim', type=int, default=512,
                    help='number of gmms')
parser.add_argument('-a','--totalEppoch', type=int, default=20,
                    help='total number of epoches')
parser.add_argument('-b','--fakeSave', type=bool, default=True,
                    help='save a fake model')
parser.add_argument('-t','--turn', type=int, default=0,
                    help='0--> turn the basemodel, 1--> turn the basemodel GED, 2--> learn the dataset finetured GED')

args = parser.parse_args()
print ('The exp arguments are ',args.exp,args.net,args.dataset)
# if the fineturn is false take the finurned network version
if args.turn==2 or args.turn==3:
   args.net= args.net+args.dataset
if args.turn==0:
   args.feature = 'X'
   
ckpt_path = '/ssd_scratch/cvit/jobinkv/'#input folder

exp_name = args.exp #output folder

dataset = args.dataset

if dataset=='rvl_cdip':
        from rvl_cdip import doc
elif dataset=='book_cover':
        from book_cover import doc
elif dataset=='docSeg':
        from docSeg import doc
elif dataset=='script':
        from cvsi import doc
else:
        print ('please specify the dataset')
network = args.net
snapShort=args.snapshot
Dataroot = '/ssd_scratch/cvit/jobinkv' #location of data
root1 = '/ssd_scratch/cvit/jobinkv/pyTorchPreTrainedModels/'#location of pretrained model
if args.net=='resnet101':
        net_arch =  models.resnet101()
        model_name = 'resnet'
        path = os.path.join(root1,  'resnet101-5d3b4d8f.pth')
if args.net=='resnet152':
        net_arch =  models.resnet152()
        model_name = 'resnet'
        path = os.path.join(root1,  'resnet152-b121ed2d.pth')
if args.net=='resnet152_rvl':
        net_arch =  models.resnet152()
        model_name = 'resnet'
        path = os.path.join(root1,  'epoch_8_loss_0.14488_testAcc_0.91075.pth')
if args.net=='resnext101rvl_cdip':
        net_arch =  models.resnext101_32x8d()
        model_name = 'resnet'
        path = os.path.join(root1,  'epoch_5_loss_0.14987_testAcc_0.91182_resNext101.pth')
if args.net=='resnext101script':
        net_arch =  models.resnext101_32x8d()
        model_name = 'resnet'
        path = os.path.join(root1,  'resnext101_script.pth')
if args.net=='resnext101docSeg':
        net_arch =  models.resnext101_32x8d()
        model_name = 'resnet'  
        path = os.path.join(root1,  'resnext101_docSeg.pth')
if args.net=='resnext101book_cover':
        net_arch =  models.resnext101_32x8d()
        model_name = 'resnet'  
        path = os.path.join(root1,  'resnext101_book_cover.pth')
if args.net=='resnext101':
        net_arch =  models.resnext101_32x8d()
        model_name = 'resnet'
        path = os.path.join(root1,  'resnext101_32x8d-8ba56ff5.pth')
if args.net=='resnet50':
        net_arch =  models.resnet50()
        model_name = 'resnet'
        path = os.path.join(root1,  'resnet50-19c8e357.pth')
if args.net=='vgg16':
        net_arch =  models.vgg16()
        model_name = 'vgg'
        path = os.path.join(root1,  'vgg16-397923af.pth')
if args.net=='vgg19':
        net_arch =  models.vgg19()
        model_name = 'vgg'
        path = os.path.join(root1,  'vgg19-dcbb9e9d.pth')

args = {
    'train_batch_size': 32*torch.cuda.device_count(),
    'alpha':0, # encoding
    'beta':0, # discriminative
    'gama':0, # global
    'lr': args.lr,
    'lr_decay': 0.9,
    'max_iter':1e5,
    'input_size': args.imgsize,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot':snapShort,  # empty string denotes learning from scratch
    'print_freq': 200,
    'max_epoch':args.totalEppoch,
    'dataset':dataset,
    'network':args.net,
    'features':args.feature,
    'discrim':args.discrim,
    'gmms': args.gmms,
    'jobid':args.exp,
    'fakesave':args.fakeSave,
    'Auxilary_loss_contribution':0.5,
    'compresDim':args.compresDim,
    'finetune':args.turn # finetune the base net by replacing the penaltimate fully connected layer
}
if 'e' in  args['features']:
    args['alpha'] = 1
if 'd' in  args['features']:
    args['beta'] = 1
if 'g' in  args['features']:
    args['gama'] = 1

sep_iou_val=[]
sep_iou_test=[]
curr_iter_print=0
print (args)
def main(train_args):
    print (doc.num_classes)
   
    if train_args['network'] in  ['dfl_vgg16']:
          print('DFL-CNN <==> Part2 : Load Network  <==> Begin')
          net = DFL_VGG16(k = 10, nclass = 16)    
     
    elif  train_args['finetune']== 2 and train_args['network'] == 'resnext101'+train_args['dataset']:
          net = net_arch
          num_features = net.fc.in_features
          fc = list(net.fc.children()) # Remove last layer
          fc.extend([nn.Linear(num_features, doc.num_classes)]) # Add our layer with 4 outputs
          net.fc = nn.Sequential(*fc) # Replace the model classifier
          net.load_state_dict(torch.load(path))
          net = DFLTEN_ResNetConC_all(net,k=train_args['discrim'], nclass = doc.num_classes,mode=train_args['features'],\
          gmms=train_args['gmms'], dataC=train_args['compresDim'])  
          print ('Yes mode 2 is started') 
    elif  train_args['finetune']== 1 and train_args['network'] == 'resnext101':
          net = net_arch
          net.load_state_dict(torch.load(path))
          net = DFLTEN_ResNetConC_all(net,k=train_args['discrim'], nclass = doc.num_classes,mode=train_args['features'])   
    elif train_args['finetune']== 0 and train_args['network'] == 'resnext101':
          net = net_arch
          net.load_state_dict(torch.load(path))
          num_features = net.fc.in_features
          fc = list(net.fc.children()) # Remove last layer
          fc.extend([nn.Linear(num_features, doc.num_classes)]) # Add our layer with 4 outputs
          net.fc = nn.Sequential(*fc) # Replace the model classifier
    elif train_args['finetune']== 3 and train_args['network'] == 'resnext101'+train_args['dataset']:
          net = net_arch
          num_features = net.fc.in_features
          fc = list(net.fc.children()) # Remove last layer
          fc.extend([nn.Linear(num_features, doc.num_classes)]) # Add our layer with 4 outputs
          net.fc = nn.Sequential(*fc) # Replace the model classifier
          net.load_state_dict(torch.load(path))
          
    # metric loss
    metric_loss = losses.TripletMarginLoss(margin = 0.2)
    mining_func = miners.TripletMarginMiner(margin = 0.2, type_of_triplets = "semihard")
    print ("number of cuda devices = ", torch.cuda.device_count())
    if len(train_args['snapshot']) == 0:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print ('training resumes from ' + train_args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, train_args['snapshot'])))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
    if train_args['finetune']<3:
        net.train()
    # set of image transormations
    if train_args['dataset'] == 'rvl-cdip':
        mean_std = ([0.9584, 0.9588, 0.9586], [0.1246, 0.1223, 0.1224])
    else:
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    val_simul_transform = simul_transforms.Scale(train_args['input_size'])
    if train_args['dataset'] == 'script':
        train_input_transform = standard_transforms.Compose([
            extended_transforms.multiscaleImg(train_args['input_size'],[100,40,80,160]),
            standard_transforms.Resize((train_args['input_size'], train_args['input_size']), interpolation=Image.Resampling.LANCZOS),
            extended_transforms.RandomGaussianBlur(),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        val_input_transform = standard_transforms.Compose([
            extended_transforms.multiscaleImg(train_args['input_size'],[100,40,80,160]),
            standard_transforms.Resize((train_args['input_size'], train_args['input_size']), interpolation=Image.Resampling.LANCZOS),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
    else:
        train_input_transform = standard_transforms.Compose([
            standard_transforms.Resize((train_args['input_size'], train_args['input_size']), interpolation=Image.Resampling.LANCZOS),
            extended_transforms.RandomGaussianBlur(),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        val_input_transform = standard_transforms.Compose([
            standard_transforms.Resize((train_args['input_size'], train_args['input_size']), interpolation=Image.Resampling.LANCZOS),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
    target_transform = extended_transforms.MaskToTensor()
    # data loser section
    train_set = doc.DOC('train',Dataroot, transform=train_input_transform,
                        target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=train_args['train_batch_size'], num_workers=1, shuffle=True, drop_last = True)
    val_set = doc.DOC('test',Dataroot,  transform=val_input_transform,
                      target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=train_args['train_batch_size'], num_workers=1, shuffle=False, drop_last = True)
    tes_set = doc.DOC('test',Dataroot,  transform=val_input_transform,
                      target_transform=target_transform, path=True)
    tes_loader = DataLoader(tes_set, batch_size=train_args['train_batch_size'], num_workers=1, shuffle=False, drop_last = True)
    # loss functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = train_args['lr'], momentum=0.9, weight_decay = 0.000005)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    # train the network
    train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader,tes_loader, metric_loss,mining_func)

    test_loss, test_acc,clas_acc = final_evaluate(tes_loader, net,criterion,train_args)
    print ('Final accuracy', str(clas_acc))

def getBestModel(path, splitingChar = '_', ComparePos = 5):
    Imagelist= os.listdir(path)
    accuracy = 0
    index= -1
    for num,item in enumerate(Imagelist):
        if '.pth' in item:
            print (item.split(splitingChar)[ComparePos])
            accuracy_temp = float(item.split(splitingChar)[ComparePos])
            if accuracy<=accuracy_temp:
                accuracy = accuracy_temp
                index = num
    if index>=0:
        return Imagelist[index]
    else:
        return 'Not found'
sep_acc_test = []
def train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader,tes_loader, metric_loss,mining_func):
    print('===:Training Starts:===')
    plot_losses_train = []
    plot_losses_trainG = []
    plot_losses_auxilaryLoss = []
    plot_losses_trainD = []
    plot_losses_trainE = []
    plot_losses_test = []
    plot_losses_val=[]
    plot_losses_aux=[]
    plot_lr=[]
    plot_accuracy=[]
    print ('first max iter: ', train_args['max_iter'], len(train_loader))
    train_args['max_iter'] = train_args['max_epoch']*len(train_loader)
    print ('calculated max iter: ', train_args['max_iter'])
    while True:
        train_main_loss = AverageMeter()
        train_main_lossG = AverageMeter()
        train_main_lossD = AverageMeter()
        train_main_lossE = AverageMeter()
        train_auxilaryLoss = AverageMeter()
        train_acc = AverageMeter()
        net.train(True)
        curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 0.5*(1+cos(radians(float(curr_iter)*180/train_args['max_iter'] ))
                   )*train_args['lr']
            inputs, labels = data
            N=inputs.size(0)
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            if 'l' in train_args['features']:
                out1, out2, out3, out4, indices = net(inputs)
                #ipdb.set_trace()
                embeddings  = torch.cat((out1, out2, out3, out4),1)
                #indices_tuple = mining_func(embeddings, labels)
                #loss = metric_loss(embeddings,labels,indices_tuple)
                out = out1*train_args['gama'] + out2*train_args['beta'] + 0.1 * out3 *train_args['beta'] + out4* train_args['alpha']
  
                preds = out.squeeze().data.max(1)[1]
                loss1 = criterion(out1, labels)
                loss21 = criterion(out2, labels)
                loss22 = criterion(out3, labels)
                loss3 = criterion(out4, labels)
                loss2 = loss21 + 0.1* loss22
                loss = loss1*train_args['gama'] + loss2*train_args['beta'] + loss3*train_args['alpha']
            elif  train_args['finetune']==0:
                out = net(inputs)
                preds = out.squeeze().data.max(1)[1]
                loss = criterion(out, labels)
            elif  train_args['features']=='ged':
                out, _, outG, outP, outC, outE = net(inputs)
                preds = out.squeeze().data.max(1)[1]
                out_aux = outP + 0.1 * outC #P global maxpool, C: cross channel pooling
                loss1 = criterion(out, labels)
                aux_loss = criterion(out_aux, labels)
                aux_loss_enc = criterion(outE, labels)
                aux_loss_glo = criterion(outG, labels)
                loss  = loss1 + train_args['Auxilary_loss_contribution']*aux_loss \
                              + train_args['Auxilary_loss_contribution']*aux_loss_enc \
                              + train_args['Auxilary_loss_contribution']*aux_loss_glo
      
            elif train_args['features'] in ['g', 'e', 'd', 'ge', 'ed', 'gd']:
                out, _  = net(inputs)
                preds = out.squeeze().data.max(1)[1]
                loss = criterion(out, labels)

            loss.backward()
            optimizer.step()
            tr_acc = torch.sum(preds==labels.data)
            train_acc.update(tr_acc,N)
            train_main_loss.update(loss.item(),N)
            if 'l' in train_args['features']:
                train_main_lossG.update(loss.item(),N)
                train_main_lossD.update(loss.item(),N)
                train_main_lossE.update(loss.item(),N)
            if train_args['features'] == 'ged':
                train_auxilaryLoss.update(aux_loss.item(),N)
                
            curr_iter += 1
            if (i + 1) % train_args['print_freq'] == 0:
                print ('[ epoch %d], [iter %d/%d],[train main loss: %.5f]' %( 
                  curr_epoch, i + 1, len(train_loader), 
                  train_main_loss.avg))
        if curr_epoch>0:
                test_loss, test_acc,clas_acc = evaluate(tes_loader, net,criterion,train_args)
                print ('[ epoch %d],[train main loss %.5f],[testing accuracy %.5f] on %s (%s,%s)' % (
                curr_epoch,train_main_loss.avg,test_acc,train_args['dataset'],train_args['features'],train_args['network']))
                snapshot_name = 'epoch_%d_loss_%.5f_testAcc_%.5f_%s_%s_%s_K_%s_G_%s_C_%s' % ( 
                            curr_epoch, train_main_loss.avg, test_acc,train_args['features'],\
                            train_args['network'],train_args['dataset'],train_args['discrim'],train_args['gmms'],train_args['compresDim'])
                plot_losses_train.append(train_main_loss.avg)
                plot_losses_test.append(test_loss)
                plot_accuracy.append(test_acc)
                plot_lr.append(optimizer.param_groups[0]['lr']*(1/train_args['lr']))
                clas_acc.insert(0,curr_iter)
                sep_acc_test.append(clas_acc)
                if 'l' in train_args['features']:
                    plot_losses_trainG.append(train_main_lossG.avg)
                    plot_losses_trainD.append(train_main_lossD.avg)
                    plot_losses_trainE.append(train_main_lossE.avg)
                if train_args['features'] == 'ged':
                    plot_losses_auxilaryLoss.append(train_auxilaryLoss.avg)
                # save good models
                if train_args['fakesave']:
                    temp_save = torch.tensor([[1, 2, 3], [4, 5, 6]])
                else:
                    temp_save = net.module.state_dict()
                if test_acc>0.91:
                        #torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
                        torch.save(temp_save, os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
                        print ('model saved in ', os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
                elif (train_args['dataset'] == 'book_cover' and test_acc>0.14):
                        #torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
                        torch.save(temp_save, os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
                        print ('model saved in ', os.path.join(ckpt_path, exp_name,  snapshot_name + '.pth'))
                elif (train_args['dataset'] == 'docSeg' and test_acc>0.65):
                        #torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
                        torch.save(temp_save, os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
                        print ('model saved in ', os.path.join(ckpt_path, exp_name,  snapshot_name + '.pth'))
                       
                ploteIt(plot_losses_train, plot_losses_test, plot_accuracy, plot_losses_trainG,plot_losses_trainD,
                    plot_losses_trainE, plot_losses_auxilaryLoss,plot_lr, train_args['print_freq'],train_args['jobid'])
                logHtml(sep_acc_test,train_args,doc.tablehead)
        curr_epoch += 1
        if curr_iter >= train_args['max_iter']:
                break

def evaluate(data_loader, net, criterion,train_args):
    acc=0
    no_imgs = 0
    test_loss = AverageMeter()
    confusion_matrix = torch.zeros(doc.num_classes,doc.num_classes)
    net.eval()
    gts_all, predictions_all = [], []
    for vi, data in enumerate(data_loader):
        inputs, gts, paths = data
        N = inputs.size(0)
        with torch.no_grad():
                inputs = inputs.cuda()
                gts = gts.cuda()
                if 'l' in train_args['features']: # loss sum
                    out1, out2, out3, out4, indices = net(inputs)
                    out = out1*train_args['gama'] + out2*train_args['beta'] \
                          +  0.1 * out3 *train_args['beta'] + out4* train_args['alpha']
                    preds = out.squeeze().data.max(1)[1]
                elif  train_args['finetune']==0 or train_args['finetune']==3: # basemodel train
                    out = net(inputs)
                    preds = out.squeeze().data.max(1)[1]
                else: # concatination
                    out, _ = net(inputs)
                    preds = out.squeeze().data.max(1)[1]
                tempss = gts.data
                test_loss.update(criterion(out, gts).item(), N)
                for t, p in zip(tempss,preds):
                    confusion_matrix[t.long(), p.long()] += 1
                #st()
    sep_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    sep_acc = sep_acc.tolist()
    mean_acc = confusion_matrix.diag().sum()/confusion_matrix.sum()
    sep_acc.append(mean_acc.numpy())
    net.train()
    return  test_loss.avg, mean_acc.numpy(), sep_acc

def final_evaluate(data_loader, net, criterion,train_args):
    acc=0
    no_imgs = 0
    test_loss = AverageMeter()
    confusion_matrix = torch.zeros(doc.num_classes,doc.num_classes)
    net.eval()
    gts_all, predictions_all = [], []
    for vi, data in enumerate(data_loader):
        inputs, gts, paths = data
        N = inputs.size(0)
        with torch.no_grad():
                inputs = inputs.cuda()
                gts = gts.cuda()
                if 'l' in train_args['features']: # loss sum
                    out1, out2, out3, out4, indices = net(inputs)
                    out = out1*train_args['gama'] + out2*train_args['beta'] \
                          +  0.1 * out3 *train_args['beta'] + out4* train_args['alpha']
                    preds = out.squeeze().data.max(1)[1]
                elif  train_args['finetune']==0 or train_args['finetune']==3: # basemodel train
                    out = net(inputs)
                    preds = out.squeeze().data.max(1)[1]
                else: # concatination
                    out, _ = net(inputs)
                    preds = out.squeeze().data.max(1)[1]
                tempss = gts.data
                test_loss.update(criterion(out, gts).item(), N)
                for t, p, img_path in zip(tempss,preds,paths):
                    confusion_matrix[t.long(), p.long()] += 1
    sep_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    sep_acc = sep_acc.tolist()
    mean_acc = confusion_matrix.diag().sum()/confusion_matrix.sum()
    sep_acc.append(mean_acc.numpy())
    return  test_loss.avg, mean_acc.numpy(), sep_acc

if __name__ == '__main__':
    main(args)
