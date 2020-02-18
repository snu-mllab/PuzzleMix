#!/usr/bin/env python
from __future__ import division

import os, sys, shutil, time, random
sys.path.append('..')
from glob import glob
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from utils import *
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle


model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Robustness test on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'tiny-imagenet-200'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--datapath', type=str, default='./')
parser.add_argument('--arch', metavar='ARCH', default='preactresnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: preresnet18)')
parser.add_argument('--ckpt', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--best', action='store_true', help='load model with best validation error while training')
parser.add_argument('--bce', action='store_true', help='use bce objective for attack')
args = parser.parse_args()

cudnn.benchmark = True

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if args.bce:
    criterion = nn.BCELoss().cuda()
else: 
    criterion = nn.CrossEntropyLoss().cuda()

softmax = nn.Softmax(dim=1).cuda()
mse_loss = nn.MSELoss().cuda()


if args.dataset == 'tiny-imagenet-200':
    stride = 2 
    width = 64 
    num_classes = 200
    mean = 127.5/255
    std = 127.5/255
elif args.dataset == 'cifar100':
    stride = 1
    width = 32
    num_classes = 100
    mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]], dtype=torch.float32).view(1,3,1,1).cuda()
    std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]], dtype=torch.float32).view(1,3,1,1).cuda()
else:
    raise AssertionError('given dataset is not supported!')


# Net load
print("=> creating model '{}'".format(args.arch))
net = models.__dict__[args.arch](num_classes, False, stride).cuda()

if not args.best:
    checkpoint = torch.load(args.ckpt+'/checkpoint.pth.tar')
else: 
    checkpoint = torch.load(args.ckpt+'/model_best.pth.tar')
checkpoint['state_dict'] = dict((key[7:], value) for (key, value) in checkpoint['state_dict'].items())
net.load_state_dict(checkpoint['state_dict'])

recorder = checkpoint['recorder']
print("=> loaded checkpoint '{}' (epoch {})" .format(args.ckpt, checkpoint['epoch']))

net.eval()


# Clean Dataset Test
test_transform = transforms.Compose([transforms.ToTensor()])
if args.dataset == 'tiny-imagenet-200':
    dataset = dset.ImageFolder(root=os.path.join(args.datapath, 'tiny-imagenet-200/val/images'), transform=test_transform)
elif args.dataset == 'cifar100':
    dataset = dset.CIFAR100(root=os.path.join(args.datapath, 'cifar100'), train=False, download=True, transform=test_transform)

testloader = DataLoader(dataset, batch_size=100, num_workers=2, pin_memory=True)

prec1_total = 0
prec5_total = 0
for batch_idx, (input, target) in enumerate(testloader):
    with torch.no_grad():
        input = input.cuda()
        target = target.cuda()
        
        output = net((input - mean)/std)
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        prec1_total += prec1.item()
        prec5_total += prec5.item()
        
print("clean accuracy: {:.2f}".format(prec1_total/100))


# Adversarial Robustness
param_set = [(8,8,1,False), (4,2,20,True)]

for eps, step, a_iter, random_init in param_set:
    print("")
    prec1_total = [0] * a_iter
    prec5_total = [0] * a_iter
    for batch_idx, (input, target) in enumerate(testloader):
        input_clean = input.cuda()
        target = target.cuda()
        input = input_clean.detach().clone()
        
        if random_init:
            noise = torch.zeros_like(input).uniform_(-eps/255., eps/255.)  
            input = torch.clamp(input + noise, 0, 1)

        for i in range(a_iter):
            input_var = Variable(input, requires_grad=True)
        
            optimizer_input = torch.optim.SGD([input_var], lr=0.1)
            if args.bce:
                output, reweighted_target = net((input_var - mean) /std, target)
                loss = criterion(softmax(output), reweighted_target)
            else:
                output = net((input_var - mean) /std)
                loss = criterion(output, target)
            optimizer_input.zero_grad()
            loss.backward()
            
            if i > 0:
                prec1, prec5 = accuracy(output, target, topk=(1,5))
                prec1_total[i-1] += prec1.item()
                prec5_total[i-1] += prec5.item()

            sign_data_grad = input_var.grad.sign()
            input = input + step / 255. * sign_data_grad
            eta = torch.clamp(input - input_clean, min=-eps/255., max=eps/255.)
            input = torch.clamp(input_clean + eta, min=0, max=1).detach()
                        
        with torch.no_grad():
            output = net((input - mean)/std)
            prec1, prec5 = accuracy(output, target, topk=(1,5))
            prec1_total[-1] += prec1.item()
                   
    if random_init:
        attack_name='PGD'
    else:
        attack_name='FGSM'

    for i in range(a_iter):
        print("{} attack (eps {}, step {}, iter: {}): {:.2f}".format(attack_name, eps, step, i+1, prec1_total[i]/100))



# Input Corruption Test
print('\nCorruption Robustness')
acc_list=[]
if args.dataset == 'tiny-imagenet-200':
    # Load corrupted dataset. Please check directory name. 
    dataset_tinyImagenet_dist_list = glob(os.path.join(args.datapath, 'tiny-imagenet-200-C/*'))

    for path in dataset_tinyImagenet_dist_list:
        name = os.path.basename(path)
        print("{:20}: ".format(name), end=' ')

        prec1_total = 0
        prec5_total = 0

        for sever in range(1,6):
            dataset = dset.ImageFolder(root=path+'/{}'.format(sever), transform=test_transform)
            testloader = DataLoader(dataset, batch_size=100, num_workers=2, pin_memory=True)

            for batch_idx, (input, target) in enumerate(testloader):
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
                    
                    output = net((input - mean)/std)
                    prec1, prec5 = accuracy(output, target, topk=(1,5))
                    prec1_total += prec1.item()
                    prec5_total += prec5.item()
            
        acc_list.append(prec1_total/len(testloader)/5)
        print("{:.2f}".format(prec1_total/len(testloader)/5))

elif args.dataset == 'cifar100':
    # Load corrupted dataset. Please check directory name. 
    dataset_cifar100_dist_list = glob(os.path.join(args.datapath, 'Cifar100-C/*.npy'))
    label = np.load(os.path.join(args.datapath, 'Cifar100-C/labels.npy'))

    for path in dataset_cifar100_dist_list:
        name = os.path.basename(path)[:-4]
        if name == 'labels':
            continue
            
        print("{:20}: ".format(name), end=' ')
        dataset_cifar100_dist = np.load(path)
        dataset_cifar100_dist = dataset_cifar100_dist.reshape(5, 100, 100, 32, 32, 3)
        
        prec1_total = 0
        prec5_total = 0
        for level in range(5):
            # print("(level{})".format(level+1), end='  ')
            
            for batch_idx, input in enumerate(dataset_cifar100_dist[level]):
                with torch.no_grad():
                    input = torch.tensor(input/255, dtype=torch.float32).permute(0,3,1,2).cuda()
                    target = torch.tensor(label[batch_idx*100: (batch_idx+1)*100], dtype=torch.int64).cuda()
                    
                    output = net((input - mean)/std)
                    prec1, prec5 = accuracy(output, target, topk=(1,5))
                    prec1_total += prec1.item()
                    prec5_total += prec5.item()
                   
        acc_list.append(prec1_total/500)
        print("{:.2f}".format(prec1_total/500))

print('\nmCE : {:.2f}'.format(np.mean(acc_list)))

