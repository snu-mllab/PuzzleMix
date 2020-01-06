#!/usr/bin/env python
from __future__ import division

import os, sys, shutil, time, random
sys.path.append('..')
from glob import glob
import argparse
from distutils.dir_util import copy_tree
from shutil import rmtree
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import *
import models

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import numpy as np
from collections import OrderedDict, Counter
from load_data  import *
from helpers import *
from plots import *
from analytical_helper_script import run_test_with_mixup
#from attacks import run_test_adversarial, fgsm, pgd


model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'tiny-imagenet-200'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--data_dir', type = str, default = 'cifar10',
                        help='file where results are to be written')
parser.add_argument('--root_dir', type = str, default = 'experiments',
                        help='folder where results are to be stored')
parser.add_argument('--labels_per_class', type=int, default=5000, metavar='NL',
                    help='labels_per_class')
parser.add_argument('--valid_labels_per_class', type=int, default=0, metavar='NL',
                    help='validation labels_per_class')

parser.add_argument('--arch', metavar='ARCH', default='resnext29_8_64', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
parser.add_argument('--initial_channels', type=int, default=64, choices=(16,64))
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--delay', type=int, default=0, help='When to start mixing')

parser.add_argument('--train', type=str, default='vanilla', choices=['vanilla', 'mixup', 'mixup_hidden','cutout'])
parser.add_argument('--in_batch', type=str2bool, default=False)
parser.add_argument('-p', '--prob', type=float, default=1.0)
parser.add_argument('--mixup_alpha', type=float, help='alpha parameter for mixup')
parser.add_argument('--augmix', type=str2bool, default=False)
parser.add_argument('--dropout', type=str2bool, default=False,
                    help='whether to use dropout or not in final layer')

parser.add_argument('--emd', type=str2bool, default=False)
parser.add_argument('--proximal', type=str2bool, default=True)
parser.add_argument('--label_inter', type=str2bool, default=False)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--itermax', type=int, default=10)

parser.add_argument('--cutout', type=int, default=16, help='size of cut out')
parser.add_argument('--box', type=str2bool, default=False)
parser.add_argument('--graph', type=str2bool, default=False)
parser.add_argument('--method', type=str, default='mix', choices=['random', 'cut', 'cut_small', 'paste','mix'])
parser.add_argument('--block_num', type=int, default=-1)
parser.add_argument('--beta', type=float, default=1.2)
parser.add_argument('--gamma', type=float, default=1.0, help='[0,1]')
parser.add_argument('--eta', type=float, default=0.2)
parser.add_argument('--neigh_size', type=int, default=16)
parser.add_argument('--n_labels', type=int, default=3)
parser.add_argument('--label_cost', type=str, default='l2')

parser.add_argument('--jsd', type=str2bool, default=False)
parser.add_argument('--jsd_lam', type=float, default=12)
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--data_aug', type=int, default=1)
parser.add_argument('--adv_unpre', action='store_true', default=False,
                     help= 'the adversarial examples will be calculated on real input space (not preprocessed)')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints

parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--seed', default=0, type=int, help='manual seed')
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--log_off', type=str2bool, default=False)
parser.add_argument('--job_id', type=str, default='')

args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cudnn.benchmark = True

def experiment_name_non_mnist(dataset=args.dataset,
                    labels_per_class=args.labels_per_class,
                    arch=args.arch,
                    epochs=args.epochs,
                    delay=args.delay,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    decay= args.decay,
                    data_aug=args.data_aug,
                    train = args.train,
                    emd = args.emd,
                    label_inter = args.label_inter,
                    proximal = args.proximal,
                    box = args.box,
                    graph = args.graph,
                    method = args.method,
                    block_num = args.block_num,
                    beta = args.beta,
                    gamma=args.gamma,
                    eta=args.eta,
                    n_labels = args.n_labels,
                    neigh_size = args.neigh_size,
                    label_cost = args.label_cost,
                    reg = args.reg,
                    itermax = args.itermax,
                    in_batch = args.in_batch,
                    p = args.prob,
                    mixup_alpha = args.mixup_alpha,
                    job_id=args.job_id,
                    add_name=args.add_name,
                    jsd=args.jsd,
                    jsd_lam=args.jsd_lam,
                    augmix=args.augmix):

    exp_name = dataset
    exp_name += '{}'.format(labels_per_class)
    exp_name += '_arch_'+str(arch)
    exp_name += '_train_'+str(train)
    if label_inter:
        exp_name += '_label'
    if proximal:
        exp_name += '_prox'
    if emd:
        exp_name += '_emd'+str(itermax) + '_reg_' + str(reg)
    if box:
        exp_name += '_box_' + method
    if graph:
        exp_name += '_graph_' + method + '_block' + str(block_num) + '_beta_' + str(beta) + '_gamma_' + str(gamma) + '_eta_' +str(eta) + '_n_labels_' + str(n_labels) + '_neigh_' + str(neigh_size) + '_cost_' + str(label_cost)
    if augmix:
        exp_name += '_augmix'
    if in_batch:
        exp_name += '_inbatch'
    exp_name += '_p_' +str(p)
    exp_name += '_m_alpha_'+str(mixup_alpha)
    if dropout:
        exp_name+='_do_'+'true'
    else:
        exp_name+='_do_'+'False'
    exp_name += '_eph_'+str(epochs)
    if delay>0:
        exp_name += '_delay'+str(delay)
    exp_name +='_bs_'+str(batch_size)
    exp_name += '_lr_'+str(lr)
    exp_name += '_mom_'+str(momentum)
    exp_name +='_decay_'+str(decay)
    exp_name += '_data_aug_'+str(data_aug)
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if jsd:
        exp_name += '_jsd_'+str(jsd)
        exp_name += '_jsd_lam_'+str(jsd_lam)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)

    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name


def print_log(print_string, log):
    print("{}".format(print_string))
    if log is not None:
        log.write('{}\n'.format(print_string))
        log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
criterion = nn.CrossEntropyLoss().cuda()
mse_loss = nn.MSELoss().cuda()


def train(train_loader, model, optimizer, epoch, args, log, mean=None, std=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mixing_avg = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # import pdb; pdb.set_trace()
        # measure data loading time
        #print (input)
       
        #unique, counts = np.unique(target.numpy(), return_counts=True)
        #print (counts)
        #print(Counter(target.numpy()))
        #if i==100:
        #    break
        #import pdb; pdb.set_trace()
        target = target.long().cuda()
        data_time.update(time.time() - end)
        #import pdb; pdb.set_trace()

        ###  clean training####
        if (args.train == 'vanilla') or (epoch < args.delay):
            if args.augmix:
                input = input[0]
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var)
            loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == 'mixup':
            unary= None
            if args.augmix:
                m = np.float32(np.random.beta(1, 1))
                n = np.float32(np.random.beta(1, 1))
                input_clean = input[0].cuda()
                if args.emd:
                    input_aug1, _ = barycenter_conv2d(input[0], input[1], reg=1e-5, weights=torch.ones(input[0].size(0))*m, proximal=True, mean=mean.cuda(), std=std.cuda())
                    if args.jsd:
                        input_aug2, _ = barycenter_conv2d(input[0], input[2], reg=1e-5, weights=torch.ones(input[0].size(0))*n, proximal=True, mean=mean.cuda(), std=std.cuda())
                else:
                    input_aug1 = ((1-m)*input[0] + m*input[1]).cuda()
                    if args.jsd:
                        input_aug2 = ((1-n)*input[0] + n*input[2]).cuda()
                
                if args.jsd:
                    logits_all = model(torch.cat([input_clean, input_aug1.float(), input_aug2.float()], 0).squeeze())
                    logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, input[0].size(0))
                
                    p_clean = F.softmax(logits_clean, dim=1)
                    p_aug1 = F.softmax(logits_aug1, dim=1)
                    p_aug2 = F.softmax(logits_aug2, dim=1)

                    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                
                    loss_JSD = args.jsd_lam * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                              F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                              F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        
                input = input_aug1.float()
            
            if args.box or args.graph:
                input_var = Variable(input, requires_grad=True)
                target_var = Variable(target)
            
                model.eval()
                optimizer_input = torch.optim.SGD([input_var], lr=0.1)
                output = model(input_var)
                loss = criterion(output, target_var)
                optimizer_input.zero_grad()
                loss.backward()

                unary = torch.sqrt(torch.mean(input_var.grad **2, dim=1))
                model.train()
                
            input_var, target_var = Variable(input).float(), Variable(target)
            output, reweighted_target = model(input_var,target_var, mixup= True, mixup_alpha = args.mixup_alpha, p=args.prob, in_batch=args.in_batch,
                    emd=args.emd, proximal=args.proximal, reg=args.reg, itermax=args.itermax, label_inter=args.label_inter, mean=mean, std=std,
                    box=args.box, graph=args.graph, method=args.method, grad=unary, block_num=args.block_num, beta=args.beta, gamma=args.gamma, eta=args.eta, neigh_size=args.neigh_size, n_labels=args.n_labels, label_cost=args.label_cost)
            loss = bce_loss(softmax(output), reweighted_target)
            if args.augmix == 1 and args.jsd == 1:
                loss += loss_JSD

        elif args.train== 'mixup_hidden':
            unary= None
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var,target_var, mixup_hidden= True, mixup_alpha = args.mixup_alpha, p=args.prob, in_batch=args.in_batch,
                    emd=args.emd, proximal=args.proximal, reg=args.reg, itermax=args.itermax, label_inter=args.label_inter, mean=mean, std=std,
                    box=args.box, graph=args.graph, method=args.method, grad=unary, block_num=args.block_num, beta=args.beta, gamma=args.gamma, neigh_size=args.neigh_size, n_labels=args.n_labels)
            loss = bce_loss(softmax(output), reweighted_target)#mixup_criterion(target_a, target_b, lam)
            
        elif args.train == 'cutout':
            cutout = Cutout(1, args.cutout)
            cut_input = cutout.apply(input)
                
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            cut_input_var = torch.autograd.Variable(cut_input)
            output, reweighted_target = model(cut_input_var, target_var)
            loss = bce_loss(softmax(output), reweighted_target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mixing_avg.append((0.5 - reweighted_target[reweighted_target>0]).abs().mean().cpu().numpy())
        '''
        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
        '''

    if ((epoch) % 10 == 0):
        print_log("average mixing weight: {:.3f}".format(np.mean(mixing_avg) + 0.5), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
    with torch.no_grad():
        input_var = Variable(input)
        target_var = Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))
    top5.update(prec5.item(), input.size(0))

  print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses), log)
  return top1.avg, losses.avg


def test_robust(net, mean, std):
    net.eval()
    # Input Corruption Test
    dataset_cifar100_dist_list = glob('/home/janghyun/Codes/Wasserstein_Preprocessor/manifold_mixup/data/Cifar100-C/*.npy')
    label = np.load('/home/janghyun/Codes/Wasserstein_Preprocessor/manifold_mixup/data/Cifar100-C/labels.npy')

    for path in dataset_cifar100_dist_list:
        name = os.path.basename(path)[:-4]
        if name == 'labels':
            continue
            
        print("{:20}:".format(name), end=' ')
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
                   
        print("{:.2f}".format(prec1_total/500))


best_acc = 0
def main():
    ### set up the experiment directories########
    if not args.log_off:
        exp_name = experiment_name_non_mnist()
        exp_dir = args.root_dir+exp_name

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
    
        copy_script_to_folder(os.path.abspath(__file__), exp_dir)

        result_png_path = os.path.join(exp_dir, 'results.png')
        log = open(os.path.join(exp_dir, 'log.txt'.format(args.seed)), 'w')
        print_log('save path : {}'.format(exp_dir), log)
    else: 
        log = None

    global best_acc

    state = {k: v for k, v in args._get_kwargs()}
    print("")
    print_log(state, log)
    print("")
    print_log("Random Seed: {}".format(args.seed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    
    if args.adv_unpre:
        per_img_std = True
        train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset_unpre(args.data_aug, args.batch_size, 2 ,args.dataset, args.data_dir,  labels_per_class = args.labels_per_class, valid_labels_per_class = args.valid_labels_per_class)
    else:
        per_img_std = False
        train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(args.data_aug, args.batch_size, 2 ,args.dataset, args.data_dir,  labels_per_class = args.labels_per_class, valid_labels_per_class = args.valid_labels_per_class, mixup_alpha = args.mixup_alpha, augmix=args.augmix)
    

    if args.dataset == 'tiny-imagenet-200':
        stride = 2 
        width = 64
        mean = 127.5/255
        std = 127.5/255
    else:
        stride = 1
        width = 32
        mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]], dtype=torch.float32).view(1,3,1,1).cuda()
        std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]], dtype=torch.float32).view(1,3,1,1).cuda()


    print_log("=> creating model '{}'".format(args.arch), log)
    net = models.__dict__[args.arch](num_classes,args.dropout,per_img_std, stride).cuda()
    # print_log("=> network :\n {}".format(net), log)
    args.num_classes = num_classes

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)

    recorder = RecorderMeter(args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, criterion, log)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    # Main loop
    train_loss = []
    train_acc=[]
    test_loss=[]
    test_acc=[]
    
    def cost_matrix(width):
        C = np.zeros([width**2, width**2], dtype=np.float32)
        for m_i in range(width**2):
            i1 = m_i // width
            j1 = m_i % width
            for m_j in range(width**2):
                i2 = m_j // width
                j2 = m_j % width
                C[m_i,m_j]= abs(i1-i2)**2 + abs(j1-j2)**2
    
        C = C/(width-1)**2
        C = torch.tensor(C)
        return C

    #C_dict = {width:cost_matrix(width), width//2:cost_matrix(width//2), width//4:cost_matrix(width//4), width//8:cost_matrix(width//8) }

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        tr_acc, tr_acc5, tr_los  = train(train_loader, net, optimizer, epoch, args, log, mean=mean, std=std)

        # evaluate on validation set
        val_acc, val_los   = validate(test_loader, net, log)
        
        train_loss.append(tr_los)
        train_acc.append(tr_acc)
        test_loss.append(val_los)
        test_acc.append(val_acc)

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc
        
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if args.log_off:
            continue

        save_checkpoint({
          'epoch': epoch + 1,
          'arch': args.arch,
          'state_dict': net.state_dict(),
          'recorder': recorder,
          'optimizer' : optimizer.state_dict(),
        }, is_best, exp_dir, 'checkpoint.pth.tar')
        
        dummy = recorder.update(epoch, tr_los, tr_acc, val_los, val_acc)
        if (epoch+1) % 100 ==0:
            recorder.plot_curve(result_png_path)
    
        #import pdb; pdb.set_trace()
        train_log = OrderedDict()
        train_log['train_loss'] = train_loss
        train_log['train_acc']=train_acc
        train_log['test_loss']=test_loss
        train_log['test_acc']=test_acc
        
        pickle.dump(train_log, open(os.path.join(exp_dir,'log.pkl'), 'wb'))
        plotting(exp_dir)
    
    acc_var = np.maximum(np.max(test_acc[-10:])- np.median(test_acc[-10:]), np.median(test_acc[-10:]) - np.min(test_acc[-10:]))
    print_log("\nfinal 10 epoch acc (median) : {:.2f} (+- {:.2f})".format(np.median(test_acc[-10:]), acc_var), log)

    test_robust(net, mean, std)

    if not args.log_off:
        log.close()
    

if __name__ == '__main__':
    main()
