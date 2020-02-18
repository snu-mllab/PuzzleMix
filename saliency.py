import math
import numpy as np
import gco
from time import time
import os, sys, shutil, time, random
sys.path.append('..')
from distutils.dir_util import copy_tree
from shutil import rmtree
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import numpy as np
from collections import OrderedDict, Counter


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


def print_fig(input, target=None, title=None):
    fig, axes = plt.subplots(1,len(input),figsize=(3*len(input),3))

    if title:
        fig.suptitle(title, size=16)
        
    for i, ax in enumerate(axes):
        if len(input.shape) == 4:
            ax.imshow(input[i].permute(1,2,0).numpy())
        else :
            ax.imshow(input[i].numpy(), cmap='gray', vmin=0., vmax=1.)
        
        if target is not None:
            output = net((input[i].unsqueeze(0) - mean)/std)
            loss = criterion(output, target[i:i+1])
            ax.set_title("loss: {:.3f}\n pred: {}\n true : {}".format(loss, CIFAR100_LABELS_LIST[output.max(1)[1][0]], CIFAR100_LABELS_LIST[target[i]]))
        ax.axis('off')
    plt.show()



def get_images_edges_cvh(channel, height, width):
    idxs = np.arange(channel * height * width).reshape(channel, height, width)
    c_edges_from = np.r_[idxs[:-1, :, :], idxs[:1, :, :]].flatten()
    c_edges_to = np.r_[idxs[1:, :, :], idxs[-1:, :, :]].flatten()

    v_edges_from = idxs[:, :-1, :].flatten()
    v_edges_to = idxs[:, 1:, :].flatten()

    h_edges_from = idxs[:, :, :-1].flatten()
    h_edges_to = idxs[:, :, 1:].flatten()

    return c_edges_from, v_edges_from, h_edges_from, c_edges_to, v_edges_to, h_edges_to


_int_types = [np.int, np.intc, np.int32, np.int64, np.longlong]
_float_types = [np.float, np.float32, np.float64, np.float128]


def cut_3d_graph(unary_cost, pairwise_cost, cost_v, cost_h, cost_c, n_iter=-1, algorithm='swap'):
    assert len(unary_cost.shape)==4, "unary_cost dimension should be 4! for 3d graph model."
    energy_is_float = (unary_cost.dtype in _float_types) or \
                      (pairwise_cost.dtype in _float_types) or \
                      (cost_v.dtype in _float_types) or \
                      (cost_h.dtype in _float_types)

    channel, height, width, n_labels = unary_cost.shape

    gc = gco.GCO()
    gc.create_general_graph(channel * height * width, n_labels, energy_is_float)
    gc.set_data_cost(unary_cost.reshape([channel * height * width, n_labels]))

    c_edges_from, v_edges_from, h_edges_from, c_edges_to, v_edges_to, h_edges_to = get_images_edges_cvh(channel, height, width)
    v_edges_w = cost_v.flatten()
    assert len(v_edges_from) == len(v_edges_w), 'different sizes of edges %i and weights %i'% (len(v_edges_from), len(v_edges_w))
    h_edges_w = cost_h.flatten()
    assert len(h_edges_from) == len(h_edges_w), 'different sizes of edges %i and weights %i' % (len(h_edges_from), len(h_edges_w))
    c_edges_w = cost_c.flatten()
    assert len(c_edges_from) == len(c_edges_w), 'different sizes of edges %i and weights %i'% (len(c_edges_from), len(c_edges_w))

    edges_from = np.r_[c_edges_from, v_edges_from, h_edges_from]
    edges_to = np.r_[c_edges_to, v_edges_to, h_edges_to]
    edges_w = np.r_[c_edges_w, v_edges_w, h_edges_w]

    gc.set_all_neighbors(edges_from, edges_to, edges_w)
    gc.set_smooth_cost(pairwise_cost)

    if algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        gc.swap(n_iter)

    labels = gc.get_labels()
    gc.destroy_graph()

    return labels


def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, label_cost='l2'):
    block_num = unary1.shape[0]
    large_val = 1000 * block_num ** 2 
    
    if n_labels == 2:
        prior=  eta * np.array([-np.log(alpha + 1e-8), -np.log(1 - alpha + 1e-8)]) / block_num ** 2
    elif n_labels == 3:
        prior= eta * np.array([-np.log(alpha**2 + 1e-8), -np.log(2 * alpha * (1-alpha) + 1e-8), -np.log((1 - alpha)**2 + 1e-8)]) / block_num ** 2
    elif n_labels == 4:
        prior= eta * np.array([-np.log(alpha**3 + 1e-8), -np.log(3 * alpha **2 * (1-alpha) + 1e-8), 
                             -np.log(3 * alpha * (1-alpha) **2 + 1e-8), -np.log((1 - alpha)**3 + 1e-8)]) / block_num ** 2
        
    unary_cost =  (large_val * np.stack([(1-lam) * unary1 + lam * unary2 + prior[i] for i, lam in enumerate(np.linspace(0,1, n_labels))], axis=-1)).astype(np.int32)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32) 

    for i in range(n_labels):
        for j in range(n_labels):
            if label_cost == 'l1':
                pairwise_cost[i, j] = abs(i-j) / (n_labels-1)**1
            elif label_cost == 'l2':
                pairwise_cost[i, j] = (i-j)**2 / (n_labels-1)**2
            elif laabel_cost == 'l4':
                pairwise_cost[i, j] = (i-j)**4 / (n_labels-1)**4
            else:
                raise AssertionError("label cost should be one of ['l1', 'l2', 'l4']")

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)
    
    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y, algorithm='swap')/(n_labels-1)
    mask = labels.reshape(block_num, block_num)
    
    return mask

def graphcut_multi_float(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, label_cost='l2', dim=2, beta_c=0.0):
    block_num = unary1.shape[-1]
    
    if n_labels == 2:
        prior=  eta * np.array([-np.log(alpha + 1e-8), -np.log(1 - alpha + 1e-8)]) / block_num ** 2
    elif n_labels == 3:
        prior= eta * np.array([-np.log(alpha**2 + 1e-8), -np.log(2 * alpha * (1-alpha) + 1e-8), -np.log((1 - alpha)**2 + 1e-8)]) / block_num ** 2
    elif n_labels == 4:
        prior= eta * np.array([-np.log(alpha**3 + 1e-8), -np.log(3 * alpha **2 * (1-alpha) + 1e-8), 
                             -np.log(3 * alpha * (1-alpha) **2 + 1e-8), -np.log((1 - alpha)**3 + 1e-8)]) / block_num ** 2
        
    unary_cost = np.stack([(1-lam) * unary1 + lam * unary2 + prior[i] for i, lam in enumerate(np.linspace(0,1, n_labels))], axis=-1)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32) 

    for i in range(n_labels):
        for j in range(n_labels):
            if label_cost == 'l1':
                pairwise_cost[i, j] = abs(i-j) / (n_labels-1)**1
            elif label_cost == 'l2':
                pairwise_cost[i, j] = (i-j)**2 / (n_labels-1)**2
            elif laabel_cost == 'l4':
                pairwise_cost[i, j] = (i-j)**4 / (n_labels-1)**4
            else:
                raise AssertionError("label cost should be one of ['l1', 'l2', 'l4']")

    pw_x = pw_x + beta
    pw_y = pw_y + beta
    
    if dim==2:
        labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y, algorithm='swap')/(n_labels-1)
        mask = labels.reshape(block_num, block_num)
    elif dim==3:
        pw_c = beta_c * np.ones(shape=(3, block_num, block_num))
        labels = 1.0 - cut_3d_graph(unary_cost, pairwise_cost, pw_x, pw_y, pw_c, algorithm='swap')/(n_labels-1)
        mask = labels.reshape(3, block_num, block_num)
    else:
        raise AssertionError('dim should be 2 or 3')
            
    return mask

#### random
def neigh_penalty(input1, input2, k, dim=2):
    pw_x = input1[:,:,:-1,:] - input2[:,:,1:,:]
    pw_y = input1[:,:,:,:-1] - input2[:,:,:,1:]

    pw_x = pw_x[:,:,k-1::k,:]
    pw_y = pw_y[:,:,:,k-1::k]

    if dim==2:
        pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1,k))
        pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k,1))
    elif dim==3:
        pw_x = F.avg_pool2d(pw_x.abs(), kernel_size=(1,k))
        pw_y = F.avg_pool2d(pw_y.abs(), kernel_size=(k,1))
        
    return pw_x, pw_y


def mixup_block_ratio(input1, input2, grad1, grad2, target1, target2, block_num=2, method='random',
                      alpha=0.5, beta=0., gamma=0., eta=0.2, neigh_size=2, n_labels=2, device='cpu'):
    batch_size, _, _, width = input1.shape
    if block_num == -1:
        block_num = 2**np.random.randint(1, 5)
    block_size = width // block_num
    neigh_size = min(neigh_size, block_size)

    ratio = np.zeros([batch_size, 1])
    beta = beta/block_num/16
    
    mask = np.zeros([batch_size, 1, width, width])
    unary1 = F.avg_pool2d(grad1, block_size)
    unary2 = F.avg_pool2d(grad2, block_size)   
    unary1 = unary1 / unary1.view(batch_size, -1).sum(1).view(batch_size, 1, 1)
    unary2 = unary2 / unary2.view(batch_size, -1).sum(1).view(batch_size, 1, 1)    
    
    if method == 'random':
        for (x,y) in points:
            val = np.random.binomial(n=1, p=alpha, size=(batch_size, 1, 1, 1))            
            mask[:, :, x:x+block_size, y:y+block_size] += val 
            ratio += val[:,:,0,0]
            mask = torch.tensor(mask, dtype=torch.float32)
            
    elif method == 'graph_mix':
        mask=[]
        unary1 = unary1.detach().cpu().numpy() 
        unary2 = unary2.detach().cpu().numpy()
        
        ### Add unnormalize tern !
        input1_pool = F.avg_pool2d(input1, neigh_size)
        input2_pool = F.avg_pool2d(input2, neigh_size)

        pw_x = torch.zeros([batch_size, 2, 2, block_num-1, block_num], device=device)
        pw_y = torch.zeros([batch_size, 2, 2, block_num, block_num-1], device=device)
        k = block_size//neigh_size
        
        pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(input2_pool, input2_pool, k)
        pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(input2_pool, input1_pool, k)
        pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(input1_pool, input2_pool, k)
        pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(input1_pool, input1_pool, k)

        pw_x = pw_x.detach().cpu().numpy()
        pw_y = pw_y.detach().cpu().numpy()
        
        for i in range(batch_size):
            pw_x_i = beta * gamma * pw_x[i]
            pw_y_i = beta * gamma * pw_y[i]
            
            unary2[i][:-1,:] += (pw_x_i[1,0] + pw_x_i[1,1])/2.
            unary1[i][:-1,:] += (pw_x_i[0,1] + pw_x_i[0,0])/2.
            unary2[i][1:,:] += (pw_x_i[0,1] + pw_x_i[1,1])/2.
            unary1[i][1:,:] += (pw_x_i[1,0] + pw_x_i[0,0])/2.

            unary2[i][:,:-1] += (pw_y_i[1,0] + pw_y_i[1,1])/2
            unary1[i][:,:-1] += (pw_y_i[0,1] + pw_y_i[0,0]) / 2
            unary2[i][:,1:] += (pw_y_i[0,1] + pw_y_i[1,1])/2
            unary1[i][:,1:] += (pw_y_i[1,0] + pw_y_i[0,0]) / 2

            pw_x_i = (pw_x_i[1,0] + pw_x_i[0,1] - pw_x_i[1,1] - pw_x_i[0,0])/2
            pw_y_i = (pw_y_i[1,0] + pw_y_i[0,1] - pw_y_i[1,1] - pw_y_i[0,0])/2
            
            mask.append(graphcut_multi(unary2[i], unary1[i], pw_x_i, pw_y_i, alpha,  beta, eta, n_labels))
            ratio[i,0] = mask[i].sum()
            
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
        mask = F.interpolate(mask.unsqueeze(1), size=width)

    else:
        raise AssertionError("wrong mixup method type !!")

    ratio = torch.tensor(ratio/block_num**2, dtype=torch.float32, device='cuda')    
    return mask, ratio


from lapjv import lapjv


def mask_transport(mask, grad_pool, eps=0.01, n_iter=None, t_type='full', verbose=False, device='cpu'):
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]
    if n_iter is None:
        n_iter = int(block_num)

    C = cost_matrix(block_num, device=device).unsqueeze(0)
    
    if t_type =='half':
        z = mask
    else :
        z = (mask>0).float()
    
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)    
    if t_type=='full' or t_type=='half':
        for step in range(n_iter):
            # row resolve, with tie-breaking
            row_best = (cost - 1e-10 * cost.min(-2, keepdim=True)[0]).min(-1)[1]
            plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

            # column resolve
            cost_fight = plan * cost + (1 - plan)
            col_best = cost_fight.min(-2)[1]
            plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
            plan_lose = (1-plan_win) * plan

            # Note unary and mask are less than 1
            cost += plan_lose
            if verbose:
                print("(step {}) obj: {:.4f}".format(step, torch.sum((plan_win * z.reshape(-1, 1, block_num**2)) * cost)))
                
    elif t_type=='exact':
        plan_indices = []
        cost_np = cost.cpu().numpy()
        for i in range(batch_size):
            plan_indices.append(torch.tensor(lapjv(cost_np[i])[0], dtype=torch.long))
        plan_indices = torch.stack(plan_indices, dim=0)
        plan_win = torch.zeros_like(cost).scatter_(-1, plan_indices.unsqueeze(-1), 1)
    
    return plan_win


def transport_image(img, plan, batch_size, block_num, block_size):
    batch_size = img.shape[0]
    channel= img.shape[1]
    input_patch = img.reshape([batch_size, channel, block_num, block_size, block_num*block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, channel, block_num, block_num, block_size, block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, channel, block_num**2, block_size, block_size]).permute(0,1,3,4,2).unsqueeze(-1)

    input_transport = plan.transpose(-2,-1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(0,1,4,2,3)
    input_transport = input_transport.reshape([batch_size, channel, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, channel, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, channel, block_num * block_size, block_num * block_size])
    
    return input_transport


def cost_matrix(width=16, dist=2, device='cpu'):
    size = width**2
    C=np.zeros([size, size], dtype=np.float32)
    for m_i in range(size):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(size):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i,m_j]= abs(i1-i2)**2 + abs(j1-j2)**2

    C = C/(width-1)**2
    C = torch.tensor(C, device=device)
    return C

def transport_image_loop(img, plan, batch_size, block_num, block_size):
    plan_move = plan.max(-2)[1] * plan.sum(-2)
    plan_move_idx = torch.nonzero(plan_move)

    input_transport = img.clone()
    for idx in plan_move_idx:
        ex_idx = idx[0]
        target_idx = idx[1]
        source_idx = plan_move[ex_idx, target_idx]

        target_idx = [target_idx//block_num, target_idx%block_num]
        source_idx = [source_idx//block_num, source_idx%block_num]

        input_transport[ex_idx,:, target_idx[0]*block_size: (target_idx[0]+1)*block_size, target_idx[1]*block_size: (target_idx[1]+1)*block_size] =\
        img[ex_idx,:, source_idx[0]*block_size: (source_idx[0]+1)*block_size, source_idx[1]*block_size: (source_idx[1]+1)*block_size]

    return input_transport


def mixup_box(input1, input2, grad1, grad2, target1, target2, block_num=2, method='random', alpha=0.5):
    batch_size, _, height, width = input1.shape
    ratio = np.zeros([batch_size, 1])
    
    if method == 'random':
        rh = np.sqrt(1 - alpha) * height
        rw = np.sqrt(1 - alpha) * width
        rx = np.random.uniform(rh / 2, height - rh / 2)
        ry = np.random.uniform(rw / 2, width - rw / 2)

        x1 = int(np.clip(rx - rh / 2, a_min=0., a_max=height))
        x2 = int(np.clip(rx + rh / 2, a_min=0., a_max=height))
        y1 = int(np.clip(ry - rw / 2, a_min=0., a_max=width))
        y2 = int(np.clip(ry + rw / 2, a_min=0., a_max=width))
        input1[:, :, x1:x2, y1:y2] = input2[:, :, x1:x2, y1:y2]
        ratio += 1 - (x2-x1)*(y2-y1)/(width*height)
        grad1[:, x1:x2, y1:y2] = grad2[:, x1:x2, y1:y2]
        
    elif method == 'cut':
        rh = int(np.sqrt(1 - alpha) * height)
        rw = int(np.sqrt(1 - alpha) * width)
        grad_conv = torch.nn.functional.conv2d(grad2.unsqueeze(1), weight=grad2.new_ones(size=(1,1,rh,rw)))
        grad_max = grad_conv.view(batch_size,-1).max(1)[1]
        for idx in range(batch_size):
            x1 = grad_max[idx]//grad_conv.shape[-1]
            y1 = grad_max[idx]%grad_conv.shape[-1]
            x2 = x1 + rh
            y2 = y1 + rw
            input1[idx, :, x1:x2, y1:y2] = input2[idx, :, x1:x2, y1:y2]
            ratio[idx] = 1 - float((x2-x1)*(y2-y1))/(width*height)
    
    elif method == 'paste':
        rh = int(np.sqrt(1 - alpha) * height)
        rw = int(np.sqrt(1 - alpha) * width)
        grad_conv1 = torch.nn.functional.conv2d(grad1.unsqueeze(1), weight=grad1.new_ones(size=(1,1,rh,rw)))
        grad_min = grad_conv1.view(batch_size,-1).min(1)[1]
        grad_conv2 = torch.nn.functional.conv2d(grad2.unsqueeze(1), weight=grad2.new_ones(size=(1,1,rh,rw)))
        grad_max = grad_conv2.view(batch_size,-1).max(1)[1]

        for idx in range(batch_size):
            x_paste = grad_min[idx]//grad_conv1.shape[-1]
            y_paste = grad_min[idx]%grad_conv1.shape[-1]
            x_cut = grad_max[idx]//grad_conv2.shape[-1]
            y_cut = grad_max[idx]%grad_conv2.shape[-1]
            
            input1[idx, :, x_paste:x_paste+rh, y_paste:y_paste+rw] = input2[idx, :, x_cut:x_cut+rh, y_cut:y_cut+rw]
            ratio[idx] = 1 - float(rh*rw)/(width*height)
    else:
        raise AssertionError("wrong mixup method type !!")
    
    ratio = torch.tensor(ratio, dtype=torch.float32, device='cuda')
    return input1, ratio, grad1


from torchvision.models.resnet import load_state_dict_from_url
import random

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, random_indices=None, alpha=0):
        layer_mix = -1
        if random_indices is not None:
            layer_mix = random.randint(0,3)
        
        if layer_mix==0:
            x = alpha*x + (1-alpha)*x[random_indices]
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if layer_mix==1:
            x = alpha*x + (1-alpha)*x[random_indices]
            
        x = self.layer1(x)
        if layer_mix==2:
            x = alpha*x + (1-alpha)*x[random_indices]
            
        x = self.layer2(x)
        if layer_mix==3:
            x = alpha*x + (1-alpha)*x[random_indices]

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x, random_indices=None, alpha=0):
        return self._forward_impl(x, random_indices, alpha)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

criterion = nn.CrossEntropyLoss()
criterion_batch = nn.CrossEntropyLoss(reduction='none')
resnet = resnet18(pretrained=True).cuda()
resnet.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean_torch = torch.tensor(mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
std_torch = torch.tensor(std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

test_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
     ])

traindir = os.path.join('/data_large/readonly/ImageNet/', 'train')
train_dataset = datasets.ImageFolder(traindir, test_transform)

valdir = os.path.join('/data_large/readonly/ImageNet/', 'val')
val_dataset = datasets.ImageFolder(valdir, test_transform)


# Transport label3
mean_torch = mean_torch.cuda()
std_torch = std_torch.cuda()

def saliency_compare(n_labels = 2,
                    beta = 0.5,
                    gamma = 0.3,
                    eta = 0.2,
                    eps = 0.2,
                    n_batch=50,
                    ):
    batch_size = 20

    alpha_len = 21
    alpha_list = np.linspace(0,1,alpha_len)

    inputmix_dict = {'saliency': [0]*alpha_len, 'loss': [0]*alpha_len, 
                     'acc': [0]*alpha_len, 'acc2': [0]*alpha_len, 'tv': [0]*alpha_len}
    manimix_dict = {'saliency': [0]*alpha_len, 'loss': [0]*alpha_len, 
                     'acc': [0]*alpha_len, 'acc2': [0]*alpha_len, 'tv': [0]*alpha_len}
    cutmix_dict = {'saliency': [0]*alpha_len, 'loss': [0]*alpha_len, 
                     'acc': [0]*alpha_len, 'acc2': [0]*alpha_len, 'tv': [0]*alpha_len}
    puzzlemix_dict = {'saliency': [0]*alpha_len, 'loss': [0]*alpha_len, 
                     'acc': [0]*alpha_len, 'acc2': [0]*alpha_len, 'tv': [0]*alpha_len}

    seed = 0
    block_num = 16
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    random_indices = [i for i in range(1, batch_size)]
    random_indices.append(0)
    random_indices = np.array(random_indices)

    block_size = 224//block_num
    t_type ='full'
    n_iter = block_num

    for batch_idx, (images, target) in enumerate(train_loader):
        images_orig = images.cuda()
        target = target.cuda()
        print(batch_idx)
        if batch_idx == n_batch:
            break

        input_var = Variable((images_orig - mean_torch)/std_torch, requires_grad=True)
        output = resnet(input_var)
        loss = criterion(output, target)
        loss.backward()

        unary_orig = torch.sqrt(torch.mean(input_var.grad **2, dim=1))  
        unary_orig = unary_orig / unary_orig.view(unary_orig.shape[0], -1).sum(1).view(unary_orig.shape[0], 1, 1)

        for k, alpha in enumerate(alpha_list):
            if batch_idx==0:
                print(alpha)
            images = images_orig.detach()
            unary = unary_orig.detach()

            mask, puzzlemix_ratio = mixup_block_ratio(images.clone(), images[random_indices].clone(), unary.clone(), unary[random_indices].clone(),
                                                      target.float(), target[random_indices].float(),
                                                      block_num=block_num, method='graph_mix', alpha=alpha, beta=beta, gamma=gamma, eta=eta,
                                            n_labels=n_labels, neigh_size=2, device='cuda')

            z = F.avg_pool2d(mask.squeeze(), block_size)
            grad_pool = F.avg_pool2d(unary, block_size)
            grad_pool = grad_pool / grad_pool.view(grad_pool.shape[0], -1).sum(1).view(grad_pool.shape[0], 1, 1)

            # input1
            plan = mask_transport(z, grad_pool, eps=eps, n_iter=n_iter, t_type=t_type, device='cuda')
            input1_transport = transport_image(images, plan, batch_size, block_num, block_size)
            grad1_transport = transport_image(unary.unsqueeze(1), plan, batch_size, block_num, block_size)[:,0,:,:]

            # input2
            plan = mask_transport(1-z, grad_pool[random_indices], eps=eps, t_type=t_type, device='cuda')
            input2_transport = transport_image(images[random_indices], plan, batch_size, block_num, block_size)
            grad2_transport = transport_image(unary[random_indices].unsqueeze(1), plan, batch_size, block_num, block_size)[:,0,:,:]

            # various mixup methods
            puzzlemix = mask * input1_transport.detach() + (1-mask) * input2_transport.detach()
            puzzlemix_sal = mask.squeeze() * grad1_transport.detach() + (1-mask).squeeze() * grad2_transport.detach()

            cutmix, cutmix_ratio, cutmix_sal= mixup_box(images.clone(), images[random_indices].clone(), unary.clone(), unary[random_indices].clone(),
                                      target.float(), target[random_indices].float(), method='random', alpha=alpha)
            inputmix = alpha * images + (1-alpha) * images[random_indices]


            # results
            cutmix_var = Variable((cutmix - mean_torch)/std_torch, requires_grad=True)
            output = resnet(cutmix_var)
            loss = (criterion_batch(output, target) * cutmix_ratio + criterion_batch(output, target[random_indices]) * (1. - cutmix_ratio)).mean()
            pred = torch.topk(output, 2, dim=1)[1]
            correct = 0
            correct2 = 0
            for idx in range(len(pred)):
                if (pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]):
                    correct += 1

                if target[idx] == target[random_indices][idx]:
                    if (pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]):
                        correct2 += 1
                else:
                    if (((pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]))
                        and ((pred[idx][1] == target[idx]) or (pred[idx][1] == target[random_indices][idx]))):
                        correct2 += 1
            tv_loss = (cutmix[:,:,1:,:] - cutmix[:,:,:-1,:]).abs().mean() + (cutmix[:,:,:,1:] - cutmix[:,:,:,:-1]).abs().mean()

            cutmix_dict['saliency'][k] += cutmix_sal.sum().detach()/batch_size
            cutmix_dict['loss'][k] += loss.detach()
            cutmix_dict['acc'][k] += correct
            cutmix_dict['acc2'][k] += correct2
            cutmix_dict['tv'][k] += tv_loss.detach()
            if batch_idx == 0:
                print('(cutmix)     saliency: {:.2f}, loss: {:.2f}, correct: {}, correct2: {}, tv: {:.3f}'.format(cutmix_sal.sum()/batch_size, loss, correct, correct2, tv_loss))


            inputmix_var = Variable((inputmix - mean_torch)/std_torch, requires_grad=True)
            output = resnet(inputmix_var)
            loss = criterion(output, target) * alpha + criterion(output, target[random_indices]) * (1. - alpha)
            pred = torch.topk(output, 2, dim=1)[1]
            correct = 0
            correct2 = 0
            for idx in range(len(pred)):
                if (pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]):
                    correct += 1

                if target[idx] == target[random_indices][idx]:
                    if (pred[idx][0] == target[idx]):
                        correct2 += 1
                else:
                    if (((pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]))
                        and ((pred[idx][1] == target[idx]) or (pred[idx][1] == target[random_indices][idx]))):
                        correct2 += 1
            tv_loss = (inputmix[:,:,1:,:] - inputmix[:,:,:-1,:]).abs().mean() + (inputmix[:,:,:,1:] - inputmix[:,:,:,:-1]).abs().mean()

            inputmix_dict['saliency'][k] += unary.sum().detach()/batch_size
            inputmix_dict['loss'][k] += loss.detach()
            inputmix_dict['acc'][k] += correct
            inputmix_dict['acc2'][k] += correct2            
            inputmix_dict['tv'][k] += tv_loss.detach()
            if batch_idx == 0:
                print('(input mix)  saliency: {:.2f}, loss: {:.2f}, correct: {}, correct2: {}, tv: {:.3f}'.format(unary.sum()/batch_size, loss, correct, correct2, tv_loss))


            input_var = Variable((images - mean_torch)/std_torch, requires_grad=True)
            output = resnet(input_var, random_indices, alpha)
            loss = criterion(output, target) * alpha + criterion(output, target[random_indices]) * (1. - alpha)
            pred = torch.topk(output, 2, dim=1)[1]
            correct = 0
            correct2 = 0
            for idx in range(len(pred)):
                if (pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]):
                    correct += 1

                if target[idx] == target[random_indices][idx]:
                    if (pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]):
                        correct2 += 1
                else:
                    if (((pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]))
                        and ((pred[idx][1] == target[idx]) or (pred[idx][1] == target[random_indices][idx]))):
                        correct2 += 1
            manimix_dict['loss'][k] += loss.detach()
            manimix_dict['acc'][k] += correct
            manimix_dict['acc2'][k] += correct2  
            if batch_idx == 0:        
                print('(manifold)   saliency: {:.2f},  loss: {:.2f}, correct: {}, correct2: {}'.format(0.0001, loss, correct, correct2))


            puzzlemix_var = Variable((puzzlemix - mean_torch)/std_torch, requires_grad=True)
            output = resnet(puzzlemix_var)
            loss = (criterion_batch(output, target) * puzzlemix_ratio + criterion_batch(output, target[random_indices]) * (1. - puzzlemix_ratio)).mean()
            pred = torch.topk(output, 2, dim=1)[1]
            correct = 0
            correct2 = 0
            for idx in range(len(pred)):
                if (pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]):
                    correct += 1

                if target[idx] == target[random_indices][idx]:
                    if (pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]):
                        correct2 += 1
                else:
                    if (((pred[idx][0] == target[idx]) or (pred[idx][0] == target[random_indices][idx]))
                        and ((pred[idx][1] == target[idx]) or (pred[idx][1] == target[random_indices][idx]))):
                        correct2 += 1
            tv_loss = (puzzlemix[:,:,1:,:] - puzzlemix[:,:,:-1,:]).abs().mean() + (puzzlemix[:,:,:,1:] - puzzlemix[:,:,:,:-1]).abs().mean()

            puzzlemix_dict['saliency'][k] += puzzlemix_sal.sum().detach()/batch_size  
            puzzlemix_dict['loss'][k] += loss.detach()
            puzzlemix_dict['acc'][k] += correct
            puzzlemix_dict['acc2'][k] += correct2            
            puzzlemix_dict['tv'][k] += tv_loss.detach()
            if batch_idx == 0:
                print('(puzzle mix) saliency: {:.2f}, loss: {:.2f}, correct: {}, correct2: {}, tv: {:.3f}'.format(puzzlemix_sal.sum()/batch_size, loss, correct, correct2, tv_loss))
    
    return inputmix_dict, manimix_dict, cutmix_dict, puzzlemix_dict



if __name__ == '__main__':
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_labels', type=int)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--eta', type=float)
 
    args = parser.parse_args()
    save_name = '../saliency_label{}_beta{}_gamma{}_eta{}.pkl'.format(args.n_labels, args.beta, args.gamma, args.eta)
    print(save_name)

    inputmix_dict1, manimix_dict1, cutmix_dict1, puzzlemix_dict1 = saliency_compare(n_labels = args.n_labels, beta = args.beta, gamma = args.gamma, eta = args.eta, eps = 0.8, n_batch=5000)

    save_files = [cutmix_dict1, inputmix_dict1, manimix_dict1, puzzlemix_dict1, 5000]
    
    with open(save_name, 'wb') as handle:
        pickle.dump(save_files, handle, protocol=pickle.HIGHEST_PROTOCOL)


