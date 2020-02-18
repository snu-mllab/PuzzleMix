import logging
import os
import datetime
import torchvision.models as models
import math
import torch
import yaml
from easydict import EasyDict
import shutil
import numpy as np
import torch.nn.functional as F
import gco

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30./n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(output_path, evaluate):
    if not os.path.isdir(os.path.join('output', output_path)):
        os.makedirs(os.path.join('output', output_path))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join('output', output_path, 'eval.txt' if evaluate else 'log.txt'),'w'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(output_path))
    logger.info('User: {}'.format(os.getenv('USER')))
    return logger

def get_model_names():
	return sorted(name for name in models.__dict__
    		if name.islower() and not name.startswith("__")
    		and callable(models.__dict__[name]))

def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*'*int(rem_len/2) + msg + '*'*int(rem_len/2)\

def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))
        
    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v
        
    # Add the output path
    config.output_name = '{:s}'.format(args.output_prefix)
    return config


def save_checkpoint(state, is_best, filepath, epoch):
    filename = os.path.join(filepath, f'checkpoint_epoch{epoch}.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

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
    return C


cost_matrix_dict = {'2':np.expand_dims(cost_matrix(2), 0), '4':np.expand_dims(cost_matrix(4), 0), '8':np.expand_dims(cost_matrix(8), 0), '16':np.expand_dims(cost_matrix(16), 0)}

'''code for graphmix'''
_int_types = [np.int, np.intc, np.int32, np.int64, np.longlong]
_float_types = [np.float, np.float32, np.float64, np.float128]


def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, label_cost='l2'):
    block_num = unary1.shape[0]
    
    large_val = 1000 * block_num ** 2 
    
    if n_labels == 2:
        prior= eta * np.array([-np.log(alpha + 1e-8), -np.log(1 - alpha + 1e-8)]) / block_num ** 2
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
                pairwise_cost[i, j] = abs(i-j) / (n_labels-1)
            elif label_cost == 'l2':
                pairwise_cost[i, j] = (i-j)**2 / (n_labels-1)**2
            elif laabel_cost == 'l4':
                pairwise_cost[i, j] = (i-j)**4 / (n_labels-1)**4
            else:
                raise AssertionError('Wrong label cost type!')

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)

    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y, algorithm='swap')/(n_labels-1)
    mask = labels.reshape(block_num, block_num)

    return mask


def graphcut_multi_float(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, label_cost='l2', beta_c=0.0):
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
                pairwise_cost[i, j] = abs(i-j) / (n_labels-1)
            elif label_cost == 'l2':
                pairwise_cost[i, j] = (i-j)**2 / (n_labels-1)**2
            elif laabel_cost == 'l4':
                pairwise_cost[i, j] = (i-j)**4 / (n_labels-1)**4
            else:
                raise AssertionError("label cost should be one of ['l1', 'l2', 'l4']")

    pw_x = pw_x + beta
    pw_y = pw_y + beta
    
    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y, algorithm='swap')/(n_labels-1)
    mask = labels.reshape(block_num, block_num)
            
    return mask
  
  
def neigh_penalty(input1, input2, k):
    pw_x = input1[:,:,:-1,:] - input2[:,:,1:,:]
    pw_y = input1[:,:,:,:-1] - input2[:,:,:,1:]

    pw_x = pw_x[:,:,k-1::k,:]
    pw_y = pw_y[:,:,:,k-1::k]

    pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1,k))
    pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k,1))

    return pw_x, pw_y

from scipy.ndimage.filters import gaussian_filter
random_state = np.random.RandomState(None)

def get_mask(input1, grad1, block_num, indices, alpha=0.5, beta=1.2, gamma=0.5, eta=0.2, neigh_size=2, n_labels=3, sigma=1.0, mean=None, std=None):
    input2 = input1[indices]
    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    neigh_size = min(neigh_size, block_size)

    alpha = np.ones([batch_size]) * alpha
    beta = beta/block_num/16
    
    grad1_pool = F.avg_pool2d(grad1, block_size)
        
    #mask = np.zeros([batch_size, 1, width, width])
    unary1_torch = grad1_pool / grad1_pool.view(batch_size, -1).sum(1).view(batch_size, 1, 1)

    mask=[]
    unary2_torch = unary1_torch[indices]
        
    input1_pool = F.avg_pool2d(input1 * std + mean, neigh_size)
    input2_pool = input1_pool[indices]

    pw_x = torch.zeros([batch_size, 2, 2, block_num-1, block_num], device='cuda')
    pw_y = torch.zeros([batch_size, 2, 2, block_num, block_num-1], device='cuda')

    k = block_size//neigh_size

    pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(input2_pool, input2_pool, k)
    pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(input2_pool, input1_pool, k)
    pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(input1_pool, input2_pool, k)
    pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(input1_pool, input1_pool, k)

    pw_x = beta * gamma * pw_x
    pw_y = beta * gamma * pw_y
        
    unary1 = unary1_torch.clone()
    unary2 = unary2_torch.clone()
    
    unary2[:,:-1,:] += (pw_x[:,1,0] + pw_x[:,1,1])/2.
    unary1[:,:-1,:] += (pw_x[:,0,1] + pw_x[:,0,0])/2.
    unary2[:,1:,:] += (pw_x[:,0,1] + pw_x[:,1,1])/2.
    unary1[:,1:,:] += (pw_x[:,1,0] + pw_x[:,0,0])/2.

    unary2[:,:,:-1] += (pw_y[:,1,0] + pw_y[:,1,1])/2.
    unary1[:,:,:-1] += (pw_y[:,0,1] + pw_y[:,0,0])/2.
    unary2[:,:,1:] += (pw_y[:,0,1] + pw_y[:,1,1])/2.
    unary1[:,:,1:] += (pw_y[:,1,0] + pw_y[:,0,0])/2.

    pw_x = (pw_x[:,1,0] + pw_x[:,0,1] - pw_x[:,1,1] - pw_x[:,0,0])/2
    pw_y = (pw_y[:,1,0] + pw_y[:,0,1] - pw_y[:,1,1] - pw_y[:,0,0])/2

    unary1 = unary1.detach().cpu().numpy()
    unary2 = unary2.detach().cpu().numpy()
    pw_x = pw_x.detach().cpu().numpy()
    pw_y = pw_y.detach().cpu().numpy()

    for i in range(batch_size):
        mask.append(graphcut_multi(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha[i], beta, eta, n_labels))

    mask = torch.tensor(mask, dtype=torch.float32, device='cuda')
    mask = mask.unsqueeze(1)

    return mask

def transport(input1, grad1, indices, block_num, mask, eps=0.8, t_type='full'):
    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    
    grad1_pool = F.avg_pool2d(grad1, block_size)
    unary1_torch = grad1_pool / grad1_pool.view(batch_size, -1).sum(1).view(batch_size, 1, 1)
        
    plan = mask_transport(mask, unary1_torch, eps=eps, t_type=t_type)
    input1 = transport_image_loop(input1, plan, batch_size, block_num, block_size)
    plan = mask_transport(1-mask, unary1_torch[indices], eps=eps, t_type=t_type)
    input2 = transport_image_loop(input1[indices], plan, batch_size, block_num, block_size)
    
    mask = F.interpolate(mask, size=width)
    ratio = torch.mean(mask, dim=(1, 2, 3))
    
    return mask*input1 + (1-mask)*input2, ratio

def mask_transport(mask, grad_pool, eps=0.01, t_type='full'):
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]
    C = torch.tensor(cost_matrix_dict[str(block_num)], device='cuda')
    
    if t_type=='full':
        z = (mask>0).float()
    else:
        z = mask
    
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)
    n_iter = int(np.sqrt(block_num))    
    
    # row and col
    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1-plan_win) * plan

        cost += plan_lose
    
    #return plan_win
    plan_move = (plan_win - torch.eye(block_num**2, device='cuda'))>0
    return plan_move

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


def transport_image(img, plan, batch_size, block_num, block_size):
    batch_size, _, w, h = img.shape
    input_patch = img.reshape([batch_size, 3, block_num, block_size, w]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, 3, block_num, block_num, block_size, block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, 3, block_num**2, block_size, block_size]).permute(0,1,3,4,2).unsqueeze(-1)
   
    print(plan.transpose(-2, -1).unsqueeze(1).unsqueeze(1).unsqueeze(1).shape)
    print(input_patch.shape)
    input_transport = plan.transpose(-2,-1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(0,1,4,2,3)
    print(input_transport.shape)
    input_transport = input_transport.reshape([batch_size, 3, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, 3, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, 3, block_num * block_size, block_num * block_size])
    
    return input_transport

  
def create_val_folder(data_set_path):
    """
    Used for Tiny-imagenet dataset
    Copied from https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/train.py
    This method is responsible for separating validation images into separate sub folders,
    so that test and val data can be read by the pytorch dataloaders
    """
    path = os.path.join(data_set_path, 'val/images')  # path where validation data is present now
    filename = os.path.join(data_set_path, 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))

if __name__ == "__main__":
    create_val_folder('data/tiny-imagenet-200')  # Call method to create validation image folders


