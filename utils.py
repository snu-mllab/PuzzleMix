import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import augmentations

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


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
  
  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
  
    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    
    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)
    

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)


def unsqueeze3(tensor):
    return tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

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
      
def K_prox(x, H, transpose=False):
    c = x.shape[1]
    width = x.shape[2]
    if transpose:
        return torch.matmul(x.view([-1,c,1,width**2]), H).view([-1,c,width, width])
    else:
        return torch.matmul(H, x.view([-1,c,width**2,1])).view([-1,c,width, width])

def K(x, xi1):
      return torch.matmul(torch.matmul(xi1, x), xi1)

def nan_recover(tensor, thres=1e100):
    tensor[tensor > thres] = thres
    return tensor

def barycenter_conv2d(input1, input2, reg=2e-3, weights=None, numItermax=10000, proximal=True,
                      stopThr=1e-9, stabThr=1e-15,
                      return_alpha=False, cost=None, norm_type='sum', mean=None, std=None, v_max=None, device='cuda'):
    r"""
    Parameters
    ----------
    input1, input2 : tensor, shape (batch_size, num_channel, width, height)
    reg : float
        Regularization term >0
    weights : tensor, shape (batch_size)
        Weights of each image on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    b : tensor, shape (batch_size, num_channel, width, height)
        2D Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters

    """
   
    input1 = input1.to(device).double()
    input2 = input2.to(device).double()

    if weights is None:
        weights = torch.ones(input1.shape[0]).to(device) / 2
    elif weights.shape[0] == 1:
        weights = torch.ones(input1.shape[0]).to(device) * weights
    else:
        pass
    
    if mean is not None and std is not None:
        mean = mean.to(device).double()
        std = std.to(device).double()

    weights = weights.to(device).double()
    stabThr = torch.tensor(stabThr).to(device).double()
    if mean is not None:
        input1 = std * input1 + mean + stabThr
        input2 = std * input2 + mean + stabThr

    sum1 = input1.sum(2, keepdim=True).sum(3, keepdim=True)
    sum2 = input2.sum(2, keepdim=True).sum(3, keepdim=True)
    total_sum = unsqueeze3(weights)*sum1 + unsqueeze3(1.-weights)*sum2
    
    if norm_type == 'max':
        max1 = input1.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        max2 = input2.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        total_max = unsqueeze3(weights)*max1 + unsqueeze3(1.-weights)*max2
    
    input1_normalized = input1 / sum1
    input2_normalized = input2 / sum2
    
    U1 = torch.ones_like(input1).double()
    U2 = torch.ones_like(input2).double()
    KV1 = torch.ones_like(input1).double()
    KV2 = torch.ones_like(input2).double()
    
    cpt = 0
    err = 1

    # build the convolution operator
    batch_size = input1.shape[0]
    c = input1.shape[1]
    width = input1.shape[2]
    
    if not proximal:
        t = torch.linspace(0, 1, input1.shape[2])
        [Y, X] = torch.meshgrid(t, t)
        xi1 = torch.exp(-(X - Y)**2 / reg).to(device).double()

        while (err > stopThr and cpt < numItermax):
            if cpt == 0:
                bold = torch.zeros_like(input1)
            else :
                bold = b
            cpt = cpt + 1

            V1 = input1_normalized / torch.max(stabThr, K(U1, xi1))
            KV1 = K(V1, xi1)
            b = torch.einsum('i,ijkl->ijkl', weights, torch.log(torch.max(stabThr, U1 * KV1))) 

            V2 = input2_normalized / torch.max(stabThr, K(U2, xi1))
            KV2 = K(V2, xi1)
            b += torch.einsum('i,ijkl->ijkl', 1-weights, torch.log(torch.max(stabThr, U2 * KV2)))

            b = torch.exp(b)

            U1 = b / torch.max(stabThr, KV1)
            U2 = b / torch.max(stabThr, KV2)

            if cpt % 10 == 1:
                err = torch.sum(torch.abs(bold - b))
                
    else:
        t = torch.linspace(0, 1, width)
        [Y, X] = torch.meshgrid(t, t)
        xi1_init = torch.exp(-(X - Y)**2 / reg).cuda().double()
        xi1 = torch.ones_like(xi1_init).cuda().double()
        U1 = torch.ones_like(input1).cuda().double()
        U2 = torch.ones_like(input2).cuda().double()
        alpha1 = torch.ones_like(input1).cuda().double()
        beta1 = torch.ones_like(input1).cuda().double()
        alpha2 = torch.ones_like(input1).cuda().double()
        beta2 = torch.ones_like(input1).cuda().double()
    
        for t in range(1): #p_iter=1
            xi1 *= xi1_init
            for _ in range(1):
                V1 = input1 / torch.max(stabThr, beta1 * K(U1 * alpha1, xi1))
                KV1 = alpha1 * K(V1 * beta1 , xi1)
                b = torch.einsum('i,ijkl->ijkl', weights, torch.log(torch.max(stabThr, U1 * KV1)))
                V2 = input2 / torch.max(stabThr, beta2 * K(U2 * alpha2, xi1))
                KV2 = alpha2 * K(V2 * beta2 , xi1)
                b += torch.einsum('i,ijkl->ijkl', 1-weights, torch.log(torch.max(stabThr, U2 * KV2)))
                b = torch.exp(b)
                U1 = b / torch.max(stabThr, KV1)
                U2 = b / torch.max(stabThr, KV2)
            alpha1 = nan_recover(alpha1 * U1)
            beta1 = nan_recover(beta1 * V1)
            alpha2 = nan_recover(alpha2 * U2)
            beta2 = nan_recover(beta2 * V2)
        
        b = torch.clamp(b, 0, 255).double()
    
    if norm_type == 'max':
        output = b / (b.max(2, keepdim=True)[0].max(3, keepdim=True)[0] + stabThr) * total_max
    else:
        output = b / (b.sum(2, keepdim=True).sum(3, keepdim=True) + stabThr) * total_sum

#    if return_alpha:
#        if not proximal:
#            M = (torch.exp(-cost/reg) * cost).unsqueeze(0).unsqueeze(0)
#            dist1 = (torch.matmul(M, V1.view([batch_size, c, width**2, 1])).squeeze(-1) * U1.view([batch_size, c, width**2])).view([batch_size, -1]).sum(-1)
#            dist2 = (torch.matmul(M, V2.view([batch_size, c, width**2, 1])).squeeze(-1) * U2.view([batch_size, c, width**2])).view([batch_size, -1]).sum(-1)
#        else:
#            dist1 = (H1 * cost.unsqueeze(0).unsqueeze(0)).view([batch_size, -1]).sum(-1)
#            dist2 = (H2 * cost.unsqueeze(0).unsqueeze(0)).view([batch_size, -1]).sum(-1)
#    
#        ratio = torch.sqrt(dist2) / (torch.sqrt(dist1) + torch.sqrt(dist2))
#
    '''
    output2 = unsqueeze3(weights) * input1 + unsqueeze3(1.-weights) * input2
    mask = ((sum1 > 0).type(torch.FloatTensor) * (sum2 > 0).type(torch.FloatTensor)).to(device)
    output = mask * output1 + (1.-mask) * output2
    '''

    if v_max is not None:
    	output = torch.clamp(output, 0, v_max)    

    if mean is not None:
        output = (output - mean)/std

    return output, weights
#    if return_alpha:
#        return output, ratio
#    else:
#        return output, weights #.clone().detach().requires_grad_(True)



def proximal_barycenter(input1, input2, reg, weights=None, numItermax=1, 
                               mean=None, std=None,
                               stopThr=1e-9, stabThr=1e-15, verbose=False,
                               return_dist=False, norm_type='sum', p_iter=5):
    if weights is None:
        weights = torch.ones(input1.size()[0]) / 2
    else:
        weights = torch.ones(input1.size()[0]).cuda() * weights
    weights = weights.double()

    if mean is not None:
        mean = mean.unsqueeze(-1).unsqueeze(-1).cuda()
        std = std.unsqueeze(-1).unsqueeze(-1).cuda()
        input1 = std * input1 + mean + stabThr
        input2 = std * input2 + mean + stabThr

    input1 = input1.double()
    input2 = input2.double()

    sum1 = input1.sum(2, keepdim=True).sum(3, keepdim=True)
    sum2 = input2.sum(2, keepdim=True).sum(3, keepdim=True)
    total_sum = unsqueeze3(weights)*sum1 + unsqueeze3(1.-weights)*sum2

    if norm_type == 'max':
        max1 = input1.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        max2 = input2.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        total_max = unsqueeze3(weights)*max1 + unsqueeze3(1.-weights)*max2 
   
    stabThr = torch.tensor(stabThr).cuda().double()
    input1 /= sum1
    input2 /= sum2
    # build the convolution operator
    batch_size = input1.shape[0]
    c = input1.shape[1]
    width = input1.shape[2]
    
    t = torch.linspace(0, 1, width)
    [Y, X] = torch.meshgrid(t, t)
    xi1_init = torch.exp(-(X - Y)**2 / reg).cuda().double()
    xi1 = torch.ones_like(xi1_init).cuda().double()
    U1 = torch.ones_like(input1).cuda().double()
    U2 = torch.ones_like(input2).cuda().double()
    alpha1 = torch.ones_like(input1).cuda().double()
    beta1 = torch.ones_like(input1).cuda().double()
    alpha2 = torch.ones_like(input1).cuda().double()
    beta2 = torch.ones_like(input1).cuda().double()
    
    def K(x, xi1):
        return torch.matmul(torch.matmul(xi1, x), xi1)
    
    for t in range(p_iter):
        xi1 *= xi1_init
        for _ in range(numItermax):
            V1 = input1 / torch.max(stabThr, beta1 * K(U1 * alpha1, xi1))
            KV1 = alpha1 * K(V1 * beta1 , xi1)
            b = torch.einsum('i,ijkl->ijkl', weights, torch.log(torch.max(stabThr, U1 * KV1)))
            V2 = input2 / torch.max(stabThr, beta2 * K(U2 * alpha2, xi1))
            KV2 = alpha2 * K(V2 * beta2 , xi1)
            b += torch.einsum('i,ijkl->ijkl', 1-weights, torch.log(torch.max(stabThr, U2 * KV2)))
            b = torch.exp(b)
            U1 = b / torch.max(stabThr, KV1)
            U2 = b / torch.max(stabThr, KV2)
        alpha1 = nan_recover(alpha1 * U1)
        beta1 = nan_recover(beta1 * V1)
        alpha2 = nan_recover(alpha2 * U2)
        beta2 = nan_recover(beta2 * V2)
    if norm_type == 'max':
        b /= b.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        b *= total_max
    else:
        b /= b.sum(2, keepdim=True).sum(3, keepdim=True)
        b *= total_sum
    b = torch.clamp(b, 0, 255).float()

    if mean is not None:
        b = (b- mean)/std
    
    return b #.clone().detach().requires_grad_(True)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_process(out, target_reweighted, lam, p=1.0, in_batch=0, hidden=0,
                  emd=0, proximal=0, reg=1e-5, itermax=1, label_inter=0, mean=None, std=None,
                  box=0, graph=0, method='random', grad=None, block_num=-1, beta=0.0):
    if block_num == -1:
        block_num = 2**np.random.randint(0, 5)
    if in_batch:
        mix_idx = int(float(out.shape[0]) * p)
        indices = np.random.permutation(mix_idx)

        out_clean = out[mix_idx:]
        target_clean = target_reweighted[mix_idx:]
        out = out[:mix_idx].clone()
        target_reweighted = target_reweighted[:mix_idx]

        coin = 1
    else :
        indices = np.random.permutation(out.size(0))
        coin = np.random.binomial(1, p)
    
    if coin == 1: 
        if box:
            lam = lam.cpu().numpy()[0]
            out, ratio = mixup_box(out, out[indices], grad, grad[indices], method=method, alpha=lam)
        elif graph:
            lam = lam.cpu().numpy()[0]
            out, ratio = mixup_graph(out, out[indices], grad, grad[indices], block_num=block_num, method=method, alpha=lam, beta=beta)
        elif emd:
            if not hidden:
                out, ratio = barycenter_conv2d(out, out[indices], reg=reg, weights=lam, numItermax=itermax, mean=mean, std=std, v_max=1., return_alpha=label_inter, proximal=proximal)

            else:
                out_pos = torch.max(out, torch.zeros_like(out))
                out_neg = -torch.min(out, torch.zeros_like(out))

                out_pos, ratio=barycenter_conv2d(out_pos, out_pos[indices], reg=reg, weights=lam, numItermax=itermax, return_alpha=label_inter, proximal=proximal)
                out_neg, ratio=barycenter_conv2d(out_neg, out_neg[indices], reg=reg, weights=lam, numItermax=itermax, return_alpha=label_inter, proximal=proximal)
                out = out_pos - out_neg
        else:
            out = out*lam + out[indices]*(1-lam)
            ratio = torch.ones(out.shape[0]).cuda() * lam

        out = out.float()
        if in_batch:
            out_clean = out_clean.float()
        ratio = ratio.float()
        target_reweighted = target_reweighted.float()
        target_shuffled_onehot = target_reweighted[indices]
        target_reweighted = target_reweighted * ratio.unsqueeze(-1) + target_shuffled_onehot * (1 - ratio.unsqueeze(-1))
    
    if in_batch:
        out = torch.cat([out, out_clean], dim=0)
        target_reweighted = torch.cat([target_reweighted, target_clean], dim=0)
    return out, target_reweighted


#def mixup_process(out, target_reweighted, lam, reg=1e-5, wasserstein=False, proximal=True, itermax=5, p=1.0, hidden=False, C=None, label_inter=False, mean=None, std=None, in_batch=True, cutmix=False):
#    if in_batch:
#        mix_idx = int(float(out.shape[0]) * p)
#        indices = np.random.permutation(mix_idx)
#
#        out_clean = out[mix_idx:]
#        target_clean = target_reweighted[mix_idx:]
#        out = out[:mix_idx].clone()
#        target_reweighted = target_reweighted[:mix_idx]
#
#        coin = 1
#    else :
#        indices = np.random.permutation(out.size(0))
#        coin = np.random.binomial(1, p)
#   
#    if coin == 1:
#        if wasserstein:
#            if not hidden:
#                #C = cost_matrix(out.size()[-1])
#                idx = np.random.binomial(1, 1)
#                reg = [2e-3, 1e-5][idx]
#                p_iter = [5, 1][idx]
#                mean = [129.3/255, 124.1/255, 112.4/255]
#                std = [68.2/255, 65.4/255, 70.4/255]
#                out = proximal_barycenter(out.clone().cuda(), out[indices].clone().cuda(), reg=reg, weights=lam, mean=torch.tensor(mean), std=torch.tensor(std), p_iter=p_iter)
#                ratio = torch.ones(out.shape[0]).cuda() * lam
#            else:
#                out_pos = torch.max(out, torch.zeros_like(out))
#                out_neg = -torch.min(out, torch.zeros_like(out))
#                
#                out_pos, ratio=barycenter_conv2d(out_pos, out_pos[indices], reg=reg, weights=lam, numItermax=itermax, cost=C, return_alpha=label_inter, proximal=proximal)
#                out_neg, ratio=barycenter_conv2d(out_neg, out_neg[indices], reg=reg, weights=lam, numItermax=itermax, cost=C, return_alpha=label_inter, proximal=proximal)
#                out = out_pos - out_neg
#        else:
#            if cutmix:
#                bbx1, bby1, bbx2, bby2 = rand_bbox(out.size(), get_lambda(1))
#                out[:,:, bbx1:bbx2, bby1:bby2] = out[indices,:,bbx1:bbx2, bby1:bby2]
#                ratio = torch.ones(out.shape[0]).cuda() * (1 - ((bbx2 - bbx1) * (bby2 - bby1) / (out.size()[-1] * out.size()[-2])))
#            else:
#                out = out*lam + out[indices]*(1-lam)
#                ratio = torch.ones(out.shape[0]).cuda() * lam
#
#        target_shuffled_onehot = target_reweighted[indices]
#        target_reweighted = target_reweighted * ratio.unsqueeze(-1) + target_shuffled_onehot * (1 - ratio.unsqueeze(-1))
#    
#    if in_batch:
#        out = torch.cat([out, out_clean], dim=0)
#        target_reweighted = torch.cat([target_reweighted, target_clean], dim=0)
#    return out, target_reweighted
#

def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def apply(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).cuda()
        img = img * mask

        return img

'''code for cutmix'''
from ortools.graph.pywrapgraph import SimpleMaxFlow
from scipy.spatial import KDTree 

class GraphCutHelper(object):
    def __init__(self, block_num, beta=0):
        self.alpha = 1000 * block_num**2
        self.beta = int(beta * self.alpha)
        self.block_num = block_num
        self.num_nodes = self.block_num**2 + 2
        
        idx = 1
        self.pos_to_idx = {}
        self.idx_to_pos = {}
        positions = []
        for x in range(self.block_num):
            for y in range(self.block_num):
                self.pos_to_idx[(x, y)] = idx
                self.idx_to_pos[idx] = (x, y)
                positions.append([x, y])
                idx += 1
                
        positions = np.array(positions, dtype=np.int32)
        self.start_nodes = []
        self.end_nodes = []
        self.capacities = []
                
        # Calculate pairwise term 
        kdtree = KDTree(positions)
        for x in range(self.block_num):
            for y in range(self.block_num):
                idx = self.pos_to_idx[(x, y)]
                neighbors = kdtree.query_ball_point([x, y], r=1)
                for neighbor in neighbors:
                    x_n, y_n = kdtree.data[neighbor]
                    idx_n = self.pos_to_idx[(x_n, y_n)]
                    if idx_n <= idx:
                        continue
                    # Intermediate edges
                    self.start_nodes.append(idx)
                    self.end_nodes.append(idx_n)
                    self.capacities.append(self.beta)

                    self.start_nodes.append(idx_n)
                    self.end_nodes.append(idx)
                    self.capacities.append(self.beta)


    def pairwise_update(self, beta):
        self.beta = int(beta * self.alpha)
        for i in range(len(self.capacities)):
            self.capacities[i] = self.beta
    
                    
    def create_graphs(self, unary1, unary2):  
        # Construct KD-Tree and calculate unary term
        start_nodes = []
        end_nodes = []
        capacities = []
        for x in range(self.block_num):
            for y in range(self.block_num):            
                # From source
                start_nodes.append(0)
                end_nodes.append(self.pos_to_idx[(x, y)])
                capacities.append(int(self.alpha*unary1[x,y]))
                # To sink
                start_nodes.append(self.pos_to_idx[(x, y)])
                end_nodes.append(self.num_nodes-1)
                capacities.append(int(self.alpha*unary2[x,y]))
                                
        self.graph = [self.start_nodes+start_nodes, self.end_nodes+end_nodes, self.capacities+capacities]
    

    def solve(self):
        mask = np.zeros([self.block_num, self.block_num])
        start_nodes, end_nodes, capacities = self.graph

        max_flow = SimpleMaxFlow()
        for i in range(len(start_nodes)):
            max_flow.AddArcWithCapacity(start_nodes[i], end_nodes[i], capacities[i])
        
        if max_flow.Solve(0, self.num_nodes-1) == max_flow.OPTIMAL:
            source = max_flow.GetSourceSideMinCut()
            sink = max_flow.GetSinkSideMinCut()
            
            for idx in source[1:]:    
                x, y = self.idx_to_pos[idx]
                mask[x, y] = 1.
                
        return mask


def mixup_box(input1, input2, grad1, grad2, method='random', alpha=0.5):
    batch_size, _, height, width = input1.shape
    ratio = np.zeros([batch_size])
    
    if method == 'random':
        rx = np.random.uniform(0,height)
        ry = np.random.uniform(0,width)
        rh = np.sqrt(1 - alpha) * height
        rw = np.sqrt(1 - alpha) * width
        x1 = int(np.clip(rx - rh / 2, a_min=0., a_max=height))
        x2 = int(np.clip(rx + rh / 2, a_min=0., a_max=height))
        y1 = int(np.clip(ry - rw / 2, a_min=0., a_max=width))
        y2 = int(np.clip(ry + rw / 2, a_min=0., a_max=width))
        input1[:, :, x1:x2, y1:y2] = input2[:, :, x1:x2, y1:y2]
        ratio += 1 - (x2-x1)*(y2-y1)/(width*height)
        
    elif method == 'cut':
        rh = int(np.sqrt(1 - alpha) * height)
        rw = int(np.sqrt(1 - alpha) * width)
        if rh >0 and rw >0:
            grad_conv = torch.nn.functional.conv2d(grad2.unsqueeze(1), weight=grad2.new_ones(size=(1,1,rh,rw)))
            grad_max = grad_conv.view(batch_size,-1).max(1)[1]
            for idx in range(batch_size):
                x1 = grad_max[idx]//grad_conv.shape[-1]
                y1 = grad_max[idx]%grad_conv.shape[-1]
                x2 = x1 + rh
                y2 = y1 + rw
                input1[idx, :, x1:x2, y1:y2] = input2[idx, :, x1:x2, y1:y2]
                ratio[idx] = 1 - float((x2-x1)*(y2-y1))/(width*height)
        else:
            ratio += 1
    
    elif method == 'paste':
        rh = int(np.sqrt(1 - alpha) * height)
        rw = int(np.sqrt(1 - alpha) * width)

        if rh>0 and rw>0:
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
            ratio += 1

    else:
        raise AssertionError("wrong mixup method type !!")
    
    ratio = torch.tensor(ratio, dtype=torch.float32).cuda()
    return input1, ratio


def mixup_graph(input1, input2, grad1, grad2, block_num=2, method='random', alpha=0.5, beta=0.):
    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    ratio = np.zeros([batch_size])
    beta = beta / block_num / 16

    x, y = np.mgrid[0: block_num, 0: block_num] * block_size
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    
    mask = np.zeros([batch_size, 1, width, width])
    unary1 = F.avg_pool2d(grad1, block_size)
    unary2 = F.avg_pool2d(grad2, block_size)
    unary1 = unary1 / unary1.view(batch_size, -1).sum(1).view(batch_size, 1, 1)
    unary2 = unary2 / unary2.view(batch_size, -1).sum(1).view(batch_size, 1, 1)    
    
    if method == 'random':
        for (x,y) in points:
            # val = np.random.binomial(n=1, p=alpha, size=(1, 1, 1, 1))
            val = np.random.binomial(n=1, p=alpha, size=(batch_size, 1, 1, 1))            
            mask[:, :, x:x+block_size, y:y+block_size] += val 
            ratio += val[:,0,0,0]
        mask = torch.tensor(mask, dtype=torch.float32).cuda()

    elif method == 'cut':
        mask=[]
        unary2 = np.sqrt(1-alpha) * unary2.detach().cpu().numpy()
        base = np.ones_like(unary2) / block_num ** 2 * np.sqrt(alpha)

        graph = GraphCutHelper(block_num=block_num, beta=beta)
        for i in range(batch_size):
            graph.create_graphs(unary2[i], base[i])
            mask.append(1.0 - graph.solve())
            ratio[i] = mask[i].sum()
        
        mask = torch.tensor(mask, dtype=torch.float32).cuda()
        mask = F.interpolate(mask.unsqueeze(1), size=width)

    elif method == 'cut_small':
        mask=[]
        if alpha<= 0.5:
            unary1 = np.sqrt(alpha) * unary1.detach().cpu().numpy()
            base = np.ones_like(unary1) / block_num ** 2 * np.sqrt(1-alpha)
            
            graph = GraphCutHelper(block_num=block_num, beta=beta)
            for i in range(batch_size):
                graph.create_graphs(unary1[i], base[i])
                mask.append(graph.solve())
                ratio[i] = mask[i].sum()
        else:
            unary2 = np.sqrt(1-alpha) * unary2.detach().cpu().numpy()
            base = np.ones_like(unary2) / block_num ** 2 * np.sqrt(alpha)

            graph = GraphCutHelper(block_num=block_num, beta=beta)
            for i in range(batch_size):
                graph.create_graphs(unary2[i], base[i])
                mask.append(1.0 - graph.solve())
                ratio[i] = mask[i].sum()
        
        mask = torch.tensor(mask, dtype=torch.float32).cuda()
        mask = F.interpolate(mask.unsqueeze(1), size=width)

    elif method == 'mix':
        mask=[]
        unary1 = np.sqrt(alpha) * unary1.detach().cpu().numpy()
        unary2 = np.sqrt(1-alpha) * unary2.detach().cpu().numpy()
        
        graph = GraphCutHelper(block_num=block_num, beta=beta)
        for i in range(batch_size):
            graph.create_graphs(unary1[i], unary2[i])
            mask.append(graph.solve())
            ratio[i] = mask[i].sum()
            
        mask = torch.tensor(mask, dtype=torch.float32).cuda()
        mask = F.interpolate(mask.unsqueeze(1), size=width)

    else:
        raise AssertionError("wrong mixup method type !!")

    ratio = torch.tensor(ratio/block_num**2, dtype=torch.float32).cuda()
    return mask * input1 + (1-mask) * input2, ratio



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

def aug(image, preprocess):
    ws = np.float32(np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(3):
        image_aug = image.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations.augmentations)
            image_aug = op(image_aug, 3)

        mix += ws[i] * preprocess(image_aug)
        
    
    mixed = (1-m)*preprocess(image) + m*mix
    
    return mix

if __name__ == "__main__":
    create_val_folder('data/tiny-imagenet-200')  # Call method to create validation image folders



