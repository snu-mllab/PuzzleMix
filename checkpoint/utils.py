import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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

def to_one_hot(inp,num_classes,device='cuda'):
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    
    return y_onehot


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
    C = torch.tensor(C).cuda()
    return C


cost_matrix_dict = {'2':cost_matrix(2).unsqueeze(0), '4':cost_matrix(4).unsqueeze(0), '8':cost_matrix(8).unsqueeze(0), '16':cost_matrix(16).unsqueeze(0)}
      
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

def barycenter_conv2d(input1, input2, reg=2e-3, weights=None, numItermax=5, proximal=True,
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
        weights = torch.ones(input1.shape[0], device=device) / 2
    elif weights.shape[0] == 1:
        weights = torch.ones(input1.shape[0], device=device) * weights[0]
    else:
        pass
    
    if mean is not None and std is not None:
        mean = mean.to(device).double()
        std = std.to(device).double()

    weights = weights.to(device).double()
    stabThr = torch.tensor(stabThr, device=device).double()
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
        t = torch.linspace(0, 1, input1.shape[2]).to(device)
        [Y, X] = torch.meshgrid(t, t)
        xi1 = torch.exp(-(X - Y)**2 / reg).double()

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
    
        for t in range(numItermax): #p_iter=1
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

    '''
    output2 = unsqueeze3(weights) * input1 + unsqueeze3(1.-weights) * input2
    mask = ((sum1 > 0).type(torch.FloatTensor) * (sum2 > 0).type(torch.FloatTensor)).to(device)
    output = mask * output1 + (1.-mask) * output2
    '''

    if v_max is not None:
    	output = torch.clamp(output, 0, v_max)    

    if mean is not None:
        output = (output - mean)/std

    return output.float(), weights.float()



def mixup_process(out, target_reweighted, mixup_alpha=1.0, loss_batch=None, p=1.0, in_batch=0, hidden=0,
                  emd=0, proximal=0, reg=1e-5, itermax=1, label_inter=0, mean=None, std=None,
                  box=0, graph=0, method='random', grad=None, block_num=-1, beta=0.0, gamma=0., eta=0.2, neigh_size=2, n_labels=2, label_cost='l2', sigma=1.0, warp=0.0, dim=2, beta_c=0.0,
                  transport=False, t_eps=10.0, t_type='full', t_size=16, adv_p=0.0, adv_type='out', input_grad=None):
    if block_num == -1:
        block_num = 2**np.random.randint(1, 5)
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

    if loss_batch is None:
        lam = get_lambda(mixup_alpha)
        lam = np.array([lam]).astype('float32')
    else :
        # this is only for graph
        lam = []
        lam_anchor = get_lambda(mixup_alpha)
        for i in range(len(indices)):
            alpha1 = 2 * loss_batch[i] / (loss_batch[i] + loss_batch[indices][i])
            alpha2 = 2 * loss_batch[indices][i] / (loss_batch[i] + loss_batch[indices][i])
            
            lam.append(np.clip(lam_anchor + (alpha1-1), 0, 1))
            # lam.append(get_lambda(mixup_alpha * alpha1, mixup_alpha * alpha2))
        lam = np.array(lam).astype('float32')
    
    if coin == 1: 
        if box:
            out, ratio = mixup_box(out, out[indices], None, None, method=method, alpha=lam[0])
        elif graph:
            if block_num > 1:
                out, ratio = mixup_graph(out, grad, indices, block_num=block_num, method=method,
                             alpha=lam, beta=beta, gamma=gamma, eta=eta, neigh_size=neigh_size, n_labels=n_labels, label_cost=label_cost, sigma=sigma, warp=warp, dim=dim, beta_c=beta_c,
                             mean=mean, std=std, emd=emd, reg=reg, itermax=itermax, transport=transport, t_eps=t_eps, t_type=t_type, t_size=t_size, adv_p=adv_p, adv_type=adv_type, input_grad=input_grad)
            else: 
                ratio = torch.ones(out.shape[0], device='cuda')
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
            out = out*lam[0] + out[indices]*(1-lam[0])
            ratio = torch.ones(out.shape[0], device='cuda') * lam[0]

        target_shuffled_onehot = target_reweighted[indices]
        target_reweighted = target_reweighted * ratio.unsqueeze(-1) + target_shuffled_onehot * (1 - ratio.unsqueeze(-1))
    
    if in_batch:
        out = torch.cat([out, out_clean], dim=0)
        target_reweighted = torch.cat([target_reweighted, target_clean], dim=0)
    return out, target_reweighted


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


def get_lambda(alpha=1.0, alpha2=None):
    '''Return lambda'''
    if alpha > 0.:
        if alpha2 is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = np.random.beta(alpha + 1e-2, alpha2 + 1e-2)
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

'''code for graphmix'''
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
                pairwise_cost[i, j] = abs(i-j) / (n_labels-1)
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


from scipy.ndimage.filters import gaussian_filter
random_state = np.random.RandomState(None)

def mixup_graph(input1, grad1, indices, block_num=2, method='random', alpha=0.5, beta=0., gamma=0., eta=0.2, neigh_size=2, n_labels=2, label_cost='l2', sigma=1.0, warp=0.0, dim=2, beta_c=0.0, mean=None, std=None, emd=False, reg=1e-5, itermax=5, transport=False, t_eps=10.0, t_type='full', t_size=16, adv_p=0.0, adv_type='out', input_grad=None):
    if adv_p > 0 and adv_type=='in':
        # adv_mask = np.random.binomial(n=1, p=adv_p, size=(batch_size, 1, 1, 1))
        adv_mask1 = np.random.binomial(n=1, p=adv_p)
        
        if adv_mask1 == 1.:
            input1 = input1 * std + mean + input_grad
            input1 = torch.clamp(input1, 0, 1)
            input1 = (input1 - mean)/std
    
    input2 = input1[indices].clone()
            
    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    neigh_size = min(neigh_size, block_size)
    t_size = min(t_size, block_size)

    if alpha.shape[0] == 1:
        alpha = np.ones([batch_size]) * alpha[0]

    ratio = np.zeros([batch_size])
    beta = beta/block_num/16
    if dim==3:
        beta_c = beta_c/block_num/16/3
        beta /= 3
    
    grad1_pool = F.avg_pool2d(grad1, block_size)
    if dim==2:
        mask = np.zeros([batch_size, 1, width, width])
        unary1_torch = grad1_pool / grad1_pool.view(batch_size, -1).sum(1).view(batch_size, 1, 1)
    elif dim==3:
        mask = np.zeros([batch_size, 3, width, width])
        unary1_torch = grad1_pool / grad1_pool.view(batch_size, -1).sum(1).view(batch_size, 1, 1, 1)
    
    if warp > 0:
        row = torch.linspace(-1, 1, block_num).cuda()
        col = torch.linspace(-1, 1, block_num).cuda()
        x, y = torch.meshgrid([row, col])
        grid = torch.stack([y, x], dim=-1).unsqueeze(0)

        dx = warp * torch.tensor(gaussian_filter((random_state.rand(batch_size, block_num, block_num) * 2 - 1), sigma, mode="constant", cval=0), dtype=torch.float32, device='cuda') * 2 / block_num
        dy = warp * torch.tensor(gaussian_filter((random_state.rand(batch_size, block_num, block_num) * 2 - 1), sigma, mode="constant", cval=0), dtype=torch.float32, device='cuda') * 2 / block_num
        grid_offset = torch.stack([dy, dx], dim=-1)


    if method == 'random':
        x, y = np.mgrid[0: block_num, 0: block_num] * block_size
        points = np.stack([x.ravel(), y.ravel()], axis=1)

        for (x,y) in points:
            val = np.random.binomial(n=1, p=alpha[0], size=(batch_size, 1, 1, 1))
            mask[:, :, x:x+block_size, y:y+block_size] += val
            ratio += val[:,0,0,0]
            mask = torch.tensor(mask, dtype=torch.float32, device='cuda')            

    elif method == 'mix':
        mask=[]
        if warp>0:
            unary1_torch = F.grid_sample(unary1_torch.unsqueeze(1), grid+grid_offset, padding_mode="reflection").squeeze(1)
        unary2_torch = unary1_torch[indices]
        
        input1_pool = F.avg_pool2d(input1 * std + mean, neigh_size)
        if warp>0:
            input1_pool = F.grid_sample(input1_pool, grid+grid_offset, padding_mode="reflection")
        input2_pool = input1_pool[indices]

        if dim==2:
            pw_x = torch.zeros([batch_size, 2, 2, block_num-1, block_num], device='cuda')
            pw_y = torch.zeros([batch_size, 2, 2, block_num, block_num-1], device='cuda')
        elif dim==3:
            pw_x = torch.zeros([batch_size, 2, 2, 3, block_num-1, block_num], device='cuda')
            pw_y = torch.zeros([batch_size, 2, 2, 3, block_num, block_num-1], device='cuda')

        k = block_size//neigh_size

        # pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(input1_pool, input1_pool, k, dim=dim)
        # pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(input1_pool, input2_pool, k, dim=dim)
        # pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(input2_pool, input1_pool, k, dim=dim)
        # pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(input2_pool, input2_pool, k, dim=dim)

        pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(input2_pool, input2_pool, k, dim=dim)
        pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(input2_pool, input1_pool, k, dim=dim)
        pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(input1_pool, input2_pool, k, dim=dim)
        pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(input1_pool, input1_pool, k, dim=dim)

        pw_x = beta * gamma * pw_x
        pw_y = beta * gamma * pw_y
        
        unary1 = unary1_torch.clone()
        unary2 = unary2_torch.clone()
        if dim==2:
            unary2[:,:-1,:] += (pw_x[:,1,0] + pw_x[:,1,1])/2.
            unary1[:,:-1,:] += (pw_x[:,0,1] + pw_x[:,0,0])/2.
            unary2[:,1:,:] += (pw_x[:,0,1] + pw_x[:,1,1])/2.
            unary1[:,1:,:] += (pw_x[:,1,0] + pw_x[:,0,0])/2.

            unary2[:,:,:-1] += (pw_y[:,1,0] + pw_y[:,1,1])/2.
            unary1[:,:,:-1] += (pw_y[:,0,1] + pw_y[:,0,0])/2.
            unary2[:,:,1:] += (pw_y[:,0,1] + pw_y[:,1,1])/2.
            unary1[:,:,1:] += (pw_y[:,1,0] + pw_y[:,0,0])/2.
        elif dim==3:
            unary2[:,:,:-1,:] += (pw_x[:,1,0] + pw_x[:,1,1])/2.
            unary1[:,:,:-1,:] += (pw_x[:,0,1] + pw_x[:,0,0])/2.
            unary2[:,:,1:,:] += (pw_x[:,0,1] + pw_x[:,1,1])/2.
            unary1[:,:,1:,:] += (pw_x[:,1,0] + pw_x[:,0,0])/2.

            unary2[:,:,:,:-1] += (pw_y[:,1,0] + pw_y[:,1,1])/2.
            unary1[:,:,:,:-1] += (pw_y[:,0,1] + pw_y[:,0,0])/2.
            unary2[:,:,:,1:] += (pw_y[:,0,1] + pw_y[:,1,1])/2.
            unary1[:,:,:,1:] += (pw_y[:,1,0] + pw_y[:,0,0])/2.

        pw_x = (pw_x[:,1,0] + pw_x[:,0,1] - pw_x[:,1,1] - pw_x[:,0,0])/2
        pw_y = (pw_y[:,1,0] + pw_y[:,0,1] - pw_y[:,1,1] - pw_y[:,0,0])/2

        unary1 = unary1.detach().cpu().numpy()
        unary2 = unary2.detach().cpu().numpy()
        pw_x = pw_x.detach().cpu().numpy()
        pw_y = pw_y.detach().cpu().numpy()

        for i in range(batch_size):
            if dim==2:
                mask.append(graphcut_multi(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha[i], beta, eta, n_labels))
            elif dim==3:
                mask.append(graphcut_multi_float(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha[i],  beta, eta, n_labels, dim=dim, beta_c=beta_c))

            ratio[i] = mask[i].sum()

        mask = torch.tensor(mask, dtype=torch.float32, device='cuda')
        if dim==2:
            mask = mask.unsqueeze(1)
        if warp>0:
            mask = F.grid_sample(mask, grid-grid_offset, padding_mode="reflection")

        if adv_p > 0 and adv_type=='out':
            # adv_mask = np.random.binomial(n=1, p=adv_p, size=(batch_size, 1, 1, 1))
            adv_mask1 = np.random.binomial(n=1, p=adv_p)
            adv_mask2 = np.random.binomial(n=1, p=adv_p)

            if adv_mask1 == 1.:
                input1 = input1 * std + mean + input_grad
                input1 = torch.clamp(input1, 0, 1)
                input1 = (input1 - mean)/std
            
            if adv_mask2 == 1.:
                input2 = input2 * std + mean + input_grad[indices]
                input2 = torch.clamp(input2, 0, 1)
                input2 = (input2 - mean)/std

        if transport:
            if t_size < block_size:
                t_block_num = width // t_size
                mask = F.interpolate(mask, size=t_block_num)
                grad1_pool = F.avg_pool2d(grad1, t_size)
                unary1_torch = grad1_pool / grad1_pool.view(batch_size, -1).sum(1).view(batch_size, 1, 1)
                unary2_torch = unary1_torch[indices]
            else:
                t_block_num = block_num
            
	    # input1
            plan = mask_transport(mask, unary1_torch, eps=t_eps, t_type=t_type)
            input1 = transport_image(input1, plan, batch_size, t_block_num, t_size)

            # input2
            plan = mask_transport(1-mask, unary2_torch, eps=t_eps, t_type=t_type)
            input2 = transport_image(input2, plan, batch_size, t_block_num, t_size)

        mask = F.interpolate(mask, size=width)

    else:
        raise AssertionError("wrong mixup method type !!")

    ratio = torch.tensor(ratio/block_num**2, dtype=torch.float32, device='cuda')
    if dim ==3: 
        ratio /= 3.
         
    return mask * input1 + (1-mask) * input2, ratio


def mask_transport(mask, grad_pool, eps=0.01, t_type='full'):
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]

    # n_iter = int(np.sqrt(block_num))  
    n_iter = int(block_num)
    C = cost_matrix_dict[str(block_num)]
    
    if t_type=='full':
        z = (mask>0).float()
    else:
        z = mask
    
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)
    
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
    
    return plan_win
    # plan_move = (plan_win - torch.eye(block_num**2, device='cuda'))>0
    # return plan_move


def transport_image(img, plan, batch_size, block_num, block_size):
    input_patch = img.reshape([batch_size, 3, block_num, block_size, block_num * block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, 3, block_num, block_num, block_size, block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, 3, block_num**2, block_size, block_size]).permute(0,1,3,4,2).unsqueeze(-1)

    input_transport = plan.transpose(-2,-1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(0,1,4,2,3)
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


def aug(image, preprocess):
    pass

if __name__ == "__main__":
    create_val_folder('data/tiny-imagenet-200')  # Call method to create validation image folders



