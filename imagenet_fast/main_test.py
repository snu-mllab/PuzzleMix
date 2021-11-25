# This module is adapted from https://github.com/mahyarnajibi/FreeAdversarialTraining/blob/master/main_free.py
# Which in turn was adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import init_paths
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from utils import *
from validation import validate
#import torchvision.models as models
import models
from models.imagenet_resnet import BasicBlock, Bottleneck
#from torchvision.models.resnet import BasicBlock, Bottleneck

from apex import amp
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--output_prefix',
                        default='fast_adv',
                        type=str,
                        help='prefix used to define output path')
    parser.add_argument('-c',
                        '--config',
                        default='configs.yml',
                        type=str,
                        metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--restarts', default=1, type=int)
    return parser.parse_args()


# Parase config file and initiate logging
configs = parse_config_file(parse_args())
logger = initiate_logger(configs.output_name, configs.evaluate)
print = logger.info
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().cuda()
criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360, 0.866816, 0.826572,
    0.819324, 0.564592, 0.853204, 0.646056, 0.717840, 0.606500
]


def compute_mce(corruption_accs):
    """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
    mce = 0.
    for i in range(len(CORRUPTIONS)):
        avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
        ce = 100 * avg_err / ALEXNET_ERR[i]
        mce += ce / 15
    return mce


def main():
    # Scale and initialize the parameters
    best_prec1 = 0

    # Create output folder
    if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
        os.makedirs(os.path.join('trained_models', configs.output_name))

    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in configs.items():
        print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    # Create the model
    if configs.pretrained:
        print("=> using pre-trained model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch]()

    def init_dist_weights(model):
        for m in model.modules():
            if isinstance(m, BasicBlock):
                m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
            if isinstance(m, Bottleneck):
                m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    init_dist_weights(model)

    # Wrap the model into DataParallel
    model.cuda()

    # reverse mapping
    param_to_moduleName = {}
    for m in model.modules():
        for p in m.parameters(recurse=False):
            param_to_moduleName[p] = str(type(m).__name__)

    group_decay = [p for p in model.parameters() if 'BatchNorm' not in param_to_moduleName[p]]
    group_no_decay = [p for p in model.parameters() if 'BatchNorm' in param_to_moduleName[p]]
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=0)]
    optimizer = torch.optim.SGD(groups,
                                0,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

    if configs.TRAIN.half and not configs.evaluate:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1024)
    model = torch.nn.DataParallel(model)

    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(configs.resume,
                                                                checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    corruption_accs = test_c(model, test_transform)

    for c in CORRUPTIONS:
        print('\t'.join(map(str, [c] + corruption_accs[c])))

    print('mCE (normalized by AlexNet):', compute_mce(corruption_accs))


def train(train_loader, model, optimizer, epoch, lr_schedule, half=False):
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if configs.TRAIN.methods != 'augmix':
            input = input.cuda(non_blocking=True)
        else:
            input = torch.cat(input, 0).cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        # update learning rate
        lr = lr_schedule(epoch + (i + 1) / len(train_loader))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        input.sub_(mean).div_(std)
        lam = np.random.beta(configs.TRAIN.alpha, configs.TRAIN.alpha)
        if configs.TRAIN.methods == 'manifold' or configs.TRAIN.methods == 'graphcut':
            permuted_idx1 = np.random.permutation(input.size(0) // 4)
            permuted_idx2 = permuted_idx1 + input.size(0) // 4
            permuted_idx3 = permuted_idx2 + input.size(0) // 4
            permuted_idx4 = permuted_idx3 + input.size(0) // 4
            permuted_idx = np.concatenate(
                [permuted_idx1, permuted_idx2, permuted_idx3, permuted_idx4], axis=0)
        else:
            permuted_idx = torch.tensor(np.random.permutation(input.size(0)))

        if configs.TRAIN.methods == 'input':
            input = lam * input + (1 - lam) * input[permuted_idx]

        elif configs.TRAIN.methods == 'cutmix':
            input, lam = mixup_box(input, lam=lam, permuted_idx=permuted_idx)

        elif configs.TRAIN.methods == 'augmix':
            logit = model(input)
            logit_clean, logit_aug1, logit_aug2 = torch.split(logit, logit.size(0) // 3)
            output = logit_clean

            p_clean = F.softmax(logit_clean, dim=1)
            p_aug1 = F.softmax(logit_aug1, dim=1)
            p_aug2 = F.softmax(logit_aug2, dim=1)

            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss_JSD = 4 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean'))

        elif configs.TRAIN.methods == 'graphcut':
            input_var = Variable(input, requires_grad=True)

            output = model(input_var)
            loss_clean = criterion(output, target)

            if half:
                with amp.scale_loss(loss_clean, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_clean.backward()
            unary = torch.sqrt(torch.mean(input_var.grad**2, dim=1))

            block_num = 2**(np.random.randint(1, 5))
            mask = get_mask(input, unary, block_num, permuted_idx, alpha=lam, mean=mean, std=std)
            output, lam = model(input,
                                graphcut=True,
                                permuted_idx=permuted_idx1,
                                block_num=block_num,
                                mask=mask,
                                unary=unary)

        if configs.TRAIN.methods == 'manifold':
            output = model(input, manifold=True, lam=lam, permuted_idx=permuted_idx1)
        elif configs.TRAIN.methods != 'augmix' and configs.TRAIN.methods != 'graphcut':
            output = model(input)

        if configs.TRAIN.methods == 'nat':
            loss = criterion(output, target)
        elif configs.TRAIN.methods == 'augmix':
            loss = criterion(output, target) + loss_JSD
        else:
            loss = lam * criterion_batch(output, target) + (1 - lam) * criterion_batch(
                output, target[permuted_idx])
            loss = torch.mean(loss)

        # compute gradient and do SGD step
        #optimizer.zero_grad()
        if half:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % configs.TRAIN.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR {lr:.3f}'.format(epoch,
                                       i,
                                       len(train_loader),
                                       batch_time=batch_time,
                                       data_time=data_time,
                                       top1=top1,
                                       top5=top5,
                                       cls_loss=losses,
                                       lr=lr))
            sys.stdout.flush()


def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0

    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            images.sub_(mean).div_(std)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)


def test_c(net, test_transform):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = {}
    for c in CORRUPTIONS:
        print(c)
        for s in range(1, 6):
            valdir = os.path.join('/home/wonhochoo/data/imagenet/imagenet-c/', c, str(s))
            val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, test_transform),
                                                     batch_size=configs.DATA.batch_size,
                                                     shuffle=False,
                                                     num_workers=configs.DATA.workers,
                                                     pin_memory=True,
                                                     drop_last=False)

            loss, acc1 = test(net, val_loader)
            if c in corruption_accs:
                corruption_accs[c].append(acc1)
            else:
                corruption_accs[c] = [acc1]

            print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(s, loss, 100. * acc1))

    return corruption_accs


if __name__ == '__main__':
    main()
