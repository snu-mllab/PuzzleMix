# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import models.resnet as RN
import models.pyramidnet as PYRM
import utils
import numpy as np
from mixup import mixup_graph

from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type',
                    default='resnet',
                    type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=300,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b',
                    '--batch_size',
                    default=64,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.25,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq',
                    '-p',
                    default=1,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--depth', default=50, type=int, help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck',
                    dest='bottleneck',
                    action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset',
                    dest='dataset',
                    default='imagenet',
                    type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose',
                    dest='verbose',
                    action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha',
                    default=240,
                    type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='', type=str, help='name of experiment')
parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta')
parser.add_argument('--mixup_prob', default=1.0, type=float, help='mixup probability')
parser.add_argument('--padding', default=4, type=int, help='padding for transform_train')

parser.add_argument('--method',
                    default='puzzle',
                    type=str,
                    help='mixup type',
                    choices=('vanilla', 'cut', 'puzzle'))
# For Puzzle Mix
parser.add_argument('--mp', type=int, default=8, help='multi-process for graphcut (CPU)')
parser.add_argument('--n_labels', type=int, default=3, help='label space size')
parser.add_argument('--neigh_size',
                    type=int,
                    default=4,
                    help='neighbor size for computing distance beteeen image regions')

parser.add_argument('--p_beta', type=float, default=1.2, help='label smoothness')
parser.add_argument('--p_gamma', type=float, default=0.5, help='data local smoothness')
parser.add_argument('--p_eta', type=float, default=0.2, help='prior term')
parser.add_argument('--transport', type=str2bool, default=True, help='whether to use transport')
parser.add_argument('--t_eps', type=float, default=0.8, help='transport cost coefficient')

parser.add_argument('--clean_lam', type=float, default=0.0, help='clean regularization')
parser.add_argument('--seed', type=int, default=-1, help='random seed')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # Save path
    args.expname += args.method
    if args.transport:
        args.expname += '_tp'
    args.expname += '_prob_' + str(args.mixup_prob)
    if args.clean_lam > 0:
        args.expname += '_clean_' + str(args.clean_lam)
    if args.seed >= 0:
        args.expname += '_seed' + str(args.seed)
    print("Model is saved at {}".format(args.expname))

    # Dataset and loader
    if args.dataset.startswith('cifar'):
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        normalize = transforms.Normalize(mean=mean, std=std)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=args.padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR100('~/Datasets/cifar100/',
                                                                         train=True,
                                                                         download=True,
                                                                         transform=transform_train),
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=args.workers,
                                                       pin_memory=True)
            val_loader = torch.utils.data.DataLoader(datasets.CIFAR100('~/Datasets/cifar100/',
                                                                       train=False,
                                                                       transform=transform_test),
                                                     batch_size=args.batch_size // 4,
                                                     shuffle=True,
                                                     num_workers=args.workers,
                                                     pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data',
                                                                        train=True,
                                                                        download=True,
                                                                        transform=transform_train),
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=args.workers,
                                                       pin_memory=True)
            val_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data',
                                                                      train=False,
                                                                      transform=transform_test),
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.workers,
                                                     pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join('/data/readonly/ImageNet-Fast/imagenet/train')
        valdir = os.path.join('/data/readonly/ImageNet-Fast/imagenet/val')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))
        train_sampler = None

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=train_sampler)
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, val_transform),
                                                 batch_size=args.batch_size // 4,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
        numberofclass = 1000
        args.neigh_size = min(args.neigh_size, 2)

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    # Model
    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    pretrained = "runs/{}/{}".format(args.expname, 'checkpoint.pth.tar')
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained)
        checkpoint['state_dict'] = dict(
            (key[7:], value) for (key, value) in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch'] + 1
        best_err1 = checkpoint['best_err1']
        print("=> loaded checkpoint '{}'(epoch: {}, best err1: {}%)".format(
            pretrained, cur_epoch, checkpoint['best_err1']))
    else:
        cur_epoch = 0
        print("=> no checkpoint found at '{}'".format(pretrained))

    model = torch.nn.DataParallel(model).cuda()
    print('the number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    if os.path.isfile(pretrained):
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("optimizer is loaded!")

    mean_torch = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
    std_torch = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
    if args.mp > 0:
        mp = Pool(args.mp)
    else:
        mp = None

    # Start training and validation
    for epoch in range(cur_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, criterion_batch, optimizer, epoch,
                           mean_torch, std_torch, mp)
        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint(
            {
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_err1': best_err1,
                'best_err5': best_err5,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train(train_loader, model, criterion, criterion_batch, optimizer, epoch, mean, std, mp=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.mixup_prob and args.method == 'cut':
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

        elif args.beta > 0 and r < args.mixup_prob and args.method == 'puzzle':
            # calculate saliency map
            input_var = Variable(input, requires_grad=True)
            if args.clean_lam == 0:
                model.eval()
                output = model(input_var)
                loss_clean = criterion(output, target)
                loss_clean.backward(retain_graph=False)
                optimizer.zero_grad()
                model.train()
            else:
                # gradient regularization
                output = model(input_var)
                loss_clean = args.clean_lam * criterion(output, target)
                loss_clean.backward(retain_graph=True)

            unary = torch.sqrt(torch.mean(input_var.grad**2, dim=1))

            # perform mixup
            alpha = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            block_num = 2**np.random.randint(1, 5)
            with torch.no_grad():
                input, lam = mixup_graph(input,
                                         unary,
                                         rand_index,
                                         block_num=block_num,
                                         alpha=alpha,
                                         beta=args.p_beta,
                                         gamma=args.p_gamma,
                                         eta=args.p_eta,
                                         neigh_size=args.neigh_size,
                                         n_labels=args.n_labels,
                                         mean=mean,
                                         std=std,
                                         transport=args.transport,
                                         t_eps=args.t_eps,
                                         dataset=args.dataset,
                                         mp=mp)
            # calculate loss
            output = model(input)
            loss = lam * criterion_batch(output, target) + (1 - lam) * criterion_batch(
                output, target[rand_index])
            loss = torch.mean(loss)

        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(train_loader),
                                                                     LR=current_LR,
                                                                     batch_time=batch_time,
                                                                     data_time=data_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

    print(
        '* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'
        .format(epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


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


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(val_loader),
                                                                     batch_time=batch_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

    print(
        '* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'
        .format(epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1**(epoch // (args.epochs * 0.5))) * (0.1**(epoch //
                                                                      (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1**(epoch // 75))
        else:
            lr = args.lr * (0.1**(epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
