#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import print_function
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100,CIFAR10
import torchvision.models as models
import moco.loader
import moco.builder_single_labeled_k_pos
from utils import *
from dataset.imagenet_custom import *
from dataset.cifar_custom import ImbalanceCIFAR10,ImbalanceCIFAR100
from tqdm import tqdm
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
try:
    import wandb
    use_wandb = True
except:
    use_wandb = False
    
def log(key, value):
    if use_wandb:
        wandb.log({key:value})
    

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('--data', default='', type=str, metavar='PATH',
                    help='path to training data')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# knn monitor
parser.add_argument('--knn-k', default=5, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--augment-weight', default=0.5, type=float,
                    help='augment weight value for loss decomposition')
parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')
parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

parser.add_argument('--dataset', default="imagenet", type=str, help='dataset type')
parser.add_argument('--imb-factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--use-center', action='store_true',
                    help='use prototype loss')
parser.add_argument('--use-class-temperature', action='store_true',
                    help='use adaptive temperature')
# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--results-dir', default='result', type=str, metavar='PATH', help='path to cache (default: none)')                   
args = parser.parse_args() 
args.cos = True
args.schedule = []  # cos in use
args.symmetric = False
args.mlp=True
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
use_class_temperature=False#args.use_class_temperature
use_center=args.use_center
print("check using center")
print(use_center)

def main():
    #args = parser.parse_args()
    
    if use_wandb:
        wandb.init(project="Imbalanced")
        wandb.config.update(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model=moco.builder_single_labeled_k_pos.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t,args.symmetric, args.mlp).cuda()
    print(model.encoder_q)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        #raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        pass
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        #raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterition_center = SupConLoss(0.1)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
    if(args.dataset=="cifar100"):
        train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        if(args.imb_factor==0.1):
            weight_augment=0.25
        else:
            weight_augment=0.5
        weight_imb=1.0
        weight_center=1.0
        weight_sum_ori=0.1
        from dataset.glister_cifar_100 import train_coreset_loader as coreset_loader
        memory_data = ImbalanceCIFAR100(root='data',imb_factor=args.imb_factor,data_type="test", train=True,
                                                    download=True, transform=test_transform)
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = CIFAR100(root='data', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        train_set = ImbalanceCIFAR100(root='data',imb_factor=args.imb_factor,data_type="train", train=True,
                                                    download=True, transform=train_transform)
        #train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    elif(args.dataset=="cifar10"):
        train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        weight_augment=0.25
        weight_imb=1.0
        weight_center=0.1
        weight_sum_ori=0.06
        from dataset.glister_cifar_10 import train_coreset_loader  as coreset_loader
        memory_data = ImbalanceCIFAR10(root='data',imb_factor=args.imb_factor,data_type="test", train=True,
                                                    download=True, transform=test_transform)
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        train_set = ImbalanceCIFAR10(root='data',imb_factor=args.imb_factor,data_type="train", train=True,
                                                    download=True, transform=train_transform)

        #train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    elif(args.dataset=="imagenet"):#ImageNet
        weight_augment=args.augment_weight
        weight_imb=1.0
        weight_center=1.0
        weight_sum_ori=0.1
        # Data loading code
        root = args.data
        txt_train = 'ImageNet_LT_train.txt'
        txt_coreset='coreset_imagenet.txt'
        txt_test='ImageNet_LT_test.txt'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        augmentation_train =transforms.Compose( [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        augmentation_test = transforms.Compose([
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])])

        train_set=ImageNetLT(root, txt_train,type="train" ,transform=augmentation_train)
        coreset=ImageNetLT(root, txt_coreset,type="coreset" ,transform=augmentation_train)
        memory_set=ImageNetLT(root, txt_train,type="test" ,transform=augmentation_test)
        test_set=ImageNetLT(root,txt_test,type="test", transform=augmentation_test)

        train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

        coreset_loader = torch.utils.data.DataLoader(
        coreset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

        memory_loader = torch.utils.data.DataLoader(
        memory_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    import logging
    logging.basicConfig(filename="log.txt", level=logging.DEBUG)
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    list_temperature_tensor_global=generate_temperature_comb(dict(train_loader.dataset.dict_index_class))
    best_acc=0
    test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss=train(model,train_loader,coreset_loader, criterition_center, optimizer, epoch, args,list_temperature_tensor_global,weight_augment,weight_imb,weight_center,weight_sum_ori)
        results['train_loss'].append(train_loss)
        if(epoch % 5==0):
            test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
            log("test accuracy", test_acc_1)
            logging.debug(test_acc_1)
            results['test_acc@1'].append(test_acc_1)
            if(test_acc_1>best_acc):
                best_acc=test_acc_1
                print("save_best model with acc "+str(test_acc_1))
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, f'model_best_{args.augment_weight}.pth')
        if(epoch%50==0):
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),},f'model_epoch_{epoch}_{args.augment_weight}.pth')

def train(net, data_loader,train_coreset, center_criterion,train_optimizer,epoch, args,list_temperature_tensor,w_augment,w_imb,w_center,weight_sum):
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    if(use_center):
        list_v1,list_v2,list_targets=[],[],[]
        for images,targets in train_coreset:
            image_root=torch.cat([images[0], images[1]], dim=0)
            image_root=image_root.cuda(non_blocking=True)
            bsz = targets.shape[0]
            with torch.no_grad():
                features = net.encoder_q(image_root)
            features=nn.functional.normalize(features, dim=1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            list_v1.append(f1.cpu().detach().clone())
            list_v2.append(f2.cpu().detach().clone())
            list_targets.append(targets.cpu().detach().clone())
        v1_tensor=torch.cat(list_v1,axis=0)
        v2_tensor=torch.cat(list_v2,axis=0)
        targets_tensor=torch.cat(list_targets,axis=0)
        center_tensor_v1,center_tensor_v2=calculate_tensor_center(v1_tensor,v2_tensor,targets_tensor)
        center_tensor_v1=center_tensor_v1.cuda()
        center_tensor_v2=center_tensor_v2.cuda()

    for images,pos,targets in train_bar:
        im_1, im_2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
        im_1_feature = net.encoder_q(im_1)  # queries: NxC
        im_1_feature = nn.functional.normalize(im_1_feature, dim=1)  # already normalized
        pos_1,pos_2,pos_3=pos[0].cuda(non_blocking=True)  ,pos[1].cuda(non_blocking=True),pos[2].cuda(non_blocking=True)
        #augment loss
        loss_augment,_,_ = net(im_1_feature, [im_2],targets,epoch)
        #positive loss
        loss_pos,_,_=net(im_1_feature,[pos_1,pos_2,pos_3],targets,epoch)
        #ImbMoco loss
        loss_pos_total=0.75*(loss_pos) + 0.25*(loss_augment)
        if(use_center):
            #calc prototype loss
            #=torch.cat([images[0], images[1]], dim=0)
            #image_root=image_root.cuda(non_blocking=True)
            targets=torch.reshape(targets,(targets.shape[0],))
            targets=targets.type(torch.LongTensor)
            bsz = targets.shape[0]
            #got temperature for each class
            class_temperature=list_temperature_tensor[targets]
            class_list=torch.unique(targets)
            class_list=class_list.type(torch.LongTensor)
            batch_center_v1=center_tensor_v1[targets]
            batch_center_v2=center_tensor_v2[targets]
            #extract feature
            im_2_feature = net.encoder_q(im_2)
            im_2_feature = nn.functional.normalize(im_2_feature, dim=1)
            #features=nn.functional.normalize(features, dim=1)
            f1, f2 = im_1_feature,im_2_feature
            for i in range(center_tensor_v1.shape[0]):
                if i not in class_list:
                    f1=torch.cat((f1,center_tensor_v1[i].reshape(1,128)),axis=0)
                    f2=torch.cat((f2,center_tensor_v2[i].reshape(1,128)),axis=0)
                    #print(batch_center_v1.shape)
                    #print(center_tensor_v1[i].unsqueeze(1).shape)
                    batch_center_v1=torch.cat((batch_center_v1,center_tensor_v1[i].reshape(1,128)),axis=0)
                    batch_center_v2=torch.cat((batch_center_v2,center_tensor_v2[i].reshape(1,128)),axis=0)
            class_temperature=torch.cat([class_temperature,class_temperature],dim=0).cuda()
    
            features_f1_1=torch.cat([f1.unsqueeze(1), batch_center_v1.unsqueeze(1)], dim=1)
            features_f1_2=torch.cat([f1.unsqueeze(1), batch_center_v2.unsqueeze(1)], dim=1)
            features_f2_1 = torch.cat([f2.unsqueeze(1), batch_center_v1.unsqueeze(1)], dim=1)
            features_f2_2 = torch.cat([f2.unsqueeze(1), batch_center_v2.unsqueeze(1)], dim=1)
            #calc symmetric prototype loss
            loss_v1_1=center_criterion(features_f1_1,class_temperature,use_class_temperature)
            loss_v2_2=center_criterion(features_f2_2,class_temperature,use_class_temperature)
            loss_v1_2=center_criterion(features_f1_2,class_temperature,use_class_temperature)
            loss_v2_1=center_criterion(features_f2_1,class_temperature,use_class_temperature)
            loss_center= 0.06*(loss_v1_1 + loss_v2_2 + loss_v1_2 + loss_v2_1 )
            #loss_v1_1=center_criterion(features_f1_1,class_temperature,use_class_temperature)
            loss = (loss_pos_total + loss_center)/2
        else:
            loss=loss_pos_total
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num))
        
        # log loss of a mini batch
        log("loss_batch", loss.item())
    
    # log for 1 epoch
    log("lr", train_optimizer.param_groups[0]['lr'])
    log("loss", total_loss / total_num)
    return total_loss / total_num



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
