# main.py COPYRIGHT Fujitsu Limited 2021

from argparse import ArgumentParser
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import time
import os

import sys

try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from cac.relaxed_sync import RelaxedSyncDistributedDataParallel as DDP
from alexnet import AlexNet


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-r', '--relaxed_sync', action='store_true',
                        help='apply relaxed_sync')
    argparser.add_argument('-s', '--simulate', action='store_true',
                        help='simulate slow process')
    argparser.add_argument('-c', '--classes', type=int, default=10,
                        help='number of classes', required=False)
    argparser.add_argument('-b', '--batch', type=int, default=64,
                        help='mini batch size', required=False)
    argparser.add_argument('-w', '--workers', type=int, default=0,
                        help='number of workers', required=False)
    argparser.add_argument('-n', '--nodes', type=int, default=1, metavar='N',
                        help='number of data loading workers (default: 1)')
    argparser.add_argument('-e', '--epochs', type=int, default=20, metavar='N',
                        help='number of total epochs to run')
    argparser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help='learning rate')
    argparser.add_argument('--local_rank', type=int)
    argparser.add_argument('-S', '--single_gpu', action='store_true',
                        help='single gpu mode')
    return argparser.parse_args()


def train_and_validate(model, criterion, optimizer, train_loader, val_loader, args):

    # start training
    start_ep = time.time()
    for epoch in range(args.epochs):

        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        model.set_relaxed_pg(epoch, min_num_processes=1, master_skip=False)
        train_loader, val_loader = model.rearrange_data_loaders(train_loader, val_loader)
        args.lerarning_rate = model.adjust_lr_by_procs(args.learning_rate)

        count = 0

        model.train()
        for _, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = (outputs.max(1)[1] == labels).sum()
            train_acc += torch.div(acc, float(labels.size()[0]))

            train_loss += loss

            optimizer.zero_grad()

            # for APEX with relaxed sync
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            count += 1

        train_loss = model.calc_reduced_tensor(train_loss.data)
        train_acc = torch.div(train_acc, float(count))
        train_acc = model.calc_reduced_tensor(train_acc.data)

        model.eval()
        count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)
                acc = (outputs.max(1)[1] == labels).sum()
                val_acc += torch.div(acc, float(labels.size()[0]))

                val_loss += loss

                count += 1

        val_loss = model.calc_reduced_tensor(val_loss.data)
        val_acc = torch.div(val_acc, float(count))
        val_acc = model.calc_reduced_tensor(val_acc.data)

        end_ep = time.time()

        if dist.get_rank() == 0:
            print ('epoch [{}/{}], loss: {loss:.4f}, val_loss: {val_loss:.4f}, acc: {acc:.4f}, val_acc: {val_acc:.4f}, time: {time:.4f}'
                    .format(epoch+1, args.epochs, loss=train_loss, val_loss=val_loss, val_acc=val_acc,acc=train_acc, time=end_ep-start_ep))

    model.finalize()


def main():

    # options
    args = get_option()

    torch.manual_seed(args.local_rank)
    if args.single_gpu:
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1
    
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        print("(rank, world_size) = ", torch.distributed.get_rank(), args.world_size)
    else:
        print("assuming distributed execution. exiting..")
        return 


    # load CIFAR10 data

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data/',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data/',
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    image, label = train_dataset[0]

    model = AlexNet(args.classes)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    lr = args.learning_rate * args.world_size
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # for APEX with relaxed sync
    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level='O2',
        loss_scale='dynamic')

    # relaxed_sync_threshold=-1.0(off)/3.0(on)
    # relaxed_sync_mode_threshold: rate of removed prossess
    if args.relaxed_sync:
        relaxed_sync_threshold=3.0
    else:
        relaxed_sync_threshold=-1.0

    if args.simulate:
        simulate=1
    else:
        simulate=0

    model = DDP(
        model,
        delay_allreduce=True,
        relaxed_sync_threshold=relaxed_sync_threshold,
        relaxed_sync_mode_threshold=0.8,
        simulate_slow_process=simulate)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = model.TrainDataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = model.ValDataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler)

    train_and_validate(model, criterion, optimizer, train_loader, val_loader, args)


if __name__ == '__main__':
    main()
