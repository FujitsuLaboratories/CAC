# Copyright (c) 2021,      Fujitsu Limited
# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import dllogger
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

import utils.resnet_model as resnet_models
import utils.alexnet_model as alexnet_models
import utils.logger as log

from utils.smoothing import LabelSmoothing
from utils.mixup import NLLMultiLabelSmooth, MixUpWrapper
from utils.dataloaders import *
from utils.utils import *

import cac

ACC_METADATA = {'unit': '%','format': ':.2f'}
IPS_METADATA = {'unit': 'img/s', 'format': ':.2f'}
TIME_METADATA = {'unit': 's', 'format': ':.5f'}
LOSS_METADATA = {'format': ':.5f'}


class ModelAndLoss(nn.Module):
    def __init__(self,
                 arch,
                 loss,
                 pretrained_weights=None,
                 cuda=True,
                 fp16=False):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print("=> creating model '{}'".format(arch))
        if arch[0] == 'alexnet':
            model = alexnet_models.build_alexnet()
        else:
            model = resnet_models.build_resnet(arch[0], arch[1])
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.cuda()
        if fp16:
            model = network_to_half(model)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def distributed(self):
        self.model = DDP(self.model)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


def get_optimizer(parameters,
                  fp16,
                  lr,
                  momentum,
                  weight_decay,
                  nesterov=False,
                  state=None,
                  static_loss_scale=1.,
                  dynamic_loss_scale=False):

    print(" ! Weight decay applied to BN parameters ")
    optimizer = cac.gradskip.SGD([v for n, v in parameters],
                                 lr,
                                 momentum=momentum,
                                 weight_decay=weight_decay,
                                 nesterov=nesterov)
    if fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=static_loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale,
                                   verbose=False)

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric('lr',
                               log.LR_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE)

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric('lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(base_lr,
                          warmup_length,
                          epochs,
                          final_multiplier=0.001,
                          logger=None):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier) / es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay**e)
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(model_and_loss,
                   optimizer,
                   fp16,
                   use_amp=False,
                   batch_size_multiplier=1):
    def _step(input, target, optimizer_step=True):
        input_var = Variable(input)
        target_var = Variable(target)
        loss, output = model_and_loss(input_var, target_var)
        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        # for k, m in enumerate(model_and_loss.modules()):
        #     print('layer index: {:d}'.format(k))
        # print(model_and_loss.modules())

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if optimizer_step:
            opt = optimizer.optimizer if isinstance(
                optimizer, FP16_Optimizer) else optimizer
            for param_group in opt.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.grad /= batch_size_multiplier

            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(train_loader,
          model_and_loss,
          optimizer,
          lr_scheduler,
          fp16,
          logger,
          epoch,
          use_amp=False,
          batch_size_multiplier=1,
          register_metrics=True):

    if register_metrics and logger is not None:
        logger.register_metric('train.loss',
                               log.LOSS_METER(),
                               verbosity=dllogger.Verbosity.DEFAULT,
                               metadata=LOSS_METADATA)
        logger.register_metric('train.compute_ips',
                               log.PERF_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=IPS_METADATA)
        logger.register_metric('train.total_ips',
                               log.PERF_METER(),
                               verbosity=dllogger.Verbosity.DEFAULT,
                               metadata=IPS_METADATA)
        logger.register_metric('train.data_time',
                               log.PERF_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=TIME_METADATA)
        logger.register_metric('train.compute_time',
                               log.PERF_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=TIME_METADATA)

    step = get_train_step(model_and_loss,
                          optimizer,
                          fp16,
                          use_amp=use_amp,
                          batch_size_multiplier=batch_size_multiplier)

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss = step(input, target, optimizer_step=optimizer_step)

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric('train.loss', to_python_float(loss), bs)
            logger.log_metric('train.compute_ips',
                              calc_ips(bs, it_time - data_time))
            logger.log_metric('train.total_ips', calc_ips(bs, it_time))
            logger.log_metric('train.data_time', data_time)
            logger.log_metric('train.compute_time', it_time - data_time)

        end = time.time()


def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            group = torch.distributed.group.WORLD
            torch.distributed.all_reduce(loss.data, group=group)
            torch.distributed.all_reduce(prec1, group=group)
            torch.distributed.all_reduce(prec5, group=group)
            reduced_loss = loss.data / torch.distributed.get_world_size(group)
            prec1 /= torch.distributed.get_world_size(group)
            prec5 /= torch.distributed.get_world_size(group)
            #reduced_loss = reduce_tensor(loss.data)
            #prec1 = reduce_tensor(prec1)
            #prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(val_loader,
             model_and_loss,
             fp16,
             logger,
             epoch,
             register_metrics=True):
    if register_metrics and logger is not None:
        logger.register_metric('val.top1',
                               log.ACC_METER(),
                               verbosity=dllogger.Verbosity.DEFAULT,
                               metadata=ACC_METADATA)
        logger.register_metric('val.top5',
                               log.ACC_METER(),
                               verbosity=dllogger.Verbosity.DEFAULT,
                               metadata=ACC_METADATA)
        logger.register_metric('val.loss',
                               log.LOSS_METER(),
                               verbosity=dllogger.Verbosity.DEFAULT,
                               metadata=LOSS_METADATA)
        logger.register_metric('val.compute_ips',
                               log.PERF_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=IPS_METADATA)
        logger.register_metric('val.total_ips',
                               log.PERF_METER(),
                               verbosity=dllogger.Verbosity.DEFAULT,
                               metadata=IPS_METADATA)
        logger.register_metric('val.data_time',
                               log.PERF_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=TIME_METADATA)
        logger.register_metric('val.compute_latency',
                               log.PERF_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=TIME_METADATA)
        logger.register_metric('val.compute_latency_at100',
                               log.LAT_100(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=TIME_METADATA)
        logger.register_metric('val.compute_latency_at99',
                               log.LAT_99(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=TIME_METADATA)
        logger.register_metric('val.compute_latency_at95',
                               log.LAT_95(),
                               verbosity=dllogger.Verbosity.VERBOSE,
                               metadata=TIME_METADATA)


    step = get_val_step(model_and_loss)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        if logger is not None:
            logger.log_metric('val.top1', to_python_float(prec1), bs)
            logger.log_metric('val.top5', to_python_float(prec5), bs)
            logger.log_metric('val.loss', to_python_float(loss), bs)
            logger.log_metric('val.compute_ips',
                              calc_ips(bs, it_time - data_time))
            logger.log_metric('val.total_ips', calc_ips(bs, it_time))
            logger.log_metric('val.data_time', data_time)
            logger.log_metric('val.compute_latency', it_time - data_time)
            logger.log_metric('val.compute_latency_at95', it_time - data_time)
            logger.log_metric('val.compute_latency_at99', it_time - data_time)
            logger.log_metric('val.compute_latency_at100', it_time - data_time)

        end = time.time()

    return top1.get_val()


# Train loop {{{
def calc_ips(batch_size, time):
    world_size = torch.distributed.get_world_size(
    ) if torch.distributed.is_initialized() else 1
    tbs = world_size * batch_size
    return tbs / time


def train_loop(model_and_loss,
               optimizer,
               lr_scheduler,
               train_loader,
               val_loader,
               epochs,
               fp16,
               logger,
               should_backup_checkpoint,
               use_amp=False,
               batch_size_multiplier=1,
               best_prec1=0,
               start_epoch=0,
               skip_training=False,
               skip_validation=False,
               save_checkpoints=True,
               checkpoint_dir='./'):

    prec1 = -1

    epoch_iter = range(start_epoch, epochs)
    for epoch in epoch_iter:
        if logger is not None:
            logger.start_epoch()
        if not skip_training:
            train(train_loader,
                  model_and_loss,
                  optimizer,
                  lr_scheduler,
                  fp16,
                  logger,
                  epoch,
                  use_amp=use_amp,
                  register_metrics=epoch == start_epoch,
                  batch_size_multiplier=batch_size_multiplier)

        if not skip_validation:
            prec1, nimg = validate(val_loader,
                                   model_and_loss,
                                   fp16,
                                   logger,
                                   epoch,
                                   register_metrics=epoch == start_epoch)
        if logger is not None:
            logger.end_epoch()

        if save_checkpoints and (not torch.distributed.is_initialized()
                                 or torch.distributed.get_rank() == 0):
            if not skip_validation:
                is_best = logger.metrics['val.top1']['meter'].get_epoch() > best_prec1
                best_prec1 = max(logger.metrics['val.top1']['meter'].get_epoch(),
                                 best_prec1)
            else:
                is_best = False
                best_prec1 = 0

            if should_backup_checkpoint(epoch):
                backup_filename = 'checkpoint-{}.pth.tar'.format(epoch + 1)
            else:
                backup_filename = None
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': model_and_loss.arch,
                    'state_dict': model_and_loss.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                checkpoint_dir=checkpoint_dir,
                backup_filename=backup_filename)
# }}}


def add_parser_arguments(parser):
    model_names = [key for key in resnet_models.resnet_versions] + ['alexnet']
    model_configs = resnet_models.resnet_configs.keys()

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--data-backend',
                        metavar='BACKEND',
                        default='pytorch',
                        choices=DATA_BACKEND_CHOICES,
                        help='data backend: ' +
                        ' | '.join(DATA_BACKEND_CHOICES) +
                        ' (default: pytorch)')

    parser.add_argument('--arch',
                        '-a',
                        metavar='ARCH',
                        default='resnet50',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' | alexnet (default: resnet50)')

    parser.add_argument('--model-config',
                        '-c',
                        metavar='CONF',
                        default='classic',
                        choices=model_configs,
                        help='model configs: ' + ' | '.join(model_configs) +
                        '(default: classic)')

    parser.add_argument('-j',
                        '--workers',
                        default=5,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 5)')
    parser.add_argument('--epochs',
                        default=90,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b',
                        '--batch-size',
                        default=256,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256) per gpu')

    parser.add_argument(
        '--optimizer-batch-size',
        default=-1,
        type=int,
        metavar='N',
        help=
        'size of a total batch size, for simulating bigger batches using gradient accumulation'
    )

    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--lr-schedule',
                        default='step',
                        type=str,
                        metavar='SCHEDULE',
                        choices=['step', 'linear', 'cosine'],
                        help='Type of LR schedule: {}, {}, {}'.format(
                            'step', 'linear', 'cosine'))

    parser.add_argument('--warmup',
                        default=0,
                        type=int,
                        metavar='E',
                        help='number of warmup epochs')

    parser.add_argument('--label-smoothing',
                        default=0.0,
                        type=float,
                        metavar='S',
                        help='label smoothing')
    parser.add_argument('--mixup',
                        default=0.0,
                        type=float,
                        metavar='ALPHA',
                        help='mixup alpha')

    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay',
                        '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='use nesterov momentum, (default: false)')

    parser.add_argument('--print-freq',
                        '-p',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained-weights',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='load weights from here')

    parser.add_argument('--fp16',
                        action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument(
        '--static-loss-scale',
        type=float,
        default=1,
        help=
        'Static loss scale, positive power of 2 values can improve fp16 convergence.'
    )
    parser.add_argument(
        '--dynamic-loss-scale',
        action='store_true',
        help='Use dynamic loss scaling.  If supplied, this argument supersedes '
        + '--static-loss-scale.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='Run model AMP (automatic mixed precision) mode.')

    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='random seed used for numpy and pytorch')

    parser.add_argument(
        '--gather-checkpoints',
        action='store_true',
        help=
        'Gather checkpoints throughout the training, without this flag only best and last checkpoints will be stored'
    )

    parser.add_argument('--raport-file',
                        default='experiment_raport.json',
                        type=str,
                        help='file in which to store JSON experiment raport')

    parser.add_argument('--evaluate',
                        action='store_true',
                        help='evaluate checkpoint/model')
    parser.add_argument('--training-only',
                        action='store_true',
                        help='do not evaluate')

    parser.add_argument(
        '--no-checkpoints',
        action='store_false',
        dest='save_checkpoints',
        help='do not store any checkpoints, useful for benchmarking')

    parser.add_argument(
        '--workspace',
        type=str,
        default='./',
        metavar='DIR',
        help='path to directory where checkpoints will be stored')

    parser.add_argument(
        '--local_rank',
        default=0,
        type=int)

def main(args):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ['LOCAL_RANK'])

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:

        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print(
                "Warning:  if --fp16 is not used, static_loss_scale will be ignored."
            )

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}"
                .format(args.optimizer_batch_size, tbs))
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    pretrained_weights = None
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print("=> loading pretrained weights from '{}'".format(
                args.pretrained_weights))
            pretrained_weights = torch.load(args.pretrained_weights)
        else:
            print("=> no pretrained weights found at '{}'".format(args.resume))

    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(args.gpu))
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_state = checkpoint['state_dict']
            optimizer_state = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None

    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    model_and_loss = ModelAndLoss((args.arch, args.model_config),
                                  loss,
                                  pretrained_weights=pretrained_weights,
                                  cuda=True,
                                  fp16=args.fp16)

    # Create data loaders and optimizers as needed
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'syntetic':
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader

    train_loader, train_loader_len = get_train_loader(args.data,
                                                      args.batch_size,
                                                      1000,
                                                      args.mixup > 0.0,
                                                      workers=args.workers,
                                                      fp16=args.fp16)
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, 1000, train_loader)

    val_loader, val_loader_len = get_val_loader(args.data,
                                                args.batch_size,
                                                1000,
                                                False,
                                                workers=args.workers,
                                                fp16=args.fp16)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        logger = log.Logger(args.print_freq, [
            dllogger.StdOutBackend(dllogger.Verbosity.DEFAULT,
                               step_format=log.format_step),
            dllogger.JSONStreamBackend(
                dllogger.Verbosity.VERBOSE,
                os.path.join(args.workspace, args.raport_file))
        ])

    else:
        logger = log.Logger(args.print_freq, [])

    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)

    optimizer = get_optimizer(list(model_and_loss.model.named_parameters()),
                              args.fp16,
                              args.lr,
                              args.momentum,
                              args.weight_decay,
                              nesterov=args.nesterov,
                              state=optimizer_state,
                              static_loss_scale=args.static_loss_scale,
                              dynamic_loss_scale=args.dynamic_loss_scale)

    if args.lr_schedule == 'step':
        lr_policy = lr_step_policy(args.lr, [30, 60, 80],
                                   0.1,
                                   args.warmup,
                                   logger=logger)
    elif args.lr_schedule == 'cosine':
        lr_policy = lr_cosine_policy(args.lr,
                                     args.warmup,
                                     args.epochs,
                                     logger=logger)
    elif args.lr_schedule == 'linear':
        lr_policy = lr_linear_policy(args.lr,
                                     args.warmup,
                                     args.epochs,
                                     logger=logger)

    if args.amp:
        model_and_loss, optimizer = amp.initialize(
            model_and_loss,
            optimizer,
            opt_level="O2",
            loss_scale="dynamic"
            if args.dynamic_loss_scale else args.static_loss_scale)

    if args.distributed:
        model_and_loss.distributed()

    model_and_loss.load_model_state(model_state)

    train_loop(model_and_loss,
               optimizer,
               lr_policy,
               train_loader,
               val_loader,
               args.epochs,
               args.fp16,
               logger,
               should_backup_checkpoint(args),
               use_amp=args.amp,
               batch_size_multiplier=batch_size_multiplier,
               start_epoch=start_epoch,
               best_prec1=best_prec1,
               skip_training=args.evaluate,
               skip_validation=args.training_only,
               save_checkpoints=args.save_checkpoints and not args.evaluate,
               checkpoint_dir=args.workspace)
    exp_duration = time.time() - exp_start_time
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        logger.end()
    print("Experiment ended")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True

    main(args)
