# main.py COPYRIGHT Fujitsu Limited 2021

from argparse import ArgumentParser
from collections import OrderedDict
import copy
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

#from cac import auto_prune
import sys
sys.path.append('../../')
from auto_prune import auto_prune

from alexnet import AlexNet

#===================================================================================
parser = ArgumentParser()
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers')
parser.add_argument('--use_gpu', action='store_true',
                    help='use gpu')
parser.add_argument('--use_DataParallel', action='store_true',
                    help='use DataParallel')
# for training
parser.add_argument('--data', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--nesterov', default=False)
# for auto pruning
parser.add_argument('--acc_control', type=float, default=1.0,
                    help='control parameter for pruned model accuracy')
parser.add_argument('--rates', nargs='*', type=float, default=[0.2, 0.1, 0.0],
                    help='candidates for pruning rates')
parser.add_argument('--max_search_times', type=int, default=1000,
                    help='maximum number of times for pruning rate search')
parser.add_argument('--epochs', type=int, default=300,
                    help='re-training epochs')
parser.add_argument('--model_path', type=str, default='./pretrained_cifar10_alexnet.pt',
                    help='pre-trained model filepath')
parser.add_argument('--pruned_model_path', type=str, default='./pruned_cifar10_alexnet.pt',
                    help='pruned model filepath')
#===================================================================================

def main():
    args = parser.parse_args()
    args.rates = ([float(f) for f in args.rates])
    print(f'args: {args}')

    device = 'cpu'
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    print('===== load data ===================')
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # get cifar10 datasets
    dataset_path = args.data
    train_dataset = datasets.CIFAR10(root=dataset_path, train=True,
                                     download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=dataset_path, train=False,
                                   download=True, transform=val_transform)

    # make DataLoader
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    # load model
    model = AlexNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device),
                          strict=True)

    if torch.cuda.device_count() > 1 and args.use_DataParallel:
        print('use {} GPUs.'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)
    print('===== model: before pruning ==========')
    print(model)
    # Model information for pruning
    model_info = OrderedDict(conv1={'arg': 'out_ch_conv1'},
                             conv2={'arg': 'out_ch_conv2'},
                             conv3={'arg': 'out_ch_conv3'},
                             conv4={'arg': 'out_ch_conv4'},
                             conv5={'arg': 'out_ch_conv5'},
                             fc1={'arg': 'out_ch_fc1'},
                             fc2={'arg': 'out_ch_fc2'},
                             fc3={'arg': None})

    # load weight of trained model
    if torch.cuda.device_count() > 1 and args.use_DataParallel:
        weights = copy.deepcopy(model.module.state_dict())
    else:
        weights = copy.deepcopy(model.state_dict())

    # calculate accuracy with non-pruning trained model
    Ab = validate(val_loader, model, device, epoch=1)
    print('Accuracy before pruning:', Ab)

    # tune pruning rate
    print('===== start pruning rate tuning =====')
    # set optimizer
    optim_params = dict(lr=args.learning_rate,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=args.nesterov)
    # set LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_params = dict(T_max=200)

    criterion = torch.nn.CrossEntropyLoss()

    weights, Afinal, n_args_channels = auto_prune(AlexNet, model_info, weights, Ab,
                                                  train_loader, val_loader, criterion,
                                                  optim_type='SGD',
                                                  optim_params=optim_params,
                                                  lr_scheduler=scheduler,
                                                  scheduler_params=scheduler_params,
                                                  update_lr='epoch',
                                                  use_gpu=args.use_gpu,
                                                  use_DataParallel=args.use_DataParallel,
                                                  acc_control=args.acc_control,
                                                  rates=args.rates,
                                                  max_search_times=args.max_search_times,
                                                  epochs=args.epochs,
                                                  model_path=args.model_path,
                                                  pruned_model_path=args.pruned_model_path)

    print('===== model: after pruning ==========')
    print(model)
    print('===== Results =====')
    print('Model size before pruning (Byte):', os.path.getsize(args.model_path))
    if os.path.exists(args.pruned_model_path):
        print('Model size after pruning  (Byte):',
              os.path.getsize(args.pruned_model_path))
        print('Compression rate                : {:.3f}'.format(
            1-os.path.getsize(args.pruned_model_path)/os.path.getsize(args.model_path)))
    print('Acc. before pruning: {:.2f}'.format(Ab))
    print('Acc. after pruning : {:.2f}'.format(Afinal))
    print('Arguments name & number of channels for pruned model: ', n_args_channels)

def validate(val_loader, model, device, epoch):
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, leave=False) as pbar:
            pbar.set_description('Epoch {} Validation'.format(epoch))
            hit = 0
            total = 0
            for _, (images, targets) in enumerate(pbar):
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                outClass = outputs.cpu().detach().numpy().argmax(axis=1)
                hit += (outClass == targets.cpu().numpy()).sum()
                total += len(targets)
                val_acc = hit / total * 100
                pbar.set_postfix({'valid Acc': val_acc})
    return val_acc

if __name__ == '__main__':
    main()

