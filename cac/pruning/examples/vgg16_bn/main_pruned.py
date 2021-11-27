# main_pruned.py COPYRIGHT Fujitsu Limited 2021

from argparse import ArgumentParser
from collections import OrderedDict
import copy
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm

#from cac import auto_prune
import sys
sys.path.append('../../')
from auto_prune import auto_prune

from vgg16_bn_pruned import VGG16_BN
from schduler import WarmupCosineLR

#===================================================================================
parser = ArgumentParser()
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers')
parser.add_argument('--use_gpu', action='store_true',
                    help='use gpu')
parser.add_argument('--use_DataParallel', action='store_true',
                    help='use DataParallel')
parser.add_argument('--data', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model_path', type=str, default='./pretrained_cifar10_vgg16bn.pt',
                    help='pre-trained model filepath')
#===================================================================================

def main():
    args = parser.parse_args()
    print(f'args: {args}')

    device = 'cpu'
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    print('===== load data ===================')
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std  = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

    val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

    # get cifar10 datasets
    dataset_path = args.data
    train_dataset = datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(
        root=dataset_path, train=False, download=True, transform=val_transform)

    # make DataLoader
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=True,
        )

    # load model
    model = VGG16_BN()
    model.load_state_dict(torch.load(
        args.model_path, map_location=device), strict=True)

    if torch.cuda.device_count() > 1 and args.use_DataParallel:
        print('use {} GPUs.'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)
    print('===== model ==========')
    print(model)

    # load weight of trained model
    if torch.cuda.device_count() > 1 and args.use_DataParallel:
        weights = copy.deepcopy(model.module.state_dict())
    else:
        weights = copy.deepcopy(model.state_dict())

    # calculate accuracy with unpruned trained model
    Ab = validate(val_loader, model, device, epoch=1)
    print('Accuracy :', Ab)

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

