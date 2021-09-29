# make_model.py COPYRIGHT Fujitsu Limited 2021

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from mlp import MultiLayerPerceptron

# model filepath
MODEL_PATH = './mlp.pt'


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    print('===== load data ===================')
    # Transform into Tensor
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    # get cifar10 datasets
    dataset_path = './data'
    train_dataset = datasets.MNIST(dataset_path, train=True, download=True,
                                   transform=transform)
    val_dataset = datasets.MNIST(dataset_path, train=False,
                                 transform=transform)

    # make DataLoader
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

    # initialize model
    model = MultiLayerPerceptron()
    if torch.cuda.device_count() > 1:
        print('use {} GPUs.'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.9, weight_decay=1e-4)

    # train and valitate model
    print('===== start train ===============')
    epochs = 15
    for epoch in range(epochs):
        train(train_loader, model, device, criterion, optimizer, epoch+1)
        acc = validate(val_loader, model, device, epoch+1, 100)
        print('Epoch: {}/{}, Acc: {}'.format(epoch+1, epochs, acc))

    # save model
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), MODEL_PATH)
    else:
        torch.save(model.state_dict(), MODEL_PATH)
    print('===== finish train ==============')


def train(train_loader, model, device, criterion, optimizer, epoch):
    model.train()
    hit = 0
    total = 0
    with tqdm(train_loader, leave=False) as pbar:
        for _, (images, targets) in enumerate(pbar):
            pbar.set_description('Epoch {} Training'.format(epoch))
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outClass = outputs.cpu().detach().numpy().argmax(axis=1)
            hit += (outClass == targets.cpu().numpy()).sum()
            total += len(targets)
            pbar.set_postfix({'train Acc': hit / total * 100})


def validate(val_loader, model, device, epoch, valid_iter):
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, leave=False) as pbar:
            pbar.set_description('Epoch {} Validation'.format(epoch))
            hit = 0
            total = 0
            for i, (images, targets) in enumerate(pbar):
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                outClass = outputs.cpu().detach().numpy().argmax(axis=1)
                hit += (outClass == targets.cpu().numpy()).sum()
                total += len(targets)
                val_acc = hit / total * 100
                pbar.set_postfix({'valid Acc': val_acc})
                if i == valid_iter:
                    break
    return val_acc


if __name__ == '__main__':
    main()
