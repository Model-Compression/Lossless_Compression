#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train VGG on all CIFAR10 dataset, which will be further used to concatenate with
some random feature layers and classification layer.
"""

__author__ = "Model_compression"
__copyright__ = "Copyright 2021, Lossless compression"
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "yongqi_du@hust.edu.cn"
__status__ = "Production"
__all__ = [
    'custome_activation_analysis', 'custome_activation_analysis_noparam'
]

import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from vgg_net_cifar10 import VGG, VGG_Feature

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    device_id = [1, 2, 3]
    batch_size = 64
    lr = 0.01
    config = {"save_path": "./model_vgg", "early_stop": 20, 'n_epochs': 2}

    # -------------------------------Data Preparation-----------------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data',
                                              train=True,
                                              download=True,
                                              transform=transform_train)
    trainloader = DataLoader(train_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=3,
                             pin_memory=True)

    test_data = torchvision.datasets.CIFAR10(root='./data',
                                             train=False,
                                             download=True,
                                             transform=transform_test)
    testloader = DataLoader(test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=3,
                            pin_memory=True)

    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog',
               'Horse', 'Ship', 'Truck')

    # ----------------------------------Model Setting-----------------------------------
    net = VGG('VGG19')

    # ----------------------------------Initilization-----------------------------------
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    epochs, best_loss, step, early_stop_count = config[
        'n_epochs'], math.inf, 0, 0
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------Train and Validation----------------------------------
    for epoch in range(epochs):
        net.train()
        loss_record = []
        accuracy_record = []
        for train_data, train_label in trainloader:
            optimizer.zero_grad()
            train_data, train_label = train_data.to(device), train_label.to(
                device)
            pred = net(train_data)
            loss = criterion(pred, train_label)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
            # accuracy
            _, index = pred.data.cpu().topk(1, dim=1)
            index_label = train_label.data.cpu()
            accuracy_batch = np.sum(
                (index.squeeze(dim=1) == index_label).numpy())
            accuracy_batch = accuracy_batch / len(train_label)
            accuracy_record.append(accuracy_batch)
        train_loss = sum(loss_record) / len(loss_record)
        train_accuracy = sum(accuracy_record) / len(accuracy_record)

        # Validation
        net.eval()
        loss_record = []
        accuracy_record = []
        for val_data, val_label in testloader:
            val_data, val_label = val_data.to(device), val_label.to(device)
            with torch.no_grad():
                pred = net(val_data)
                loss = criterion(pred, val_label)
            loss_record.append(loss.item())
            # accuracy
            _, index = pred.data.cpu().topk(1, dim=1)
            index_label = val_label.data.cpu()
            accuracy_batch = np.sum(
                (index.squeeze(dim=1) == index_label).numpy())
            accuracy_batch = accuracy_batch / len(val_label)
            accuracy_record.append(accuracy_batch)
        val_loss = sum(loss_record) / len(testloader)
        val_accuracy = sum(accuracy_record) / len(accuracy_record)

        print(
            f'Epoch [{epoch+1}/{epochs}]: Train loss: {train_loss:.4f},Train accuracy: {train_accuracy:.4f}, Valid loss: {val_loss:.4f}, Valid accuracy: {val_accuracy:.4f}'
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(),
                       config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            break
    'here just a test for all binary activation'
