#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Using custome network to classify data sampled from MNIST data.

Network can be customed, see class My_Model for more details about how to custome model,
which will be added with a classification layer.
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

import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import math
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_prepare import my_dataset_custome
from model_define.model import My_Model
from utils.utils import estim_tau_tensor

device = "cuda:2" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    # gpu_usage()
    # ---------------------------------Data---------------------------------------------
    cs = [0.5, 0.5]
    K = len(cs)

    # load data
    res = my_dataset_custome('var',
                             T_train=10000,
                             T_test=1800,
                             p=784,
                             cs=[0.5, 0.5])
    # res = my_dataset_custome('MNIST', T_train=10000, T_test=1800, cs=cs, selected_target=[6,  8])
    dataset_train, dataset_test = res[0], res[1]
    'to be done here new net configuration here just binaryzero and binary last'
    tau_zero = np.sqrt(estim_tau_tensor(dataset_train.X))
    print(tau_zero)

    # --------------------------------Network--------------------------------------------
    layer_num = 2  # layer number for network
    input_num = 784  # input dimension for network
    weight_num_list = [3000, 1000]  # number for neurons for each layer
    activation_list = [
        {
            'name': 'ReLU',
            'args': None
        },
        {
            'name': 'ReLU',
            'args': None
        },
        {
            'name': 'ReLU',
            'args': None
        },
    ]

    # define origin model
    model = My_Model(layer_num=layer_num,
                     input_num=input_num,
                     weight_num_list=weight_num_list,
                     activation_list=activation_list,
                     tau_zero=tau_zero)

    # add a classification layer
    model_origin = nn.Sequential(
        OrderedDict([
            ('feature', model),
            ('classification',
             nn.Linear(model.weight_num_list[-1], 2, bias=False)),
            ('activation', nn.Softmax()),
        ]))

    # model initialization
    for fc in model_origin.feature.fc_layers:
        nn.init.normal_(fc.weight)
        fc.weight.requires_grad = False

    # --------------------------------Preparing-------------------------------------------
    # define trainning network
    net = model_origin

    batch_size = 128
    lr = 0.01
    config = {
        "save_path": "./compression/mnist/model_origin",
        "early_stop": 20,
        'n_epochs': 200
    }
    epochs, best_loss, step, early_stop_count = config[
        'n_epochs'], math.inf, 0, 0

    # save path
    if not os.path.exists(os.path.split(config['save_path'])[0]):
        os.makedirs(os.path.split(config['save_path'])[0])
    # define data
    dataset_train.Y = F.one_hot(torch.tensor(dataset_train.Y).long(), K)
    dataset_train.Y = dataset_train.Y.float()
    dataset_test.Y = F.one_hot(torch.tensor(dataset_test.Y).long(), K)
    dataset_test.Y = dataset_test.Y.float()
    train_loader = DataLoader(dataset_train,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=True)
    test_loader = DataLoader(dataset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=True)
    # shuffle?????????????????????????
    net = net.to(device)
    if device == 'cuda:2':
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    # --------------------------Trainning and Validation-----------------------------------
    for epoch in range(epochs):
        net.train()
        loss_record = []
        accuracy_record = []

        # trainning
        for train_data, train_label in train_loader:
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
            _, index_label = train_label.data.cpu().topk(1, dim=1)
            accuracy_batch = np.sum(
                (index.squeeze(dim=1) == index_label.squeeze(dim=1)).numpy())
            accuracy_batch = accuracy_batch / len(train_label)
            accuracy_record.append(accuracy_batch)
        train_loss = sum(loss_record) / len(loss_record)
        train_accuracy = sum(accuracy_record) / len(accuracy_record)

        # validation
        net.eval()
        loss_record = []
        accuracy_record = []
        for val_data, val_label in test_loader:
            val_data, val_label = val_data.to(device), val_label.to(device)
            with torch.no_grad():
                pred = net(val_data)
                loss = criterion(pred, val_label)
            loss_record.append(loss.item())
            # accuracy
            _, index = pred.data.cpu().topk(1, dim=1)
            _, index_label = val_label.data.cpu().topk(1, dim=1)
            accuracy_batch = np.sum(
                (index.squeeze(dim=1) == index_label.squeeze(dim=1)).numpy())
            accuracy_batch = accuracy_batch / len(val_label)
            accuracy_record.append(accuracy_batch)
        val_loss = sum(loss_record) / len(loss_record)
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
            print(
                '\nModel is not improving, so we halt the trainning session.')
            break

    # origin performance
    model_origin_final_accuracy, model_origin_final_loss = val_accuracy, val_loss

    print(
        f'origin model: Valid loss: {model_origin_final_loss:.4f}, Valid accuracy: {model_origin_final_accuracy:.4f}'
    )
