import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn  # import modules
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils.data_prepare import my_dataset_custome
from model_define.model import My_Model
from equation_solve.solve_equation import solve_equation
from utils.utils import estim_tau_tensor

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # gpu_usage()
    # ---------------------------------Data---------------------------------------------
    cs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    K = len(cs)
    # load data
    res = my_dataset_custome('MNIST',
                             T_train=50000,
                             T_test=8000,
                             cs=cs,
                             selected_target=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # res = my_dataset_custome('MNIST', T_train=10000, T_test=1600, cs=cs, selected_target=[6,  8])
    dataset_train, dataset_test = res[0], res[1]
    tau_zero = np.sqrt(estim_tau_tensor(dataset_train.X))
    print(tau_zero)

    # --------------------------------Network--------------------------------------------
    # origin network setting
    layer_num = 3  # layer number for network
    input_num = 784  # input dimension for network 784/256
    weight_num_list = [2000, 2000, 1100]  # number for neurons for each layer
    activation_list = [
        {
            'name': 'ReLU',
            'args': None
        },
        # {'name' : 'Binary_Zero', 'args' : {'s1':-1, 's2': 1, 'b1': 1}},
        # {'name' : 'Binary_Zero', 'args' : {'s1':-1, 's2': 1, 'b1': 1}},
        {
            'name': 'ReLU',
            'args': None
        },
        # {'name' : 'ReLU', 'args' : None},
        {
            'name': 'ReLU',
            'args': None
        }
    ]

    #  define origin model
    model = My_Model(layer_num=layer_num,
                     input_num=input_num,
                     weight_num_list=weight_num_list,
                     activation_list=activation_list,
                     tau_zero=tau_zero)

    res = solve_equation(model, tau_zero, loop=1000)
    activation_list = res
    new_model = My_Model(layer_num=layer_num,
                         input_num=input_num,
                         weight_num_list=weight_num_list,
                         activation_list=activation_list,
                         tau_zero=tau_zero)

    # define model
    model_origin = nn.Sequential(
        OrderedDict([
            ('feature', model),
            ('classification',
             nn.Linear(model.weight_num_list[-1], K, bias=False)),
            ('activation', nn.Softmax()),
        ]))
    model_new = nn.Sequential(
        OrderedDict([
            ('feature', new_model),
            ('classification',
             nn.Linear(new_model.weight_num_list[-1], K, bias=False)),
            ('activation', nn.Softmax()),
        ]))

    # -----------------------------weight initialization!---------------------------------
    initialization_way = 'normal'  # select from ['normal', 'random_sparsity', 'ternary']
    kesi = 0  # change from 0 to 1, only used for ['random sparsity', 'ternary']

    if initialization_way == 'normal':
        # normal initialization
        for fc in model_new.feature.fc_layers:
            nn.init.normal_(fc.weight)
            fc.weight.requires_grad = False
    elif initialization_way == 'random_sparsity':
        # random sparse gaussian weight(break assumption1)
        for fc in model_new.feature.fc_layers:
            mask = np.zeros(fc.weight.shape).flatten()
            mask[:round((1 - kesi) * mask.size)] = 1
            np.random.shuffle(mask)
            mask = torch.tensor(mask.reshape(fc.weight.shape)).float()
            nn.init.normal_(fc.weight)
            with torch.no_grad():
                fc.weight = torch.nn.Parameter(mask * fc.weight.data,
                                               requires_grad=False)
    elif initialization_way == 'ternary':
        # tarnary weight with sparsity kesi
        for fc in model_new.feature.fc_layers:
            init = np.zeros(fc.weight.shape).flatten()
            init[:round(1 / 2 * (1 - kesi) *
                        init.size)] = 1 / np.sqrt(1 - kesi)
            init[round(1 / 2 * (1 - kesi) * init.size):2 *
                 round(1 / 2 *
                       (1 - kesi) * init.size)] = -1 / np.sqrt(1 - kesi)
            # c = Counter(init)
            np.random.shuffle(init)
            init = torch.tensor(init.reshape(fc.weight.shape)).float()
            with torch.no_grad():
                fc.weight = torch.nn.Parameter(init, requires_grad=False)

    # --------------------------------Preparing-------------------------------------------
    # define network setting
    net = model_new

    batch_size = 128
    lr = 0.001
    config = {"save_path": "./model_origin", "early_stop": 20, 'n_epochs': 500}
    early_stop_count = 0
    epochs, best_loss, step, early_stop_count = config[
        'n_epochs'], math.inf, 0, 0

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
    # shuffle????????????

    net = net.to(device)
    if device == 'cuda:2':
        cudnn.benchmark = True

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # --------------------------Trainning and Validation-----------------------------------
    for epoch in range(epochs):
        net.train()
        loss_record = []
        accuracy_record = []
        # training
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
            _, index = pred.data.cpu().topk(1)
            _, index_label = val_label.data.cpu().topk(1, dim=1)
            accuracy_batch = np.sum(
                (index.squeeze(dim=1) == index_label.squeeze(dim=1)).numpy())
            accuracy_batch = accuracy_batch / len(val_label)
            accuracy_record.append(accuracy_batch)
        val_loss = sum(loss_record) / len(test_loader)
        val_accuracy = sum(accuracy_record) / len(accuracy_record)

        print(
            f'Epoch [{epoch+1}/{epochs}]: Train loss: {train_loss:.4f},Train accuracy: {train_accuracy:.4f}, Valid loss: {val_loss:.4f}, Valid accuracy: {val_accuracy:.4f}'
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), "./model_new")  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            break

    # origin performance
    model_new_final_accuracy, model_new_final_loss = val_accuracy, val_loss

    print(
        f'compressed model(with retrain): Valid loss: {model_new_final_loss:.4f}, Valid accuracy: {model_new_final_accuracy:.4f}'
    )
