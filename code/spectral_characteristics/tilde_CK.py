#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Verify consistency for spectral distrubutions of CK and CK_tilde.
'''

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_prepare import my_dataset_custome
from utils.expect_calculate import expect_calcu
from utils.model import My_Model
from utils.utils import estim_tau_tensor

from plot_eigen import plot_eigen

device = "cuda:2" if torch.cuda.is_available() else "cpu"


def calculate_CK(model, data, mode='normal'):
    """Calculate model_output * model_output.T for one time.(you can choose weight initialization: student t or normal)

    Arguments:
        model -- model instace of class MyModel
        data -- all data(used to calculate output)
        mode -- initilization type(student t or normal)
    Returns:
        Phi_loop -- output * output.T
    """
    for j in range(model.layer_num):
        with torch.no_grad():
            if mode == 'student t':
                #student t
                if j == 0:
                    r = scipy.stats.t.rvs(50.0, size=(model.weight_num_list[j], model.input_num))
                else:
                    r = scipy.stats.t.rvs(50.0, size=(model.weight_num_list[j], model.weight_num_list[j - 1]))
                model.fc_layers[j].weight = nn.Parameter(torch.tensor(r).float())
            elif mode == 'normal':
                # normal
                nn.init.normal_(model.fc_layers[j].weight)
                model.fc_layers[j].weight.requires_grad = False
    with torch.no_grad():
        out = model(data).detach().cpu().numpy()
    Phi_loop = out @ out.T
    return Phi_loop


def calculate_CK_loop(model, data, loop):
    """Calculate model_output * model_output.T for loop times.

    Arguments:
        model -- model instace of class MyModel
        data -- all data(used to calculate output)
        loop -- loop times 
    Returns:
        Phi_loop -- means of output * output.T over loop times
    """
    mode = 'normal'  # you can change to student t here.
    [n, _] = data.shape
    Phi = np.zeros((n, n))
    for i in range(loop):
        print(i)
        r = calculate_CK(model, data, mode)
        Phi = Phi + r
    Phi = Phi / loop
    print(Phi)

    return Phi


def calculate_CK_tilde_coef(model, tau_zero):
    '''calculate coefficients each layer(alpha1, alpha2, alpha3, alpha4, tau) and print.
    Note: no variables, this function can be used only for a known network(without unknown variables for activation functions)

    Arguments:
        model -- model instance of class My_Model in model.py
        tau_zero -- tau_zero calculated with data
    Returns:
        d_last -- alpha1, alpha2, alpha3, alpha4, tau for last layer, which can be used to calculate CK_tilde
    '''
    tao_last = tau_zero
    d_last = np.array([tao_last, 1, 0, 0, 1])  # input d1, d2, d3, d4
    for activation in model.activation_list:
        name = activation['name']
        args = activation['args']
        if args:
            zero_order, first_order, second_order, square_second_order, tau = expect_calcu(name, **args)
        else:
            zero_order, first_order, second_order, square_second_order, tau = expect_calcu(name)
        temp = zero_order(tao_last)
        print(temp)
        d1 = first_order(tao_last)**2 * d_last[1]
        d2 = first_order(tao_last)**2 * d_last[2] + 1 / 4 * second_order(tao_last)**2 * d_last[4]**2
        d3 = first_order(tao_last)**2 * d_last[3] + 1 / 2 * second_order(tao_last)**2 * d_last[1]**2
        d4 = 1 / 2 * square_second_order(tao_last) * d_last[4]
        tao_last = np.sqrt(tau(tao_last))
        d_last = np.array([tao_last, d1, d2, d3, d4])
        print(d1, d2, d3, d4)
        print(tao_last)
    return d_last


def calculate_CK_tilde(model, tau_zero, X, T, K, p, means, covs, y, Omega):
    """Calculate CK_tilde using expressions we derived.

    Arguments:
        model -- model instance of class MyModel
        tau_zero -- tau_zero calculated using all data
        X -- data input
        T -- number of data
        K -- number of class
        p -- dimension of data
        means -- means for different classes 
        covs -- covs for different classes 
        y -- labels
        Omega -- data - means
    Returns:
        tilde_Phi -- calculated CK_tilde using expressions we derived
    """
    d_last = calculate_CK_tilde_coef(model, tau_zero)
    M = np.array([]).reshape(p, 0)
    t0 = []
    J = np.zeros((T, K))

    for i in range(K):
        M = np.concatenate((M, means[i].reshape(p, 1)), axis=1)
        t0.append(np.trace(covs[i]) / p)
        J[:, i] = (y == i) * 1

    phi = np.diag(Omega.T @ Omega - J @ t0)
    t = (t0 - tau_zero**2) * np.sqrt(p)
    S = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            S[i, j] = np.trace(covs[i] @ covs[j]) / p

    V = np.concatenate((J / np.sqrt(p), phi.reshape(T, 1)), axis=1)  # whats omega here for
    A11 = d_last[2] * np.outer(t, t) + d_last[3] * S
    A = np.zeros((K + 1, K + 1))
    A[0:K, 0:K] = A11
    A[0:K, K] = d_last[2] * t
    A[K, 0:K] = d_last[2] * t.T
    A[K, K] = d_last[2]

    tilde_Phi = d_last[1] * (X) @ (X.T) + V @ A @ (V.T) + (
        d_last[0]**2 - d_last[1] * tau_zero**2 - d_last[3] * tau_zero**4) * np.eye(T)  # check tau and tau**2 ,am i right here?

    print(d_last)

    return tilde_Phi


if __name__ == "__main__":

    cs = [0.5, 0.5]
    K = len(cs)
    # load data
    res = my_dataset_custome('mixed', T_train=8000, T_test=0, cs=cs, p=2000)
    # res = my_dataset_custome('MNIST',T_train=1000, T_test=0, cs=cs, selected_target=[6, 8])
    dataset_train = res[0]
    X, T, K, p, means, covs, y, Omega = dataset_train.X, res[6], res[4], res[5], res[2], res[3], dataset_train.Y, res[8]

    train_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    data_inference, _ = next(iter(train_loader))

    tau_zero = np.sqrt(estim_tau_tensor(X))
    print(tau_zero)

    # origin network setting
    layer_num = 3  # layer number for network
    input_num = 2000  # input dimension for network 784/256
    weight_num_list = [10000, 10000, 10000, 10000]  # number for neurons for each layer
    activation_list = [
        # {'name' : 'Sigmoid', 'args' : None},
        # {'name' : 'Binary_Zero', 'args' : {'s1':1, 's2': 2, 'b1': 1}},
        # {'name' : 'poly2', 'args' : {'coe1': 1, 'coe2': 1 , 'coe3': 1}},
        {
            'name': 'ReLU',
            'args': None
        },
        {
            'name': 'ReLU',
            'args': None
        },
        # {'name' : 'poly2', 'args' : {'coe1': 1, 'coe2': 1 , 'coe3': 1}},
        # {'name' : 'Sigmoid', 'args' : None}
        {
            'name': 'ReLU',
            'args': None
        }  # activation for each layer, if with param, write as Binary_Zero here
    ]

    #  define origin model
    model = My_Model(layer_num=layer_num,
                     input_num=input_num,
                     weight_num_list=weight_num_list,
                     activation_list=activation_list,
                     tau_zero=tau_zero)

    loop = 500
    # calculate two CK_tilde
    CK_tilde = calculate_CK_tilde(model, tau_zero, X, T, K, p, means, covs, y, Omega)
    # CK_new = calculate_CK_tilde(new_model, tau_zero)

    CK_loop = calculate_CK_loop(model, data_inference, loop=loop)
    # a = scipy.linalg.norm(CK_origin, CK_new, ord=2)
    # print(a)
    # performance calculate

    # CK_loop = CK_tilde
    # CK_tilde = CK_loop

    setting = {
        'data': 'GMM-mixed',
        'T': str(T),
        'p': str(p),
        'layer_num': str(model.layer_num),
        'loop': str(loop),
        'activation': str(activation_list),
        'weight_num_list': str(weight_num_list)
    }
    plot_eigen(CK_tilde, CK_loop, setting=setting)
