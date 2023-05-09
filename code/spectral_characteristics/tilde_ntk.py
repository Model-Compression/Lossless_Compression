#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Verify consistency for spectral distrubutions of CK and CK_tilde.
'''
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_prepare import my_dataset_custome
from expect_cal.expect_calculate import expect_calcu
from model_define.model import My_Model
from utils.utils import estim_tau_tensor
from plot_eigen import plot_eigen


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from functorch import make_functional, vmap, vjp, jvp, jacrev

device = "cuda" if torch.cuda.is_available() else "cpu"



def calculate_NTK_tilde_coef(model, tau_zero):
    '''calculate coefficients each layer(beta1, beta2, beta3, tau, prime_tau) and print.
    Note: no variables, this function can be used only for a known network(without unknown variables for activation functions)

    Arguments:
        model -- model instance of class My_Model in model.py
        tau_zero -- tau_zero calculated with data
    Returns:
        beta_last -- beta1, beta2, beta3, tau, and prime_tau for last layer, which can be used to calculate NTK_tilde
    '''
    tao_last = tau_zero
    kappa_last_square = tau_zero # prime_tau defined for L bigger than 1
    beta_last = np.array([tao_last, 1, 0, 0, kappa_last_square])   # input beta1, beta2, beta3, beta4
    alpha_last = np.array([tao_last, 1, 0, 0, 1])  # input alpha1, alpha2, alpha3, alpha4
    
    for activation in model.activation_list:
        name = activation['name']
        args = activation['args']
        if args:
            zero_order, first_order, second_order, square_second_order, tau, prime_tau = expect_calcu(
                name, prime_tau_cal=True, **args)
        else:
            zero_order, first_order, second_order, square_second_order, tau, prime_tau = expect_calcu(
                name, prime_tau_cal=True)
        temp = zero_order(tao_last)
        print(temp)
        # calculation for alpha
        alpha1 = first_order(tao_last)**2 * alpha_last[1]
        alpha2 = first_order(tao_last)**2 * alpha_last[2] + 1 / 4 * second_order(
            tao_last)**2 * alpha_last[4]**2
        alpha3 = first_order(tao_last)**2 * alpha_last[3] + 1 / 2 * second_order(
            tao_last)**2 * alpha_last[1]**2
        alpha4 = 1 / 2 * square_second_order(tao_last) * alpha_last[4]

        # calculation for beta
        dot_alpha0 = first_order(tao_last)**2
        dot_alpha1 = second_order(tao_last)**2 * alpha_last[1]
        beta1 = alpha1 + beta_last[1] * dot_alpha0
        beta2 = alpha2 + beta_last[2] * dot_alpha0
        beta3 = alpha3 + beta_last[3] * dot_alpha0 + beta_last[1] * dot_alpha1

        # calculate and iterate tao_last
        tao_last = np.sqrt(tau(tao_last))

        # calculate for kappa
        prime_tau_last_square = prime_tau(tao_last)
        kappa_last_square = tau(tao_last) + kappa_last_square * prime_tau_last_square

        # print alpha
        print('alpha:\n')
        alpha_last = np.array([tao_last, alpha1, alpha2, alpha3, alpha4])
        print(alpha1, alpha2, alpha3, alpha4)
        print(tao_last)

        # print beta
        print('beta:\n')
        beta_last = np.array([tao_last, beta1, beta2, beta3, kappa_last_square])
        print(beta1, beta2, beta3)
        print(tao_last, kappa_last_square)

    return beta_last
    
def calculate_NTK_tilde(model, tau_zero, X, T, K, p, means, covs, y, Omega):
    """Calculate NTK_tilde using expressions we derived.

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
    d_last = calculate_NTK_tilde_coef(model, tau_zero)
    M = np.array([]).reshape(p, 0)
    t0 = []
    J = np.zeros((T, K))

    for i in range(K):
        M = np.concatenate((M, means[i].reshape(p, 1)), axis=1)
        t0.append(np.trace(covs[i]) / p) # \EE[z^T*z]
        J[:, i] = (y == i) * 1

    phi = np.diag(Omega.T @ Omega - J @ t0)
    t = (t0 - tau_zero**2) * np.sqrt(p)
    S = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            S[i, j] = np.trace(covs[i] @ covs[j]) / p

    V = np.concatenate((J / np.sqrt(p), phi.reshape(T, 1)),
                       axis=1)  # whats omega here for
    A11 = d_last[2] * np.outer(t, t) + d_last[3] * S
    A = np.zeros((K + 1, K + 1))
    A[0:K, 0:K] = A11
    A[0:K, K] = d_last[2] * t
    A[K, 0:K] = d_last[2] * t.T
    A[K, K] = d_last[2]

    tilde_Phi = d_last[1] * (X) @ (X.T) + V @ A @ (V.T) + (
        d_last[4]**2 - d_last[1] * tau_zero**2 - d_last[3] *
        tau_zero**4) * np.eye(T)  # check tau and tau**2 ,am i right here?

    print(d_last)

    return tilde_Phi

def empirical_ntk_ntk_vps(func, params, x1, x2, compute='full'):
    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)
        
    # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to empirical_ntk_ntk_vps are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the vmaps here do.
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    
    if compute == 'full':
        return result
    if compute == 'trace':
        return torch.einsum('NMKK->NM', result)
    if compute == 'diagonal':
        return torch.einsum('NMKK->NMK', result)
    

def claculate_empirical_NTK(model, x_test, x_train):
    net = model.to(device)
    fnet, params = make_functional(net)
    fnet_single = lambda params, x: fnet(params, x.unsqueeze(0)).squeeze(0)
    x_train = torch.tensor(x_train)
    x_test = torch.tensor(x_test)
    result_from_ntk_vps = empirical_ntk_ntk_vps(fnet_single, params, x_test, x_train)
    return result_from_ntk_vps


if __name__ == "__main__":
    cs = [0.5, 0.5]
    K = len(cs)
    # load data
    res = my_dataset_custome('means_binary', T_train=8000, T_test=0, cs=cs, p=4000)
    # res = my_dataset_custome('mixed', T_train=8000, T_test=0, cs=cs, p=4650)
    # res = my_dataset_custome('MNIST',
    #                          T_train=3200,
    #                          T_test=0,
    #                          cs=cs,
    #                          selected_target=[6, 8])
    dataset_train = res[0]
    X, T, K, p, means, covs, y, Omega = dataset_train.X, res[6], res[4], res[
        5], res[2], res[3], dataset_train.Y, res[8]
    print(X.shape)

    train_loader = DataLoader(dataset_train,
                              batch_size=len(dataset_train),
                              shuffle=False)
    data_inference, _ = next(iter(train_loader))

    tau_zero = np.sqrt(estim_tau_tensor(X))
    print(tau_zero)

    # origin network setting
    layer_num = 3  # layer number for network
    input_num = 4000  # input dimension for network 784/256
    # number for neurons for each layer
    weight_num_list = [2000, 2000, 1000]
    activation_list = [
        # {'name' : 'Sigmoid', 'args' : None},
        # {'name' : 'Binary_Zero', 'args' : {'s1':1, 's2': 2, 'b1': 1}},
        # {
        #     'name': 'Poly2',
        #     'args': {
        #         'coe1': 0.2,
        #         'coe2': 1,
        #         'coe3': 0
        #     }
        # },
        {
            'name': 'Sin',
            'args': None
        },
        {
            'name': 'Sin',
            'args': None
        },
        # {'name' : 'poly2', 'args' : {'coe1': 1, 'coe2': 1 , 'coe3': 1}},
        # {'name' : 'Sigmoid', 'args' : None}
        {
            'name': 'Sin',
            'args': None
        }  # activation for each layer, if with param, write as Binary_Zero here
    ]

    #  define origin model
    model = My_Model(layer_num=layer_num,
                     input_num=input_num,
                     weight_num_list=weight_num_list,
                     activation_list=activation_list,
                     tau_zero=tau_zero)

    # loop = 500
    # calculate two NTK_tilde
    NTK_tilde = calculate_NTK_tilde(model, tau_zero, X, T, K, p, means, covs, y,
                                  Omega)
    NTK_empirical = claculate_empirical_NTK(model, X, X)
    # CK_new = calculate_CK_tilde(new_model, tau_zero)

    # CK_loop = calculate_CK_loop(model, data_inference, loop=loop)

    error_norm = scipy.linalg.norm(NTK_tilde - NTK_empirical, ord=2)
    NTK_tilde_norm = scipy.linalg.norm(NTK_tilde, ord=2)
    NTK_loop_norm = scipy.linalg.norm(NTK_empirical, ord=2)
    print(error_norm)
    print(NTK_tilde_norm)
    print(NTK_loop_norm)

    # a = scipy.linalg.norm(CK_tilde, CK_new, ord=2)
    # print(a)
    # performance calculate

    # CK_loop = CK_tilde
    # CK_tilde = CK_loop

    setting = {
        'data': 'GMM-mixed',
        'T': str(T),
        'p': str(p),
        'layer_num': str(model.layer_num),
        'loop': 'none',
        'activation': 'rrr',
        'weight_num_list': str(weight_num_list)
    }
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir1 = save_path_small = os.path.join(root, 'fig\\small')
    if not os.path.isdir(dir1):
        os.makedirs(dir1)
    dir2 = save_path_small = os.path.join(root, 'hist_return\\small')
    if not os.path.isdir(dir2):
        os.makedirs(dir2)
    dir3 = save_path_small = os.path.join(root, 'plot_return')
    if not os.path.isdir(dir3):
        os.makedirs(dir3)
    # plot and save fig and data points
    save_path_small = os.path.join(
        root, 'fig\\small', ''.join(
            (setting['data'], '_', setting['T'], '_', setting['p'], '_',
             setting['loop'], '_', setting['activation'], '_',
             setting['weight_num_list'])))
    save_path_small_data = os.path.join(
        root, 'hist_return\\small', ''.join(
            (setting['data'], '_', setting['T'], '_', setting['p'], '_',
             setting['loop'], '_', setting['activation'], '_',
             setting['weight_num_list'])))
    save_path_vector_data = os.path.join(
        root, 'plot_return', ''.join(
            (setting['data'], '_', setting['T'], '_', setting['p'], '_',
             setting['loop'], '_', setting['activation'], '_',
             setting['weight_num_list'])))

    U_Phi_c, D_Phi_c, _ = np.linalg.svd(NTK_tilde)
    # tilde_U_Phi_c, tilde_D_Phi_c, _ = np.linalg.svd(M2)

    # plot eigenvalue distribution for two matrix and save
    plt.figure(1)
    # plt.subplot(211)
    xs = (min(D_Phi_c), max(D_Phi_c))
    n1, bins1, _, = plt.hist(D_Phi_c,
                             50,
                             facecolor='b',
                             alpha=0.5,
                             rwidth=0.5,
                             range=xs,
                             label='Eigenvalues of $\Phi_c$')

    eigen_value_data_hist = pd.DataFrame.from_dict({
        'bins1': bins1,
        'n1': np.append(n1, 0)
    })
    with open(save_path_small_data, 'w+') as f:
        eigen_value_data_hist.to_csv(f)
    plt.legend()
    plt.show()
    plt.savefig(save_path_small)
