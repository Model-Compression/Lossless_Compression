import torch
import torch.nn as nn
import numpy as np
from model import FCnet
import os
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import scipy
import torch.optim as optim
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy,math
from scipy import optimize

def get_or_param(L,act,tau_0):
    order_1 = L * [0]
    order_2 = L * [0]
    square = L * [0]
    mean = L * [0]
    tau = (L+1)*[0]
    tau[0] = tau_0
    if act == 'ReLU':
        for i in range(1,L+1):
            mean[i-1] = tau[i-1] / np.sqrt(2 * np.pi)
            tau[i] = np.sqrt((tau[i-1]**2)*(1 / 2 - 1 / (2 * np.pi)))
            order_1[i-1] = 1 / 2
            order_2[i-1] = np.sqrt(1 / (2 * np.pi * tau[i-1]**2))
            square[i-1] = 1 - 1 / np.pi
            # i += 1
    return order_1, order_2, square, tau, mean
    
def get_act0(tao_last):
    act0_mean = (lambda s1, s2, b1: b1 / 2 *
                (math.erf(s1 / (pow(2, 1 / 2) * tao_last)) - math.erf(s2 / (pow(2, 1 / 2) * tao_last))) + b1)

    act0_order_1 = (lambda s1, s2, b1: b1 *
                (math.exp(-pow(s2 / tao_last, 2) / 2) - math.exp(-pow(s1 / tao_last, 2) / 2)) /
                (pow(2 * math.pi, 1 / 2) * tao_last))

    act0_order_2 = (lambda s1, s2, b1: b1 *
                    (s2 * math.exp(-pow(s2 / tao_last, 2) / 2) - s1 * math.exp(-pow(s1 / tao_last, 2) / 2)) /
                    (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)))

    act0_square = (lambda s1, s2, b1: (pow(b1, 2) - 2 * b1 * act0_mean(s1, s2, b1)) *
                        (s2 * math.exp(-pow(s2 / tao_last, 2) / 2) - s1 * math.exp(-pow(s1 / tao_last, 2) / 2)) /
                        (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)))
    act0_tau = (lambda s1, s2, b1: (1 / 2 * pow(b1, 2)) *
        (math.erf(s1 / (pow(2, 1 / 2) * tao_last)) - math.erf(s2 / (pow(2, 1 / 2) * tao_last))) + pow(b1, 2) - pow(
            act0_mean(s1, s2, b1), 2))

    return act0_order_1, act0_order_2, act0_square, act0_tau, act0_mean

def get_act1(b1, b2, tao_last):
    act1_mean = (lambda s1, s2, s3, s4: (b1 / 2) *
                (math.erf(s1 / (pow(2, 1 / 2) * tao_last)) - math.erf(s4 / (pow(2, 1 / 2) * tao_last))) + b1 + (b2 / 2) *
                (math.erf(s3 / (pow(2, 1 / 2) * tao_last)) - math.erf(s2 / (pow(2, 1 / 2) * tao_last))))

    act1_order_1 = (lambda s1, s2, s3, s4: b1 * (math.exp(-pow(s4 / tao_last, 2) / 2) - math.exp(-pow(
                    s1 / tao_last, 2) / 2)) / (pow(2 * math.pi, 1 / 2) * tao_last) + b2 * (math.exp(-pow(
                    s2 / tao_last, 2) / 2) - math.exp(-pow(s3 / tao_last, 2) / 2)) / (pow(2 * math.pi, 1 / 2) * tao_last))

    act1_order_2 = (lambda s1, s2, s3, s4:  b1 * (s4 * math.exp(-pow(s4 / tao_last, 2) / 2) - s1 * math.exp(
                    -pow(s1 / tao_last, 2) / 2)) / (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)) + b2 * (s2 * math.exp(-pow(
                    s2 / tao_last, 2) / 2) - s3 * math.exp(-pow(s3 / tao_last, 2) / 2)) / (pow(2 * math.pi, 1 / 2) * pow(
                    tao_last, 3)))

    act1_square = (lambda s1, s2, s3, s4: b1**2 *
                            (s4 * math.exp(-pow(s4 / tao_last, 2) / 2) - s1 * math.exp(-pow(s1 / tao_last, 2) / 2)) /
                            (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)) + b2**2 *
                            (s2 * math.exp(-pow(s2 / tao_last, 2) / 2) - s3 * math.exp(-pow(s3 / tao_last, 2) / 2)) /
                            (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)) - 2 * act1_mean(
                                s1, s2, s3, s4) * act1_order_2(s1, s2, s3, s4))

    act1_tau = (lambda s1, s2, s3, s4: (b1**2 / 2) *
            (math.erf(s1 / (pow(2, 1 / 2) * tao_last)) - math.erf(s4 / (pow(2, 1 / 2) * tao_last))) + b1**2 + (b2**2 / 2) *
            (math.erf(s3 / (pow(2, 1 / 2) * tao_last)) - math.erf(s2 / (pow(2, 1 / 2) * tao_last))) -
            (act1_mean(s1, s2, s3, s4))**2)

    return act1_order_1, act1_order_2, act1_square, act1_tau, act1_mean

def my_solve(func, var_num, loop, ratios, **args):
    
    res_final = scipy.optimize.OptimizeResult(fun=100000, x=np.ones(var_num))
    for i in range(loop):
        # initial values
        X = np.array([])
        for ratio in ratios:
            X = np.concatenate((X, ratio * np.random.randn(1)))
        # to ensure s1<s2<s3
        if var_num > 3:
            X[1] = X[1] if X[1] > X[0] else 2 * X[0] - X[1]
            X[2] = X[2] if X[2] > X[1] else 2 * X[1] - X[2]
            X[3] = X[3] if X[3] > X[2] else 2 * X[2] - X[3]
        # optimization
        res = optimize.minimize(func, X, **args)
        # find the best one
        if res.fun < res_final.fun:
            res_final = res
            print('in loop')
            print(res_final.fun)
        i += 1
    return res_final

def sovle_0(e_or, e_new, loop):
    def func(x):
            return ((e_new[0](x[0], x[1], x[2])**2 - e_or[0]**2)**2 +
                    (e_new[1](x[0], x[1], x[2])**2 - e_or[1]**2)**2 +
                    (e_new[2](x[0], x[1], x[2]) - e_or[2])**2)

    cons = {'type': 'ineq', 'fun': lambda x: x[1] - x[0]}
    res = my_solve(
        func,
        var_num=3,
        loop=loop,
        ratios=[-0.5, 0.5, 0.5],
        method='SLSQP',
        constraints=cons,
        options={'ftol': 1e-30},
    )
    print('fomer-2 layer')
    print(res.fun, res.x)   

    return res.x

def sovle_1(e_or, e_new, loop):
    def func(x):
            return ((e_new[0](x[0], x[1], x[2], x[2] + x[1] - x[0])**2 - e_or[0]**2)**2+
                    (e_new[1](x[0], x[1], x[2], x[2] + x[1] - x[0])**2 - e_or[1]**2)**2 +
                    (e_new[2](x[0], x[1], x[2], x[2] + x[1] - x[0]) - e_or[2])**2 +
                    (np.sqrt(e_new[3](x[0], x[1], x[2], x[2] + x[1] - x[0])) - e_or[3])**2)

    cons = ({
        'type': 'ineq',
        'fun': lambda x: x[1] - x[0]
    }, {
        'type': 'ineq',
        'fun': lambda x: x[2] - x[1]
    },
    {
        'type': 'ineq',
        'fun': lambda x: x[2] - x[0]
    }, 
    )

    res = my_solve(
        func,
        var_num=3,
        loop=loop,
        ratios=[1, 1, 1],
        method='SLSQP',
        constraints=cons,
        options={'ftol': 1e-30},
    )
    print('last layer')
    print(res.fun, res.x)

    return [res.x[0], res.x[1], res.x[2], res.x[2] + res.x[1] - res.x[0]]