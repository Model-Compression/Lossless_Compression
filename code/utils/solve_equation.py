#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve equation system to match coefficients each layer.

Uing custom_activation_analysis in expect_calculate_math to calculate coefficient expressions with variables, 
match with coefficients calculated for origin network to determine variables values.
Note: To use this module, see solve_equation.
'''
__author__ = "Yongqi_Du"
__copyright__ = "Copyright 2021, Lossless compression"
__license__ = "GPL"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Development"  # status is one of "Prototype", "Development", or "Production"
__all__ = ['solve_equation']

import numpy as np
import scipy
import torch
from scipy import optimize

from data_prepare import my_dataset_custome
from expect_calculate import expect_calcu
from expect_calculate_math import (custome_activation_analysis,
                                   custome_activation_analysis_noparam)
from model import My_Model
from utils import calculate_CK_tilde_coef, estim_tau_tensor

device = "cuda:3" if torch.cuda.is_available() else "cpu"


def calculate_CK_match_coef_full(model, tau_zero):
    '''calculate five expect(or squared) each layer(alpha1, alpha2, alpha3, alpha4, tau).  
    Note: no variables, this function can be used only for a known network(without unknown variables for activation functions)

    Arguments:
        model -- model instance of class My_Model in model.py
        tau_zero -- tau_zero calculated with data
    Returns:
        d_full -- coefficients for all layers(a list)
    '''
    tao_last = tau_zero
    d_full = []
    for activation in model.activation_list:
        name = activation['name']
        args = activation['args']
        if args:
            if name in ['Binary_Zero', 'Binary_Last']:
                if name == 'Binary_Zero':
                    (
                        _,
                        first_order,
                        second_order,
                        square_second_order,
                        tau,
                    ) = custome_activation_analysis_noparam(name='binary_zero', **args)
                elif name == 'Binary_Last':
                    (
                        _,
                        first_order,
                        second_order,
                        square_second_order,
                        tau,
                    ) = custome_activation_analysis_noparam(name='binary_last', **args)

            else:
                (
                    _,
                    first_order,
                    second_order,
                    square_second_order,
                    tau,
                ) = expect_calcu(name, **args)
        else:
            (
                _,
                first_order,
                second_order,
                square_second_order,
                tau,
            ) = expect_calcu(name)
        d1 = (first_order(tao_last))**2
        d2 = (second_order(tao_last))**2
        d3 = square_second_order(tao_last)
        tao_last = np.sqrt(tau(tao_last))
        d_last = np.array([tao_last, d1, d2, d3])
        d_full.append(d_last)
    print(d_full)
    return d_full


def solve_equation(model, tau_zero, loop):
    '''Here just a test for all binary activation(first two layer, binary_zero; last layer, binary_last).

    Arguments:
        model: origin model
        tau_zero: tau calculated with all data
        loop: loop times for solve equations(each loop given a random initial value)
    Returns:
        params_full: binary_zero+args  +  binary_last+args (which can be use as activation_list when configure My_Model)
    '''
    # calculate origin expects
    d_full = calculate_CK_match_coef_full(model, tau_zero)
    tau_last = tau_zero

    # for layers before last layer(we don't match tao for layers before last layer)
    (
        _,
        first_order,
        second_order,
        square_second_order,
        tau,
    ) = custome_activation_analysis('binary_zero')
    params_full = []
    for i in range(model.layer_num - 1):
        d1_origin, d2_origin, d3_origin = d_full[i][1], d_full[i][2], d_full[i][3]

        def d1(s1, s2, b1):
            return first_order(s1, s2, b1, tau_last)

        def d2(s1, s2, b1):
            return second_order(s1, s2, b1, tau_last)

        def d3(s1, s2, b1):
            return square_second_order(s1, s2, b1, tau_last)

        def func(x):
            return ((d1(x[0], x[1], x[2])**2 - d1_origin)**2 + (d2(x[0], x[1], x[2])**2 - d2_origin)**2 +
                    (d3(x[0], x[1], x[2]) - d3_origin)**2)

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
        print(res.fun, res.x)

        # update tau_last(calculating expects each layer needs tau for last layer, so we need update tau_last here)
        tau_last = np.sqrt(tau(res.x[0], res.x[1], res.x[2], tau_last))
        params_full.append({
            'name': 'Binary_Zero',
            'args': {
                's1': res.x[0],
                's2': res.x[1],
                'b1': res.x[2]
            },
        })

    # for last layer(we match tao for last layer)
    tau_origin, d1_origin, d2_origin, d3_origin = (
        d_full[i + 1][0],
        d_full[i + 1][1],
        d_full[i + 1][2],
        d_full[i + 1][3],
    )
    (
        _,
        first_order,
        second_order,
        square_second_order,
        tau,
    ) = custome_activation_analysis('binary_last')

    def t(s1, s2, s3, s4, b1, b2):
        return np.sqrt(tau(s1, s2, s3, s4, b1, b2, tau_last))

    def d1(s1, s2, s3, s4, b1, b2):
        return first_order(s1, s2, s3, s4, b1, b2, tau_last)

    def d2(s1, s2, s3, s4, b1, b2):
        return second_order(s1, s2, s3, s4, b1, b2, tau_last)

    def d3(s1, s2, s3, s4, b1, b2):
        return square_second_order(s1, s2, s3, s4, b1, b2, tau_last)

    def func(x):
        return ((t(x[0], x[1], x[2], x[3], x[4], x[5])**2 - tau_origin**2)**2 +
                (d1(x[0], x[1], x[2], x[3], x[4], x[5])**2 - d1_origin)**2 +
                (d2(x[0], x[1], x[2], x[3], x[4], x[5])**2 - d2_origin)**2 +
                (d3(x[0], x[1], x[2], x[3], x[4], x[5]) - d3_origin)**2)

    cons = ({
        'type': 'ineq',
        'fun': lambda x: x[1] - x[0]
    }, {
        'type': 'ineq',
        'fun': lambda x: x[2] - x[1]
    }, {
        'type': 'ineq',
        'fun': lambda x: x[3] - x[2]
    }, {
        'type': 'eq',
        'fun': lambda x: x[3] - x[2] - x[1] + x[0]
    })

    res = my_solve(
        func,
        var_num=6,
        loop=loop,
        ratios=[1, 1, 1, 1, 1, 1],
        method='SLSQP',
        constraints=cons,
        options={'ftol': 1e-30},
    )
    print(res.fun, res.x)
    params_full.append({
        'name': 'Binary_Last',
        'args': {
            's1': res.x[0],
            's2': res.x[1],
            's3': res.x[2],
            's4': res.x[3],
            'b1': res.x[4],
            'b2': res.x[5],
        },
    })

    return params_full


def my_solve(func, var_num, loop, ratios, **args):
    '''Solve equations for loop times, each time a random initial value, and find the best one.

    Arguments:
        func: equations system
        var_num: number of variable in func
        loop: loop times (each time a random initial value)
        ratios: random value scaling value
    Returns:
        res_final: value and solutions for equtions of mnimize target over loop times
    '''
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
            print(res_final.fun)
    return res_final


if __name__ == "__main__":

    cs = [0.5, 0.5]
    K = len(cs)
    # load data
    res = my_dataset_custome('MNIST', T_train=8000, T_test=1000, cs=cs, selected_target=[6, 8])
    dataset_train, dataset_test = res[0], res[1]
    # dataset = my_dataset_custome('iid',T=8000, p=300, cs=[0.4,0.6],selected_target=[1,2])
    'to be done here new net configuration here just binaryzero and binary last'
    # activation_new = ['binary_zero', 'binary_last']
    tau_zero = np.sqrt(estim_tau_tensor(dataset_train.X))
    print(tau_zero)
    # tau_zero = np.sqrt(estim_tau(dataset.data)), these two are equal!!!!!!!!

    # origin network setting
    layer_num = 3  # layer number for network
    input_num = 784  # input dimension for network 784/256
    weight_num_list = [3000, 1000, 1000]  # number for neurons for each layer
    activation_list = [
        {
            'name': 'LReLU',
            'args': {
                'coe1': 0.1,
                'coe2': 1
            }
        },
        {
            'name': 'LReLU',
            'args': {
                'coe1': 0.1,
                'coe2': 1
            }
        },
        {
            'name': 'LReLU',
            'args': {
                'coe1': 0.1,
                'coe2': 1
            }
        },
    ]

    #  define origin model
    model = My_Model(
        layer_num=layer_num,
        input_num=input_num,
        weight_num_list=weight_num_list,
        activation_list=activation_list,
        tau_zero=tau_zero,
    )

    res = solve_equation(model, tau_zero, 1000)
    activation_list = res
    new_model = My_Model(layer_num = layer_num, input_num = input_num, weight_num_list = weight_num_list, activation_list = activation_list, tau_zero=tau_zero)

    # CK_origin = calculate_CK_tilde_coef(model, tau_zero)

    # CK_new = calculate_CK_tilde_coef(new_model, tau_zero)
