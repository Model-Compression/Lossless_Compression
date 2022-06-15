#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect with its expression.

This module can be used for custom activation, and calculate these five expect in a methematical way.
If you want to add a new custom activation functions, you should calculate the expression of these five expect respectively, and
then add with code.
"""

__author__ = "Model_compression"
__copyright__ = "Copyright 2021, Lossless compression"
__credits__ = ["Rob Knight", "Peter Maxwell", "Gavin Huttley", "Matthew Wakefield"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Development"
__all__ = ['calculate_CK_tilde_coef']
# status is one of "Prototype", "Development", or "Production"

import numpy as np

from expect_calculate import expect_calcu
from expect_calculate_math import custome_activation_analysis_noparam


def estim_tau_tensor(X):
    tau = np.mean(np.diag(X @ X.T))

    return tau


def estim_tau(X):
    tau = np.mean(np.diag(X.T @ X))

    return tau


def calculate_CK_tilde_coef(model, tau_zero):
    '''calculate coefficients each layer(alpha1, alpha2, alpha3, alpha4, tau) and print.
    Note: no variables, this function can be used only for a known network(without unknown variables for activation functions), here just for test

    Arguments:
        model -- model instance of class My_Model in model.py
        tau_zero -- tau_zero calculated with data
    '''
    tao_last = tau_zero
    d_last = np.array([tao_last, 1, 0, 0, 1])  # input d1, d2, d3, d4
    for activation in model.activation_list:
        name = activation['name']
        args = activation['args']
        if args:
            if name == 'Binary_Zero':
                (
                    zero_order,
                    first_order,
                    second_order,
                    square_second_order,
                    tau,
                ) = custome_activation_analysis_noparam('binary_zero', **args)
            elif name == 'Binary_Last':
                (
                    zero_order,
                    first_order,
                    second_order,
                    square_second_order,
                    tau,
                ) = custome_activation_analysis_noparam('binary_last', **args)
            else:
                (
                    zero_order,
                    first_order,
                    second_order,
                    square_second_order,
                    tau,
                ) = expect_calcu(name, **args)
        else:
            (
                zero_order,
                first_order,
                second_order,
                square_second_order,
                tau,
            ) = expect_calcu(name)
        temp = zero_order(tao_last)
        d1 = first_order(tao_last)**2 * d_last[1]
        d2 = (first_order(tao_last)**2 * d_last[2] + 1 / 4 * second_order(tao_last)**2 * d_last[4]**2)
        d3 = (first_order(tao_last)**2 * d_last[3] + 1 / 2 * second_order(tao_last)**2 * d_last[1]**2)
        d4 = 1 / 2 * square_second_order(tao_last) * d_last[4]
        tao_last = np.sqrt(tau(tao_last))
        d_last = np.array([tao_last, d1, d2, d3, d4])
        print(d1)
        print(d2)
        print(d3)
        print(d4)
        print(tao_last)
