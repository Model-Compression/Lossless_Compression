#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Select one activation with given name and args.

Activation name in [Binary_Zero, Binary_Last, T, Sign, Abs, LReLU, Posit, Poly2, Cos, Sin, Erf, Exp, Sigmoid], 
with all activation functions implemented with numpy for future quad.
Note: To use this module, see function my_activation_torch
'''
__author__ = "Yongqi_Du"
__copyright__ = "Copyright 2021, Lossless compression"
__license__ = "GPL"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Development"  # status is one of "Prototype", "Development", or "Production"
__all__ = ['my_activation_numpy']

import numpy as np
import scipy


def sigma_numpy(x, a):
    return np.heaviside(x - a, 0)


def Binary_Zero(s1, s2, b1):
    return lambda x: b1 * (sigma_numpy(x, s2) + sigma_numpy(-x, -s1))


def Binary_Last(s1, s2, s3, s4, b1, b2):
    return lambda x: b1 * (sigma_numpy(-x, -s1) + sigma_numpy(x, s4)) + b2 * (((sigma_numpy(-x, -s3) + sigma_numpy(x, s2)) - 1))


def my_activation_numpy(name, **args):
    '''Select one activation Module with given name and args.

    Arguments:
        name -- activation name [binary_zero, binary_last, Binary_Zero, Binary_Last, T, Sign, ABS, LReLU, POSIT, Poly2, Cos, Sin, ERF, EXP, Sigmoid]
        args -- args for activation function construction 
    Return: 
        activation functions
    Note:
        if the activation has parameters, pass kwargs(the key and the value), for example: activ = my_activation('binary_zero_nonparam', s1 = 1, s2 = 2, b1 = 1)
    '''
    if name == 'binary_zero':
        return lambda x, s1, s2, b1: b1 * (sigma_numpy(x, s2) + sigma_numpy(-x, -s1))
    elif name == 'binary_last':
        return lambda x, s1, s2, s3, s4, b1, b2, b3: b1 * (sigma_numpy(-x, -s1) + sigma_numpy(x, s4))  \
        + b2 * (((sigma_numpy(-x, -s3) + sigma_numpy(x, s2)) - 1))  + (b3-b1) * (sigma_numpy(-x, -s1))
    elif name == 'Binary_Zero':
        return Binary_Zero(**args)
    elif name == 'Binary_Last':
        return Binary_Last(**args)
    elif name == 'T':
        sig = lambda x: x
    elif name == 'ReLU':
        sig = lambda x: np.maximum(x, 0)
    elif name == 'Abs':
        sig = lambda x: np.abs(x)
    elif name == 'LReLU':
        sig = lambda x: args['coe1'] * np.maximum(x, 0) + args['coe2'] * np.maximum(-x, 0)
    elif name == 'Posit':
        sig = lambda x: (x > 0).astype(int)
    elif name == 'Sign':
        sig = lambda x: np.sign(x)
    elif name == 'Poly2':
        sig = lambda x: args['coe1'] * x**2 + args['coe2'] * x + args['coe3']
    elif name == 'Cos':
        sig = lambda x: np.cos(x)
    elif name == 'Sin':
        sig = lambda x: np.sin(x)
    elif name == 'Erf':
        sig = lambda x: scipy.special.erf(x)
    elif name == 'Exp':
        sig = lambda x: np.exp(-x**2 / 2)
    elif name == 'Sigmoid':
        sig = lambda x: 1 / (1 + np.exp(-x))
    elif name == 'Sign':
        sig = lambda x: np.sign(x)
    return sig

