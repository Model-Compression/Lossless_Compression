#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Select one activation Module with given name and args.

Activation name in [Binary_Zero, Binary_Last, T, Sign, Abs, LReLU, Posit, Poly2, Cos, Sin, Erf, Exp] or activation Module in torch.nn, 
with all activation functions packed as nn.Module.
Note: To use this module, see function my_activation_torch
'''
__author__ = "Yongqi_Du"
__copyright__ = "Copyright 2021, Lossless compression"
__license__ = "GPL"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Development"  # status is one of "Prototype", "Development", or "Production"
__all__ = ['my_activation_torch']

# Imports
import torch
import torch.nn as nn


# activation functions for tensor
def sigma_torch(x, a):
    '''
    define sigma function
    '''
    return torch.heaviside(x - a, torch.tensor(0.0))


def binary_zero(s1, s2, b1):
    return lambda x: b1 * (sigma_torch(x, s2) + sigma_torch(-x, -s1))


def binary_last(s1, s2, s3, s4, b1, b2):
    return lambda x: b1 * (sigma_torch(-x, -s1) + sigma_torch(x, s4)) + b2 * (((sigma_torch(-x, -s3) + sigma_torch(x, s2)) - 1))


def poly2(coe1, coe2, coe3):
    return lambda x: coe1 * x**2 + coe2 * x + coe3


def sign(x):
    return torch.sign(x)


def cos(x):
    return torch.cos(x)


def sin(x):
    return torch.sin(x)


def lrelu(coe1, coe2):
    return lambda x: coe1 * torch.maximum(x, torch.tensor(0)) + coe2 * torch.maximum(-x, torch.tensor(0))


def abs(x):
    return torch.abs(x)


def t(x):
    return x


def exp(x):
    return torch.exp(-x**2 / 2)


def posit(x):
    return (x > 0).long()


def erf(x):
    return torch.erf(x)


# pack above functions to class for better control when need train(pack as nn.function then nn.Module if need backpropogation).
class Binary_Zero(nn.Module):

    def __init__(self, s1: float, s2: float, b1: float) -> None:
        super().__init__()
        self.func = binary_zero(s1, s2, b1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Binary_Last(nn.Module):

    def __init__(self, s1: float, s2: float, s3: float, s4: float, b1: float, b2: float) -> None:
        super().__init__()
        self.func = binary_last(s1, s2, s3, s4, b1, b2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Poly2(nn.Module):

    def __init__(self, coe1: float, coe2: float, coe3: float) -> None:
        super().__init__()
        self.func = poly2(coe1, coe2, coe3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Sign(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.func = sign

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Cos(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.func = cos

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Sin(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.func = sin

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class LReLU(nn.Module):

    def __init__(self, coe1: float, coe2: float) -> None:
        super().__init__()
        self.func = lrelu(coe1, coe2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Abs(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.func = abs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class T(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.func = t

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Exp(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.func = exp

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Posit(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.func = posit

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


class Erf(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.func = erf

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)


def my_activation_torch(name, **args):
    '''Select one activation Module with given name and args.

    Arguments:
        name -- activation name [Binary_Zero, Binary_Last, T, Sign, Abs, LReLU, Posit, Poly2, Cos, Sin, Erf, Exp] or activation Module in torch.nn
        args -- args for activation function construction 
    Return: 
        activation functions(packed as nn.Module)
    Note:
        if the activation has parameters, pass kwargs(the key and the value), for example: activ = my_activation('binary_zero_nonparam', s1 = 1, s2 = 2, b1 = 1)
    '''
    if name == 'Binary_Zero':
        return Binary_Zero(**args)
    elif name == 'Binary_Last':
        return Binary_Last(**args)
    elif name == 'T':
        sig = T()
    elif name == 'Sign':
        sig = Sign()
    elif name == 'ABS':
        sig = Abs()
    elif name == 'LReLU':
        sig = LReLU(**args)
    elif name == 'POSIT':
        sig = Posit()
    elif name == 'Poly2':
        return Poly2(**args)
    elif name == 'Cos':
        sig = Cos()
    elif name == 'Sin':
        sig = Sin()
    elif name == 'ERF':
        sig = Erf()
    elif name == 'EXP':
        sig = Exp()
    else:
        sig = getattr(nn, name)(**args)
    return sig
