#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect with it's expressions.

This module can be used for custom activation, and calculate these five expect in a methematical way.
(compared with expect_calculate in expect_calculate.py)
If you want to add a new custom activation functions, you should calculate the expression of these five expect respectively, and
then add the codes(expressions).
"""

__author__ = "Model_compression"
__copyright__ = "Copyright 2022, Lossless compression"
__credits__ = ["Yongqi Du"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Yongqi"
__email__ = "yongqi@hust.edu.cn"
__status__ = "Development"
__all__ = ['custome_activation_analysis',
           'custome_activation_analysis_noparam']
# status is one of "Prototype", "Development", or "Production"

import math
import sympy as sp
from sympy import oo


def custome_activation_analysis(name):
    '''Calculate zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect for a given activation name and args.

    Arguments:
        name -- activation name [binary_zero, binary_last]. You can add other custome activations here(remember to add codes).
        args -- args for activation function construction 
    Returns: 
        functions of zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect with variables,
        which can be used for future equations solve.
    Notes:
        if the activation has parameters, pass kwargs(the key and the value), for example: activ = my_activation('binary_zero_nonparam', s1 = 1, s2 = 2, b1 = 1)
    '''
    if name == 'binary_zero':
        zero_order = (lambda s1, s2, b1, tao_last: b1 / 2 *
                      (math.erf(s1 / (pow(2, 1 / 2) * tao_last)) - math.erf(s2 / (pow(2, 1 / 2) * tao_last))) + b1)

        first_order = (lambda s1, s2, b1, tao_last: b1 *
                       (math.exp(-pow(s2 / tao_last, 2) / 2) - math.exp(-pow(s1 / tao_last, 2) / 2)) /
                       (pow(2 * math.pi, 1 / 2) * tao_last))

        second_order = (lambda s1, s2, b1, tao_last: b1 *
                        (s2 * math.exp(-pow(s2 / tao_last, 2) / 2) - s1 * math.exp(-pow(s1 / tao_last, 2) / 2)) /
                        (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)))

        square_second_order = (lambda s1, s2, b1, tao_last: (pow(b1, 2) - 2 * b1 * zero_order(s1, s2, b1, tao_last)) *
                               (s2 * math.exp(-pow(s2 / tao_last, 2) / 2) - s1 * math.exp(-pow(s1 / tao_last, 2) / 2)) /
                               (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)))
        tau = (lambda s1, s2, b1, tao_last: (1 / 2 * pow(b1, 2)) *
               (math.erf(s1 / (pow(2, 1 / 2) * tao_last)) - math.erf(s2 / (pow(2, 1 / 2) * tao_last))) + pow(b1, 2) - pow(
                   zero_order(s1, s2, b1, tao_last), 2))
        # define sympy expressions
        # # wait to solve equation for
        # s1, s2, a1 = sp.symbols('s1,s2,a1', real=True)

        # x, y = sp.symbols('x,y', real=True)  # gaussian var

        # # some coe related to tao
        # a, b, c = sp.symbols('a,b,c', real=True, positive=True)

        # R = (-x + b/c y) * sp.e
        # tau_prime = sp.lambdify((s1, s2, b1, tao_prime_last),)

        # alpha_prime = sp.lambdify((s1, s2, b1, a, b, c), sp.integrate(, (x, -oo, oo)))

        # d_prime = sp.lambdify((s1, s2, b1, a, b, c), sp.integrate(, (x, -oo, oo)))

        return zero_order, first_order, second_order, square_second_order, tau
        # , tau_prime, alpha_prime, d_prime

    elif name == 'binary_last':
        zero_order = (lambda s1, s2, s3, s4, b1, b2, tao_last: (b1 / 2) *
                      (math.erf(s1 / (pow(2, 1 / 2) * tao_last)) - math.erf(s4 / (pow(2, 1 / 2) * tao_last))) + b1 + (b2 / 2) *
                      (math.erf(s3 / (pow(2, 1 / 2) * tao_last)) - math.erf(s2 / (pow(2, 1 / 2) * tao_last))))

        def first_order(s1, s2, s3, s4, b1, b2, tao_last): return b1 * (math.exp(-pow(s4 / tao_last, 2) / 2) - math.exp(-pow(
            s1 / tao_last, 2) / 2)) / (pow(2 * math.pi, 1 / 2) * tao_last) + b2 * (math.exp(-pow(
                s2 / tao_last, 2) / 2) - math.exp(-pow(s3 / tao_last, 2) / 2)) / (pow(2 * math.pi, 1 / 2) * tao_last)

        def second_order(s1, s2, s3, s4, b1, b2, tao_last): return b1 * (s4 * math.exp(-pow(s4 / tao_last, 2) / 2) - s1 * math.exp(
            -pow(s1 / tao_last, 2) / 2)) / (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)) + b2 * (s2 * math.exp(-pow(
                s2 / tao_last, 2) / 2) - s3 * math.exp(-pow(s3 / tao_last, 2) / 2)) / (pow(2 * math.pi, 1 / 2) * pow(
                    tao_last, 3))

        square_second_order = (lambda s1, s2, s3, s4, b1, b2, tao_last: b1**2 *
                               (s4 * math.exp(-pow(s4 / tao_last, 2) / 2) - s1 * math.exp(-pow(s1 / tao_last, 2) / 2)) /
                               (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)) + b2**2 *
                               (s2 * math.exp(-pow(s2 / tao_last, 2) / 2) - s3 * math.exp(-pow(s3 / tao_last, 2) / 2)) /
                               (pow(2 * math.pi, 1 / 2) * pow(tao_last, 3)) - 2 * zero_order(
                                   s1, s2, s3, s4, b1, b2, tao_last) * second_order(s1, s2, s3, s4, b1, b2, tao_last))

        tau = (lambda s1, s2, s3, s4, b1, b2, tao_last: (b1**2 / 2) *
               (math.erf(s1 / (pow(2, 1 / 2) * tao_last)) - math.erf(s4 / (pow(2, 1 / 2) * tao_last))) + b1**2 + (b2**2 / 2) *
               (math.erf(s3 / (pow(2, 1 / 2) * tao_last)) - math.erf(s2 / (pow(2, 1 / 2) * tao_last))) -
               (zero_order(s1, s2, s3, s4, b1, b2, tao_last))**2)
        return zero_order, first_order, second_order, square_second_order, tau


def custome_activation_analysis_noparam(name, **args):
    '''Calculate zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect for a given activation name and args.

    Arguments:
        name -- activation name [binary_zero, binary_last, ReLU, Sign]. You can add other custome activations here(remember to add codes).
        args -- args for activation function construction 
    Returns: 
        values of zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect.(no variables)
    Notes:
        if the activation has parameters, pass kwargs(the key and the value), for example: activ = my_activation('binary_zero_nonparam', s1 = 1, s2 = 2, b1 = 1)
    '''
    if name == 'ReLU':
        def zero_order(tao_last): return tao_last / math.sqrt(2 * math.pi)

        def first_order(tao_last): return 1 / 2

        def second_order(tao_last): return 1 / \
            (math.sqrt(2 * math.pi) * tao_last)

        def square_second_order(tao_last): return 1 - 1 / math.pi
        def tau(tao_last): return tao_last**2 * (1 / 2 - 1 / (2 * math.pi))

        return zero_order, first_order, second_order, square_second_order, tau

    elif name == 'Sign':
        def zero_order(tao_last): return 0

        def first_order(tao_last): return 2 / \
            (math.sqrt(2 * math.pi) * tao_last)

        def second_order(tao_last): return 0

        def square_second_order(tao_last): return 0
        def tau(tao_last): return 1

        return zero_order, first_order, second_order, square_second_order, tau

    elif name in ['binary_zero', 'binary_last']:
        (
            zero_order,
            first_order,
            second_order,
            square_second_order,
            tau,
        ) = custome_activation_analysis(name)

        def zero_order_noparam(tao_last): return zero_order(
            **args, tao_last=tao_last)

        def first_order_noparam(tao_last): return first_order(
            **args, tao_last=tao_last)

        def second_order_noparam(tao_last): return second_order(
            **args, tao_last=tao_last)
        def square_second_order_noparam(
            tao_last): return square_second_order(**args, tao_last=tao_last)

        def tau_noparam(tao_last): return tau(**args, tao_last=tao_last)
        return (
            zero_order_noparam,
            first_order_noparam,
            second_order_noparam,
            square_second_order_noparam,
            tau_noparam,
        )
