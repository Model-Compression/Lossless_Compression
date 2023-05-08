#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect with quad function.

This module can be used for simple activation, complex activation is recommanded to calculate these five expect in a methematical way.
Notation: all function using numpy because scipy.integrate.quad, which is different from activation.py where all function using torch.
"""

__author__ = "Model_compression"
__copyright__ = "Copyright 2022, Lossless compression"
__credits__ = [
    "Yongqi Du"
]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Yongqi Du"
__email__ = "yongqi_du@hust.edu.cn"
__status__ = "Development"
# status is one of "Prototype", "Development", or "Production"
__all__ = ['expect_calcu']
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
from scipy import stats
from scipy.integrate import quad

from utils.activation_numpy import my_activation_numpy


def first_order(name, **args):
    """Return first_order of activation function.

    Arguments:
        name -- activation name

    Returns:
        first_order of activation function
    """
    if name == 'binary_zero':
        return lambda x, s1, s2, b1, tao_last: x * my_activation_numpy(
            name, **args)(tao_last * x, s1, s2, b1) * stats.norm.pdf(
                x) / tao_last

    elif name == 'binary_last':
        return lambda x, s1, s2, s3, s4, b1, b2, b3, tao_last: x * my_activation_numpy(
            name, **args)(tao_last * x, s1, s2, s3, s4, b1, b2, b3
                          ) * stats.norm.pdf(x) / tao_last

    elif name == 'binary_last_final':
        return lambda x, s1, s2, b1, b2, b3, tao_last: x * my_activation_numpy(
            name, **args)(tao_last * x, s1, s2, b1, b2, b3) * stats.norm.pdf(
                x) / tao_last

    elif name == 'binary_last_four':
        return lambda x, s1, s2, s3, s4, b1, b2, tao_last: x * my_activation_numpy(
            name, **args)(tao_last * x, s1, s2, s3, s4, b1, b2
                          ) * stats.norm.pdf(x) / tao_last

    else:
        return lambda x, tao_last: x * my_activation_numpy(name, **args)(
            tao_last * x) * stats.norm.pdf(x) / tao_last


def second_order(name, **args):
    """Return second_order of activation function.

    Arguments:
        name -- activation name

    Returns:
        second_order of activation function
    """

    if name == 'binary_zero':
        return lambda x, s1, s2, b1, tao_last: (pow(
            x, 2) - 1) * my_activation_numpy(name, **args)(
                tao_last * x, s1, s2, b1) * stats.norm.pdf(x) / pow(
                    tao_last, 2)

    # notation:!!!!!!s1 s2 location zero

    elif name == 'binary_last':
        return lambda x, s1, s2, s3, s4, b1, b2, b3, tao_last: (pow(
            x, 2) - 1) * my_activation_numpy(name, **args)(
                tao_last * x, s1, s2, s3, s4, b1, b2, b3) * stats.norm.pdf(
                    x) / pow(tao_last, 2)

    elif name == 'binary_last_final':
        return lambda x, s1, s2, b1, b2, b3, tao_last: (pow(
            x, 2) - 1) * my_activation_numpy(name, **args)(
                tao_last * x, s1, s2, b1, b2, b3) * stats.norm.pdf(x) / pow(
                    tao_last, 2)

    elif name == 'binary_last_four':
        return lambda x, s1, s2, s3, s4, b1, b2, tao_last: (pow(
            x, 2) - 1) * my_activation_numpy(name, **args)(
                tao_last * x, s1, s2, s3, s4, b1, b2) * stats.norm.pdf(
                    x) / pow(tao_last, 2)

    else:
        return lambda x, tao_last: (pow(x, 2) - 1) * my_activation_numpy(
            name, **args)(tao_last * x) * stats.norm.pdf(x) / pow(tao_last, 2)


def square_second_order(name: str, d0, **args):
    """Return square_second_order of activation function.

    Arguments:
        name -- activation name

    Returns:
        square_second_order of activation function(the quad)
    """
    if name == 'binary_zero':
        return lambda x, s1, s2, b1, tao_last : (pow(x, 2) - 1) * pow(my_activation_numpy(name, **args)(tao_last * x, s1, s2, b1) - d0, 2)\
              * stats.norm.pdf(x)/ pow(tao_last, 2)
    # notation:!!!!!!s1 s2 location zero

    elif name == 'binary_last':
        return lambda x, s1, s2, s3, s4, b1, b2, b3, tao_last : (pow(x, 2) - 1)  * pow(my_activation_numpy(name, **args)(tao_last *x, s1, s2, s3, s4, b1, b2, b3) - d0 , 2)\
             * stats.norm.pdf(x)/ pow(tao_last, 2)

    elif name == 'binary_last_final':
        return lambda x, s1, s2, b1, b2, b3, tao_last : (pow(x, 2) - 1)  * pow(my_activation_numpy(name, **args)(tao_last *x, s1, s2, b1, b2, b3) - d0 , 2)\
             * stats.norm.pdf(x)/ pow(tao_last, 2)

    elif name == 'binary_last_four':
        return lambda x, s1, s2, s3, s4, b1, b2, tao_last : (pow(x, 2) - 1)  * pow(my_activation_numpy(name, **args)(tao_last *x, s1, s2, s3, s4, b1, b2) - d0 , 2)\
             * stats.norm.pdf(x)/ pow(tao_last, 2)

    else:
        return lambda x, tao_last : (pow(x, 2) - 1)  * pow(my_activation_numpy(name, **args)(tao_last *x ) - d0 , 2)\
             * stats.norm.pdf(x)/ pow(tao_last, 2)


def zero_order(name, **args):
    """Return zero_order of activation function.

    Arguments:
        name -- activation name

    Returns:
        zero_order of activation function(the quad)
    """

    if name == 'binary_zero':
        return lambda x, s1, s2, b1, tao_last: my_activation_numpy(
            name, **args)(tao_last * x, s1, s2, b1) * stats.norm.pdf(x)

    elif name == 'binary_last':
        return lambda x, s1, s2, s3, s4, b1, b2, b3, tao_last: my_activation_numpy(
            name, **args)(tao_last * x, s1, s2, s3, s4, b1, b2, b3
                          ) * stats.norm.pdf(x)

    elif name == 'binary_last_final':
        return lambda x, s1, s2, b1, b2, b3, tao_last: my_activation_numpy(
            name, **args)(tao_last * x, s1, s2, b1, b2, b3) * stats.norm.pdf(x)

    elif name == 'binary_last_four':
        return lambda x, s1, s2, s3, s4, b1, b2, tao_last: my_activation_numpy(
            name, **args)(tao_last * x, s1, s2, s3, s4, b1, b2
                          ) * stats.norm.pdf(x)

    else:
        return lambda x, tao_last: my_activation_numpy(name, **args)(
            tao_last * x) * stats.norm.pdf(x)


def tao(name, d0, **args):
    """Return tao of activation function.

    Arguments:
        name -- activation name

    Returns:
        tao of activation function(the quad)
    """
    if name == 'binary_zero':
        return lambda x, s1, s2, b1, tao_last : pow(my_activation_numpy(name, **args)(tao_last * x, s1, s2, b1) - d0, 2)\
             * stats.norm.pdf(x)

    elif name == 'binary_last':
        return lambda x, s1, s2, s3, s4, b1, b2, b3, tao_last : pow(my_activation_numpy(name, **args)(tao_last * x, s1, s2, s3, s4, b1, b2, b3)- d0, 2)\
             * stats.norm.pdf(x)

    elif name == 'binary_last_final':
        return lambda x, s1, s2, b1, b2, b3, tao_last : pow(my_activation_numpy(name, **args)(tao_last * x, s1, s2, b1, b2, b3)- d0, 2)\
             * stats.norm.pdf(x)

    elif name == 'binary_last_four':
        return lambda x, s1, s2, s3, s4, b1, b2, tao_last : pow(my_activation_numpy(name, **args)(tao_last * x, s1, s2, s3, s4, b1, b2)- d0, 2)\
             * stats.norm.pdf(x)

    else:
        return lambda x, tao_last : pow(my_activation_numpy(name, **args)(tao_last * x)- d0, 2)\
             * stats.norm.pdf(x)


def expect_calcu(name, **args):
    '''Calculate zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect for a given activation name and args.

    Arguments:
        name -- activation name [binary_zero, binary_last, Binary_Zero, Binary_Last, T, Sign, ABS, LReLU, POSIT, Poly2, Cos, Sin, ERF, EXP, Sigmoid]
        args -- args for activation function construction 
    Returns: 
        zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect(the quad result)
    Notes:
        if the activation has parameters, pass kwargs(the key and the value), for example: activ = my_activation('binary_zero_nonparam', s1 = 1, s2 = 2, b1 = 1)
    '''
    if name == 'binary_zero':
        zero_order_expect = lambda s1, s2, b1, tao_last: quad(
            zero_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, tao_last))[0]
        tao_expect = lambda s1, s2, b1, tao_last: quad(
            tao(name, zero_order_expect(s1, s2, b1, tao_last), **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, tao_last))[0]
        first_order_expect = lambda s1, s2, b1, tao_last: quad(
            first_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, tao_last))[0]
        second_order_expect = lambda s1, s2, b1, tao_last: quad(
            second_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, tao_last))[0]
        square_second_order_expect = lambda s1, s2, b1, tao_last: quad(
            square_second_order(name, zero_order_expect(s1, s2, b1, tao_last),
                                **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, tao_last),
            epsrel=1.49e-10)[0]

    elif name == 'binary_last':
        zero_order_expect = lambda s1, s2, s3, s4, b1, b2, b3, tao_last: quad(
            zero_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, b3, tao_last))[0]
        tao_expect = lambda s1, s2, s3, s4, b1, b2, b3, tao_last: quad(
            tao(name, zero_order_expect(s1, s2, s3, s4, b1, b2, b3, tao_last),
                **args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, b3, tao_last))[0]
        first_order_expect = lambda s1, s2, s3, s4, b1, b2, b3, tao_last: quad(
            first_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, b3, tao_last))[0]
        second_order_expect = lambda s1, s2, s3, s4, b1, b2, b3, tao_last: quad(
            second_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, b3, tao_last))[0]
        square_second_order_expect = lambda s1, s2, s3, s4, b1, b2, b3, tao_last: quad(
            square_second_order(
                name, zero_order_expect(s1, s2, s3, s4, b1, b2, b3, tao_last),
                **args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, b3, tao_last),
            epsrel=1.49e-10)[0]

    elif name == 'binary_last_final':
        zero_order_expect = lambda s1, s2, b1, b2, b3, tao_last: quad(
            zero_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, b2, b3, tao_last))[0]
        tao_expect = lambda s1, s2, b1, b2, b3, tao_last: quad(
            tao(name, zero_order_expect(s1, s2, b1, b2, b3, tao_last), **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, b2, b3, tao_last))[0]
        first_order_expect = lambda s1, s2, b1, b2, b3, tao_last: quad(
            first_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, b2, b3, tao_last))[0]
        second_order_expect = lambda s1, s2, b1, b2, b3, tao_last: quad(
            second_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, b2, b3, tao_last))[0]
        square_second_order_expect = lambda s1, s2, b1, b2, b3, tao_last: quad(
            square_second_order(
                name, zero_order_expect(s1, s2, b1, b2, b3, tao_last), **args),
            -np.inf,
            np.inf,
            args=(s1, s2, b1, b2, b3, tao_last),
            epsrel=1.49e-10)[0]

    elif name == 'binary_last_four':
        zero_order_expect = lambda s1, s2, s3, s4, b1, b2, tao_last: quad(
            zero_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, tao_last))[0]
        tao_expect = lambda s1, s2, s3, s4, b1, b2, tao_last: quad(
            tao(name, zero_order_expect(s1, s2, s3, s4, b1, b2, tao_last), **
                args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, tao_last))[0]
        first_order_expect = lambda s1, s2, s3, s4, b1, b2, tao_last: quad(
            first_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, tao_last))[0]
        second_order_expect = lambda s1, s2, s3, s4, b1, b2, tao_last: quad(
            second_order(name, **args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, tao_last))[0]
        square_second_order_expect = lambda s1, s2, s3, s4, b1, b2, tao_last: quad(
            square_second_order(
                name, zero_order_expect(s1, s2, s3, s4, b1, b2, tao_last), **
                args),
            -np.inf,
            np.inf,
            args=(s1, s2, s3, s4, b1, b2, tao_last),
            epsrel=1.49e-10)[0]
    else:
        zero_order_expect = lambda tao_last: quad(zero_order(name, **args),
                                                  -np.inf,
                                                  np.inf,
                                                  args=(tao_last),
                                                  limit=1000)[0]
        tao_expect = lambda tao_last: quad(tao(name, zero_order_expect(
            tao_last), **args),
                                           -np.inf,
                                           np.inf,
                                           args=(tao_last),
                                           limit=1000)[0]
        first_order_expect = lambda tao_last: quad(first_order(name, **args),
                                                   -np.inf,
                                                   np.inf,
                                                   args=(tao_last),
                                                   limit=1000)[0]
        second_order_expect = lambda tao_last: quad(second_order(name, **args),
                                                    -np.inf,
                                                    np.inf,
                                                    args=(tao_last),
                                                    limit=1000)[0]
        square_second_order_expect = lambda tao_last: quad(square_second_order(
            name, zero_order_expect(tao_last), **args),
                                                           -np.inf,
                                                           np.inf,
                                                           limit=1000,
                                                           args=(tao_last),
                                                           epsrel=1.49e-10)[0]

    return zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect


if __name__ == "__main__":
    # a simple implemention, you can change activation name and corresponding parameters at will.
    zero_order_expect, first_order_expect, second_order_expect, square_second_order_expect, tao_expect =\
        expect_calcu('binary_last_four')
    s1, s2, s3, s4, b1, b2, tao_last = 1, 2, 3, 4, 1, 2, 3

    print(zero_order_expect(s1, s2, s3, s4, b1, b2, tao_last),
          first_order_expect(s1, s2, s3, s4, b1, b2, tao_last),
          second_order_expect(s1, s2, s3, s4, b1, b2, tao_last),
          square_second_order_expect(s1, s2, s3, s4, b1, b2, tao_last),
          tao_expect(s1, s2, s3, s4, b1, b2, tao_last))
