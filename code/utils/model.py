#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Custome module with config, you can config with configuration file.

See class My_Model for more details.
'''
__author__ = "Yongqi_Du"
__copyright__ = "Copyright 2021, Lossless compression"
__license__ = "GPL"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Development"  # status is one of "Prototype", "Development", or "Production"
__all__ = ['My_Model']

import numpy as np
import torch
import torch.nn as nn

from activation_tensor import my_activation_torch
from utils.expect_calculate import expect_calcu
from utils.expect_calculate_math import custome_activation_analysis_noparam

device = "cuda:3" if torch.cuda.is_available() else "cpu"


class My_Model(nn.Module):

    def __init__(self, layer_num, input_num, weight_num_list, activation_list, tau_zero) -> None:
        """Custome module with config, you can config with configuration file.

        Arguments:
            layer_num -- number of layer(depth of DNN)
            input_num -- data dimension(input layer neuron number)
            weight_num_list -- number of neurons in each layer(without input layer)
            activation_list -- list of activation functions
            tau_zero -- tau eatimated from input data
        """
        super().__init__()
        self.layer_num = layer_num
        self.activation_list = activation_list
        self.weight_num_list = weight_num_list
        self.activation_list = activation_list
        self.fc_layers = []
        self.act_layers = []
        self.tau_zero = tau_zero

        bias = []
        tau_last = self.tau_zero
        for i in range(layer_num):
            # fc_layer
            if i == 0:
                self.fc_layers.append(nn.Linear(input_num, weight_num_list[i], bias=False))
            else:
                self.fc_layers.append(nn.Linear(weight_num_list[i - 1], weight_num_list[i], bias=False))

            # activation_layer
            if activation_list[i]['args']:
                activ = my_activation_torch(activation_list[i]['name'], **activation_list[i]['args'])
                self.act_layers.append(activ)
            else:
                activ = my_activation_torch(activation_list[i]['name'])
                self.act_layers.append(activ)
            name_activ = self.activation_list[i]['name']
            args_activ = self.activation_list[i]['args']

            # recursive(calculate zero_order, "centering"
            # what we have calculated expressions use custome_activation_analysis_noparam, other simple activation functions use expect_calculate, which use quad functions)
            if args_activ:
                if name_activ == 'Binary_Zero':
                    zero_order, _, _, _, tau_square = custome_activation_analysis_noparam('binary_zero', **args_activ)
                elif name_activ == 'Binary_Last':
                    zero_order, _, _, _, tau_square = custome_activation_analysis_noparam('binary_last', **args_activ)
                else:
                    zero_order, _, _, _, tau_square = expect_calcu(name_activ, **args_activ)
            else:
                if name_activ in ['ReLU', 'Sign']:
                    zero_order, _, _, _, tau_square = custome_activation_analysis_noparam(name_activ)
                else:
                    zero_order, _, _, _, tau_square = expect_calcu(name_activ)
            d0_last = zero_order(tau_last)
            bias.append(d0_last)
            tau_last = np.sqrt(tau_square(tau_last))
        self.bias = torch.tensor(bias)

    def forward(self, X):
        X = X.float().to(device)
        for i in range(len(self.fc_layers)):
            # fc_layer
            self.fc_layers[i].to(device)
            X = self.fc_layers[i](X)
            # activation_layer
            self.act_layers[i].to(device)
            X = self.act_layers[i](X)
            # centering(minus zero_order)
            self.bias.to(device)
            X = X - self.bias[i]
            # normalizes
            X = 1 / torch.sqrt(torch.tensor(self.weight_num_list[i])) * X
        return X


if __name__ == "__main__":
    tau_zero = 1
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
