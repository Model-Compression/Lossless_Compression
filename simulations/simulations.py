# pylint: disable=invalid-name
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate data from GMM setting or sample data from datasets alredy exist and packed as torch.Dataset .(MNIST OR CIFAR10)

See my_dataset_custome for more detais.
"""

import json
import os

import numpy as np
import scipy
import scipy.linalg
import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset


def gen_data(testcase,
             selected_target=[6, 8],
             T=None,
             p=None,
             cs=None,
             means=None,
             covs=None,
             mode='train'):
    '''Generate GMM data from existing datasets or self sampling datasets.

    Arguments:
        testcase -- 'MNIST'/'CIFAR10'/'iid'/'means'/'vars'/'mixed'
        selected_traget -- list[xx, xx], only used for testcase=='MNIST'/'CIFAR10'
        T -- len of datasets
        p -- dimension of data, only used for testcase=='iid'/'means'/'vars'/'mixed'
        cs -- list[0.xx, 0.xx], ratio for diff classes, len(cs) is number of class of the dataset
        means -- matrix, means for diff classes, only used for testcase=='iid'/'means'/'vars'/'mixed'
        covs -- matrix, covs for diff classes, only used for testcase=='iid'/'means'/'vars'/'mixed'
        mode -- 'train'/'test', generate data for train and test
    Returns:
        X -- data
        Omega -- data - means
        y -- targets
        means -- means
        covs -- covs
        K -- number of class
        p -- dimension of data
        T -- number of data

    '''
    rng = np.random

    if testcase == 'MNIST':
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'data')
        if not os.path.isdir(root):
            os.makedirs(root)
        if mode == 'train':
            mnist = dset.MNIST(root=os.path.join(root, 'train'),
                               train=True,
                               download=True)
        else:
            mnist = dset.MNIST(root=os.path.join(root, 'test'),
                               train=False,
                               download=True)
        data, labels = mnist.data.view(mnist.data.shape[0], -1), mnist.targets

        # feel free to choose the number you like :)
        selected_target = selected_target
        p = 784
        K = len(selected_target)

        # get the whole set of selected number
        data_full = []
        data_full_matrix = np.array([]).reshape(p, 0)
        ind = 0
        for i in selected_target:
            locate_target_train = np.where(labels == i)[0]
            data_full.append(data[locate_target_train].T)
            data_full_matrix = np.concatenate(
                (data_full_matrix, data[locate_target_train].T), axis=1)
            ind += 1

        # recentering and normalization to satisfy Assumption 1 and
        T_full = data_full_matrix.shape[1]
        mean_selected_data = np.mean(data_full_matrix, axis=1).reshape(p, 1)
        norm2_selected_data = np.sum(
            (data_full_matrix -
             np.mean(data_full_matrix, axis=1).reshape(p, 1))**2,
            (0, 1)) / T_full
        for i in range(K):
            data_full[i] = data_full[i] - mean_selected_data
            data_full[i] = data_full[i] * np.sqrt(p) / np.sqrt(
                norm2_selected_data)

        # get the statistics of MNIST data
        means = []
        covs = []
        for i in range(K):
            data_tmp = data_full[i]
            T_tmp = data_tmp.shape[1]
            means.append(np.mean(data_tmp.numpy(), axis=1).reshape(p, 1))
            covs.append((data_tmp @ (data_tmp.T) / T_tmp -
                         means[i] @ (means[i].T)).reshape(p, p))

        # data for train

        X = np.array([]).reshape(p, 0)
        Omega = np.array([]).reshape(p, 0)
        y = []

        ind = 0
        for i in range(K):
            data_tmp = data_full[i]
            X = np.concatenate((X, data_tmp[:, range(int(cs[ind] * T))]),
                               axis=1)
            Omega = np.concatenate(
                (Omega, data_tmp[:, range(int(cs[ind] * T))] -
                 np.outer(means[ind], np.ones((1, int(T * cs[ind]))))),
                axis=1)
            y = np.concatenate((y, ind * np.ones(int(T * cs[ind]))))
            ind += 1

        X = X / np.sqrt(p)
        Omega = Omega / np.sqrt(p)

    elif testcase == 'CIFAR10':
        root = '../data'
        if not os.path.isdir(root):
            os.makedirs(root)
        if mode == 'train':
            cifar = dset.CIFAR10(root=os.path.join(root, 'train'),
                                 train=True,
                                 download=True)
        else:
            cifar = dset.CIFAR10(root=os.path.join(root, 'test'),
                                 train=False,
                                 download=True)
        data = cifar.data  # numpy
        targets = np.array(cifar.targets)  # numpy
        data, labels = data.reshape(data.shape[0], -1), targets

        # feel free to choose the number you like :)
        selected_target = selected_target
        p = 3072
        K = len(selected_target)

        # get the whole set of selected number
        data_full = []
        data_full_matrix = np.array([]).reshape(p, 0)
        ind = 0
        # print(np.where(labels==6))
        for i in selected_target:
            locate_target_train = np.where(labels == i)[0]
            data_full.append(data[locate_target_train].T)
            data_full_matrix = np.concatenate(
                (data_full_matrix, data[locate_target_train].T), axis=1)
            ind += 1

        # recentering and normalization to satisfy Assumption 1 and
        # for full datasets
        T_full = data_full_matrix.shape[1]
        mean_selected_data = np.mean(data_full_matrix, axis=1).reshape(p, 1)
        norm2_selected_data = np.sum(
            (data_full_matrix -
             np.mean(data_full_matrix, axis=1).reshape(p, 1))**2,
            (0, 1)) / T_full
        for i in range(K):
            data_full[i] = data_full[i] - mean_selected_data
            data_full[i] = data_full[i] * np.sqrt(p) / np.sqrt(
                norm2_selected_data)

        # get the statistics of CIFAR data
        # for each class
        means = []
        covs = []
        for i in range(K):
            data_tmp = data_full[i]
            T_tmp = data_tmp.shape[1]
            means.append(np.mean(data_tmp, axis=1).reshape(p, 1))
            covs.append((data_tmp @ (data_tmp.T) / T_tmp -
                         means[i] @ (means[i].T)).reshape(p, p))

        # data for train
        X = np.array([]).reshape(p, 0)
        Omega = np.array([]).reshape(p, 0)
        y = []

        # for each class , sample cs[class]*T samples, their Statistical Features from last part
        # why this part, last part is enough??
        ind = 0
        for i in range(K):
            data_tmp = data_full[i]
            X = np.concatenate((X, data_tmp[:, range(int(cs[ind] * T))]),
                               axis=1)
            Omega = np.concatenate(
                (Omega, data_tmp[:, range(int(cs[ind] * T))] -
                 np.outer(means[ind], np.ones((1, int(T * cs[ind]))))),
                axis=1)
            y = np.concatenate((y, ind * np.ones(int(T * cs[ind]))))
            ind += 1

        X = X / np.sqrt(p)
        Omega = Omega / np.sqrt(p)

    else:
        # root = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        #     'data/self_define')
        # if not os.path.isdir(root):
        #     os.makedirs(root)

        # data_path = os.path.join(
        #     root, ''.join(
        #         (testcase, '_', str(T), '_', str(p), '_', str(cs), '_', mode)))
        # if os.path.exists(data_path):
        #     with open(data_path, 'r') as f:
        #         data = json.load(f)
        #     X, Omega, y, means, covs, K, p, T = np.array(data['X']), np.array(data['Omega']), np.array(data['y']), \
        #                                             np.array(data['means']), np.array(data['covs']), data['K'],data['p'], data['T']
        # else:
          X = np.array([]).reshape(p, 0)
          Omega = np.array([]).reshape(p, 0)
          y = []

          K = len(cs)
          for i in range(K):
              tmp = rng.multivariate_normal(means[i],covs[i],size=int(T * cs[i])).T
              X = np.concatenate((X, tmp), axis=1)
              Omega = np.concatenate(
                  (Omega,
                    tmp - np.outer(means[i], np.ones((1, int(T * cs[i]))))),
                  axis=1)
              y = np.concatenate((y, i * np.ones(int(T * cs[i]))))

          X = X / np.sqrt(p)
          Omega = Omega / np.sqrt(p)

          data_save = {
              'X': X.tolist(),
              'Omega': Omega.tolist(),
              'y': y.tolist(),
              'means': means.tolist(),
              'covs': covs.tolist(),
              'K': K,
              'p': p,
              'T': T
          }
          # with open(data_path, 'w') as f:
          #     json.dump(data_save, f)

    return X, Omega, y, means, covs, K, p, T

    multivariate_normal