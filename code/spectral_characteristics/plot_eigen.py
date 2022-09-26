'''Plot eigen value distribution for two matrix, 
and the eigen vector corresponding to the top eigen value.'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use('agg')
import os

import pandas as pd


def plot_eigen(M1, M2, setting):
    """plot and save igenvalue distributions and data points for matrix1 and matrix2.

    Arguments:
        M1 -- matrix1
        M2 -- matrix2
        setting -- some setting for name_generation
    """
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

    U_Phi_c, D_Phi_c, _ = np.linalg.svd(M1)
    tilde_U_Phi_c, tilde_D_Phi_c, _ = np.linalg.svd(M2)

    # plot eigenvalue distribution for two matrix and save
    plt.figure(1)
    plt.subplot(211)
    xs = (min(min(D_Phi_c),
              min(tilde_D_Phi_c)), max(max(tilde_D_Phi_c), max(tilde_D_Phi_c)))
    n1, bins1, _, = plt.hist(D_Phi_c,
                             50,
                             facecolor='b',
                             alpha=0.5,
                             rwidth=0.5,
                             range=xs,
                             label='Eigenvalues of $\Phi_c$')
    n2, bins2, _, = plt.hist(tilde_D_Phi_c,
                             50,
                             facecolor='r',
                             alpha=0.5,
                             rwidth=0.5,
                             range=xs,
                             label='Eigenvalues of $\~\Phi_c$')

    eigen_value_data_hist = pd.DataFrame.from_dict({
        'bins1': bins1,
        'n1': np.append(n1, 0),
        'bins2': bins2,
        'n2': np.append(n2, 0)
    })
    with open(save_path_small_data, 'w+') as f:
        eigen_value_data_hist.to_csv(f)
    plt.legend()

    # plot eigenvector corresponding to top eigen value and save
    plt.subplot(212)
    pl1, = plt.plot(U_Phi_c[:, 0],
                    'b',
                    label='Leading eigenvector of $\Phi_c$')
    pl2, = plt.plot(tilde_U_Phi_c[:, 0] *
                    np.sign(U_Phi_c[1, 0] * tilde_U_Phi_c[1, 0]),
                    'r--',
                    label='Leading eigenvector of $\~\Phi_c$')

    eigen_vector_data_hist = pd.DataFrame.from_dict({
        'pl1':
        U_Phi_c[:, 0],
        'pl2':
        tilde_U_Phi_c[:, 0] * np.sign(U_Phi_c[1, 0] * tilde_U_Phi_c[1, 0])
    })
    with open(save_path_vector_data, 'w+') as f:
        eigen_vector_data_hist.to_csv(f)

    plt.show()

    plt.savefig(save_path_small)
