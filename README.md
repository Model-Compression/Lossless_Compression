# Lossless Compression(An NTK based method)

This repository contains code for the paper "*Lossless Compression of Deep Neural Networks: A High-dimensional Neural Tangent Kernel Approach*"[^1] submitted to NeurIPS 2022.

## 
## About the code

We packed the code into three different folders--compression, spectral_characteristics, utils:
- **compression** : We perform compression on two Dataset: MNIST, CIFAR10(experiment 2)
  - **mnist**
    - mnist_origin.py (Classification performance of the original neural network on MNIST)
    - performance_match.py (Classification performance of the compressed neural network compressed by our NTK-LC on MNIST)
  - **cifar10**
    - vgg_net_cifar10.py (Define VGG19 used for CIFAR10)
    - vgg_train.py (Train VGG19 defined in vgg_net_cifar10.py on CIFAR10, and get parameters for convolution layers, which will be used for feature extraction)
    - performance_origin.py (Classification performance of the original neural network on CIFAR10)
    - performance_two.py (Classification performance of the compressed neural network compressed by our NTK-LC on CIFAR10)
- **spectral_characteristics** : (experiment 1)
  - tilde_CK.py (Verify the consistency of theoretical and practical conjugate kernel(CK) spectrum distribution)
  - plot_eigen.py (Plot eigenvalues and eigen vectors)
- **utils** : some utils
  - activation_numpy.py (Activations define)
  - activation_tensor.py (Activations define)
  - data_prepare.py (Data prepare, including data sampled from MNIST/CIFAR10 and generated GMM data)
  - expect_calculate.py (expect calculated by numerical integration)
  - expect_calculate_math.py (expect calculated with analytical expresion)
  - model.py (model define)
  - solve_equation.py (solve equation to define activation functions)
  - utils.py (some more utils)

## Dependencies

'''bash
To be able to test this code requires the following:

* [Python](https://www.python.org/): tested with version 3.6.13
* [Numpy](http://www.numpy.org/) and [Scipy](https://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/) for visulazation
* [Scikit-learn](http://scikit-learn.org/stable/) for kernel matrix
* [Pytorch](https://pytorch.org/): tested with version 1.10.0
* [Pandas](https://pandas.pydata.org/) for data record

## Reference


[^1]: 
