# Lossless Compression(An NTK based method)

This repository contains code for the paper "*Lossless Compression of Deep Neural Networks: A High-dimensional Neural Tangent Kernel Approach*" submitted to NeurIPS 2022.

## 
## About the code

We packed the code into three different folders--compression, spectral_characteristics, utils:
-compression
 -we perform compression on two Dataset: MNIST, CIFAR10;
-spectral_characteristics
-utils















- *baselines.py* contains code for baseline transfer learning methods, including TCA[^1], JDA[^2], and CORAL[^3].
- *DaNN.py* contains code for DaNN[^4].
- *GFK.py* contains code for GFK[^5].
- *main.py* is the main file for the proposed RF-TCA approach.
- *-classifier* directory contains the different classifiers used in the paper.
- *-utils* directory contains functions to process data and files.
- Other files are functions used in main files *main.py*:
	- *TCA.py* contains code for the proposed R-TCA and RF-TCA; and 
	- *RandomF.py* contains code for random features.
- Some of them are forked from [Jindong Wang's transfer learning repository](https://github.com/jindongwang/transferlearning).

## Dependencies

To be able to test this code requires the following:

* [Python](https://www.python.org/): tested with version 3.6.13
* [Numpy](http://www.numpy.org/) and [Scipy](https://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/) for visulazation
* [Scikit-learn](http://scikit-learn.org/stable/) for kernel matrix
* [Pytorch](https://pytorch.org/): tested with version 1.10.0
* [Pandas](https://pandas.pydata.org/) for data record

## Reference


[^1]: Pan S J, Tsang I W, Kwok J T, et al. [Domain adaptation via transfer component analysis](https://ieeexplore.ieee.org/abstract/document/5640675)[J]. IEEE transactions on neural networks, 2010, 22(2): 199-210.
