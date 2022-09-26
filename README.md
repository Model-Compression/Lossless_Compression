# NTK-LC

This repository contains code to reproduces the results in the paper "* 'Lossless' Compression of Deep Neural Networks: A High-dimensional Neural Tangent Kernel Approach*" (**NTK-LC**) [^1].

## About the code

* `code/compression` contains 
  * `code/compression/mnist` **(Experiment 2.1)**
    * **mnist_origin.py** for classification with original dense neural network on MNIST dataset.
    * **performance_match.py** for classification with compressed neural network utlizing the proposed NTK*LC approach on MNIST dataset.
    * **mnist_magnitude_pruning.py** for classification with compressed neural network by magnitude pruning.
  * `code/compression/cifar10` **(Experiment 2.2)**
    * **vgg_net_cifar10.py** for defining VGG19 suitable for CIFAR10
    * **vgg_train.py** for training VGG19 defined in vgg_net_cifar10.py on CIFAR10, and get parameters of convolution layers for feature extraction.
    * **performance_origin.py** for classification performance of the original dense neural network on CIFAR10.
    * **performance_two.py** for classification performance of the compressed neural network (compressed by our NTK-LC algorithm) on CIFAR10.
    * **cifar10_magnitude_pruning.py** for classification with compressed neural network by magnitude pruning.

* `code/spectral_characteristics` **(Experiment 1)** contains
  * **tilde_CK.py** for verifying the consistency of spectrum distribution for **theoretical** (calculated with our theorem results) and **practical** (calculated by the original definition) conjugate kernel(CK).
  * **plot_eigen.py** for ploting eigenvalues and eigenvectors for a given matrix.

* `code/equation_solve` contains
  * **solve_equation.py** for solving equatiosn to define parameters of activation functions

* `code/expect_cal` contains
  * **expect_calculate.py** for expect calculated by numerical integration
  * **expect_calculate_math.py** for expect calculated with analytical expression
 
* `code/model_define` contains
  * **model.py** for model defining
 
* `code/utils` contains
  * **activation_numpy.py** for activations defined with numpy
  * **activation_tensor.py** for activations defined with torch
  * **data_prepare.py** for data preparation, containing data sampled from MNIST/CIFAR10 and generated GMM data
  * **utils.py** for some more utils 

## Dependencies

You can run following bash command to install packages used in this repository
```bash
pip install requirments.txt
```

or you can install follwing basic packages yourself:

* [Python](https://www.python.org/): tested with version 3.8.13
* [Numpy](http://www.numpy.org/) and [Scipy](https://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/) for visulazation
* [Pytorch](https://pytorch.org/): tested with version 1.12.0
* [Pandas](https://pandas.pydata.org/) for data record


## Contact information
* Zhenyu LIAO
  * Assistant Professor at EIC, Huazhong University of Science and Tech
  * Website: [https://zhenyu-liao.github.io/](https://zhenyu-liao.github.io/)
  * E-mail: [zhenyu_liao@hust.edu.cn](mailto:zhenyu_liao@hust.edu.cn)

* Linyu Gu
  * Master at EIC, Huazhong University of Science and Tech
  * E-mail: [gulingyu@hust.edu.cn](mailto:m202172384@hust.edu.cn)

* Yongqi Du
  * Master at EIC, Huazhong University of Science and Tech
  * E-mail: [yongqi_du@hust.edu.cn](mailto:yongqi_du@hust.edu.cn)



## References

[^1]: Gu L, Du Y, Zhang Y, et al. Lossless Compression of Deep Neural Networks: A High-dimensional Neural Tangent Kernel Approach[J].[link](https://zhenyu-liao.github.io/pdf/conf/RMT4DeepCompress_nips22.pdf)

[^2]: Ali H T, Liao Z, Couillet R. Random matrices in service of ML footprint: ternary random features with no performance loss[J]. arXiv preprint arXiv:2110.01899, 2021.[link](https://arxiv.org/abs/2110.01899)
