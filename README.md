# NTK-LC

This repository contains code to reproduces the results in the paper "*Lossless Compression of Deep Neural Networks: A High-dimensional Neural Tangent Kernel Approach*" (**NTK-LC**) [^1].

## About the code

* `code/compression` contains
  * `code/compression/mnist`
    * **mnist_origin.py** for classification using original neural network on MNIST dataset
    * **performance_match.py** for classification with compressed neural network using the proposed NTK*LC approach on MNIST dataset
  * `code/compression/cifar10`
    * **vgg_net_cifar10.py** for defining VGG19 used for CIFAR10
    * **vgg_train.py** for training VGG19 defined in vgg_net_cifar10.py on CIFAR10, and get parameters for convolution layers, which will be used for feature extraction
    * **performance_origin.py** for classification performance of the original neural network on CIFAR10
    * **performance_two.py** for classification performance of the compressed neural network compressed by our NTK-LC on CIFAR10

* `code/spectrum` contains
  * **tilde_CK.py** for verifying the consistency of theoretical and practical conjugate kernel(CK) spectrum distribution
  * **plot_eigen.py** for ploting eigenvalues and eigenvectors

* `code/utils` contains
  * **activation_numpy.py** for activations definition using numpy
  * **activation_tensor.py** for activations definition using torch
  * **data_prepare.py** for data preparation, including data sampled from MNIST/CIFAR10 and generated GMM data
  * **expect_calculate.py** for expect calculated by numerical integration
  * **expect_calculate_math.py** for expect calculated with analytical expresion
  * **model.py** for model definition
  * **solve_equation.py** for solving equation to define activation functions
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


## Contact information (This par to update!)
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

[^1]: Gu L, Du Y, Zhang Y, et al. Lossless Compression of Deep Neural Networks: A High-dimensional Neural Tangent Kernel Approach[J].[link](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fzhenyu-liao.github.io%2Fpdf%2Fconf%2FRMT4DeepCompress_nips22.pdf)

[^2]: Ali H T, Liao Z, Couillet R. Random matrices in service of ML footprint: ternary random features with no performance loss[J]. arXiv preprint arXiv:2110.01899, 2021.[link](https://arxiv.org/abs/2110.01899)
