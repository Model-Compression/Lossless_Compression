# Lossless Compression(An NTK based method)

This repository contains code for the paper "*Revisiting and Accelerating Transfer Component Analysis*" submitted to NeurIPS 2022.

## About the code

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

[^2]: Long M, Wang J, Ding G, et al. [Transfer feature learning with joint distribution adaptation](http://openaccess.thecvf.com/content_iccv_2013/html/Long_Transfer_Feature_Learning_2013_ICCV_paper.html)[C]//Proceedings of the IEEE international conference on computer vision. 2013: 2200-2207.

[^3]: Sun B, Feng J, Saenko K. [Return of frustratingly easy domain adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/10306)[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2016, 30(1).

[^4]: Ghifary M, Kleijn W B, Zhang M. [Domain adaptive neural networks for object recognition](https://link.springer.com/chapter/10.1007/978-3-319-13560-1_76)[C]//Pacific Rim international conference on artificial intelligence. Springer, Cham, 2014: 898-904.

[^5]: Gong B, Shi Y, Sha F, et al. [Geodesic flow kernel for unsupervised domain adaptation](https://ieeexplore.ieee.org/abstract/document/6247911/)[C]//2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012: 2066-2073.