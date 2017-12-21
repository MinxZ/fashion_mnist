# Image similarity on Fashion MNIST
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/MinxZ/fashion_mnist/blob/master/LICENSE)
## How to use

You need to changed the image size limitation in MobileNet of Keras firstly.

- Use SiameseNet_pretrain.ipynb to pre-train on MobileNet.
- Use SiameseNet.ipynb to train SiameseNet and visualize results.
- You can also use train.sh to train on command-line mode.

MobileNet.h5 and MobileNet_sim.h5 is the saved model and weights.

I use random_eraser.py from [cutout-random-erasing](https://github.com/yu4u/cutout-random-erasing.).
Thanks yu4u.

## References
[1] T. DeVries and G. W. Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout," in arXiv:1708.04552, 2017.

[2] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," in arXiv:1708.04896, 2017.

[3] Srikar Appalaraju, Vineet Chaoji, "Image similarity using Deep CNN and Curriculum Learning," in arXiv:1709.08761, 2017.
