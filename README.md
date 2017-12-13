# Image similarity on Fashion MNIST
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fchollet/keras/blob/master/LICENSE)
## How to use

You need to changed the image size limitation in MobileNet of Keras firstly.

- Use SiameseNet_pretrain.ipynb to pre-train on MobileNet.
- Use SiameseNet.ipynb to train SiameseNet and visualize results.
- You can also use train.sh to train on command-line mode.

MobileNet.h5 and MobileNet_sim.h5 is the saved model and weights.

I use random_eraser.py from [cutout-random-erasing](https://github.com/yu4u/cutout-random-erasing.).
Thanks yu4u.

I also reference on [Image similarity using Deep CNN and Curriculum Learning] (https://arxiv.org/pdf/1709.08761.pdf.). 
