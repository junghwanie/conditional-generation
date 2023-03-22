## Pix2Pix with facades
This code implements the paper: [Image-to-Image Translation with Conditional Adversarial Networks (CVPR 2017 / Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros)](https://arxiv.org/abs/1611.07004).

## Description
### Objective
1. **cGAN Loss**: As an object function, conditional GAN (cGAN) was used to train whether the input image and output image match well.
2. **L1 Loss**: pix2pix adds an additional loss term. Using cGAN loss, Generator is trained to fool Discriminator well, 
### Architecture
1. The basic architecture of Generator follows UNet with skip connection added to encoder-decoder.
2. PatchGAN is used as the architecture of the Discriminator: The L1 loss helps to produce a realistic image at low-frequency, but does not make it possible to produce realistic image at high-frequency. Therefore, a Discriminator is used to model the high-frequency structure.

## Dependencies
+ Python 3.8.12+
+ PyTorch 1.10.2+

## Dataset and usage
You can download the facades dataset by entering [CMP Facade Database](https://cmp.felk.cvut.cz/~tylecr1/facade/).
Before implementing the code, you should change the path of the dataset.
<pre><code>python pix2pix.py</code></pre>
