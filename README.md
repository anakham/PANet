# Path Aggregation Network (PANet) for Chameleon AI Tools Highwai simulator by Mindtech 
on PyTorch 1.2 and AMD Radeon Open Compute (ROCm)

This is an adaptation of
[Path Aggregation Network for Instance Segmentation (PANet)](https://github.com/ShuLiu1993/PANet)
by [Shu Liu](http://shuliu.me), Lu Qi, Haifang Qin, [Jianping Shi](https://shijianping.me/), [Jiaya Jia](http://jiaya.me/)
for training PANet on Cityscapes dataset along with synthetic data produced by **Chameleon AI Tools
[Highwai simulator](https://www.mindtech.global/products)** by [Mindtech](https://www.mindtech.global)  

Domain-Adversarial Training of Neural Networks ([DANN](https://arxiv.org/abs/1505.07818)) is implemented.

This implementation can be installed both on **NVIDIA CUDA** and 
**AMD Radeon Open Compute (ROCm)** with PyTorch 1.2. 

To install:
```shell
git clone https://github.com/anakham/PANet.git
cd PANet/lib
git checkout Chameleon
sh make.sh
```

See the [original README](https://github.com/ShuLiu1993/PANet/blob/master/README.md)
for more details about installation and usage.

See **<TODO: link to the whole description>** for the whole experiment description.