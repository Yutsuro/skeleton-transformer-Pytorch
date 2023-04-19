# Skeleton Transformer Module in Pytorch

Pytorch implementation of "skeleton transformer module", which is mentioned in [Skeleton-based Action Recognition with Convolutional Neural Networks](https://arxiv.org/abs/1704.07595).

## Install

You can install this module from [PyPI](https://pypi.org/project/skeletorch).

```sh
pip install skeletorch
```

## How to use

### Module

#### skeletorch.SkeletonTransformer

##### Parameters:

All parameters are required.

**timesteps:** Timesteps of input time-series data (equal to number of frames, mentioned as 'T' in the paper)  
**kpts_dim:** Dimentions of keypoints (usually 2 (x, y) or 3 (x, y, z))  
**input_kpts_num:** Number of joints in original keypoints (mentioned as 'N' in the paper)  
**output_dim:** Dimentions of output (mentioned as 'M' in the paper)

##### Input:

**x:** 2-dimentional tensor of shape (timesteps, input_kpts_num*kpts_dim)

## Example        

```python
import torch
from skeletorch import SkeletonTransformer

# parameters
timesteps = 20
kpts_dim = 3
input_kpts_num = 17
output_dim = 10

# input (the size is (20, 51) in this example)
x = torch.Tensor(torch.randn(timesteps, kpts_dim*input_kpts_num))

# make layer
layer = SkeletonTransformer(timesteps, kpts_dim, input_kpts_num, output_dim)

layer(x)
```

Of course you can use this module in your Pytorch network.

## Keras implementation

We also have Keras implementation of this module:  
[https://github.com/Yutsuro/skeleton-transformer-Keras](https://github.com/Yutsuro/skeleton-transformer-Keras)