---
tags: Pytorch
---
# 变换
数据并不总是以训练机器学习算法所需的最终处理形式出现。我们使用转换来对数据进行一些操作并使其适合训练。

所有 TorchVision 数据集都有两个参数 -transform修改特征和 target_transform修改标签 - 接受包含转换逻辑的可调用对象。torchvision.transforms模块提供了几个开箱即用的常用转换。

FashionMNIST 特征是 PIL 图像格式，标签是整数。对于训练，我们需要将特征作为归一化张量，并将标签作为 one-hot 编码张量。为了进行这些转换，我们使用ToTensor和Lambda。

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```
OUT:
```bash
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 32768/26421880 [00:00<01:28, 299765.65it/s]
  0%|          | 65536/26421880 [00:00<01:28, 299191.01it/s]
  0%|          | 131072/26421880 [00:00<01:00, 434884.81it/s]
  1%|          | 229376/26421880 [00:00<00:42, 617628.19it/s]
  2%|1         | 491520/26421880 [00:00<00:20, 1254839.54it/s]
  4%|3         | 950272/26421880 [00:00<00:11, 2247250.80it/s]
  7%|7         | 1933312/26421880 [00:00<00:05, 4375937.74it/s]
 13%|#3        | 3440640/26421880 [00:00<00:03, 7429829.13it/s]
 24%|##4       | 6356992/26421880 [00:00<00:01, 13425422.18it/s]
 36%|###5      | 9469952/26421880 [00:01<00:00, 17838019.87it/s]
 48%|####7     | 12582912/26421880 [00:01<00:00, 21023827.16it/s]
 59%|#####9    | 15663104/26421880 [00:01<00:00, 23401209.75it/s]
 70%|#######   | 18513920/26421880 [00:01<00:00, 23852721.08it/s]
 82%|########1 | 21659648/26421880 [00:01<00:00, 25253963.30it/s]
 93%|#########3| 24674304/26421880 [00:01<00:00, 25829143.09it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 15936096.08it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 271014.05it/s]
100%|##########| 29515/29515 [00:00<00:00, 269952.64it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|          | 32768/4422102 [00:00<00:14, 296450.62it/s]
  1%|1         | 65536/4422102 [00:00<00:14, 295773.24it/s]
  3%|2         | 131072/4422102 [00:00<00:09, 430422.11it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 611205.37it/s]
 11%|#1        | 491520/4422102 [00:00<00:03, 1242451.42it/s]
 21%|##1       | 950272/4422102 [00:00<00:01, 2227955.27it/s]
 44%|####3     | 1933312/4422102 [00:00<00:00, 4395357.96it/s]
 87%|########6 | 3833856/4422102 [00:00<00:00, 8454963.91it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 4975278.73it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 26140771.18it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```
## ToTensor()
ToTensor 将 PIL 图像或 NumPyndarray转换为FloatTensor. 并在 [0., 1.] 范围内缩放图像的像素强度值
## Lambda 变换
Lambda 转换应用任何用户定义的 lambda 函数。在这里，我们定义了一个函数来将整数转换为 one-hot 编码张量。它首先创建一个大小为 10 的零张量（我们数据集中的标签数量）并调用 scatter_，它在标签上分配 a value=1给定的索引y。

```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```