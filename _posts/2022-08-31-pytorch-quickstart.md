---
tags: Pytorch
---

# 快速开始
>本节包含ML中常见任务API，具体深入请参考相关链接
## 数据处理
>pytorch 有两个处理数据的包：**torch.utils.data.DataLoader** 和**torch.utils.data.Dataset**。dataset存储样本及对应标签，并通过DataLoader加载Dataset

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch 提供特定领域的库，比如**TorchText**、**TorchVision**和**TorchAudio**,所有这些库都包含数据集。在本教程中我们将使用**TorchVision**数据集
>该torchvision.datasets模块包含Dataset许多真实世界视觉数据的对象，如 CIFAR、COCO（[此处为完整列表](https://pytorch.org/vision/stable/datasets.html)）。在本教程中，我们使用 FashionMNIST 数据集。每个 TorchVision 都Dataset包含两个参数：transform和 target_transform分别修改样本和标签

```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```
out:
```bash
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:23, 316892.83it/s]
  0%|          | 131072/26421880 [00:00<01:00, 436688.36it/s]
  1%|          | 229376/26421880 [00:00<00:42, 610932.89it/s]
  2%|1         | 491520/26421880 [00:00<00:21, 1233770.64it/s]
  4%|3         | 950272/26421880 [00:00<00:11, 2215146.36it/s]
  7%|7         | 1933312/26421880 [00:00<00:05, 4375338.79it/s]
 15%|#4        | 3833856/26421880 [00:00<00:02, 8455842.06it/s]
 26%|##6       | 6979584/26421880 [00:00<00:01, 14652653.31it/s]
 38%|###8      | 10092544/26421880 [00:01<00:00, 18878820.79it/s]
 50%|####9     | 13172736/26421880 [00:01<00:00, 21654219.14it/s]
 62%|######1   | 16285696/26421880 [00:01<00:00, 23649710.22it/s]
 72%|#######2  | 19136512/26421880 [00:01<00:00, 24129058.38it/s]
 84%|########4 | 22282240/26421880 [00:01<00:00, 25412955.15it/s]
 96%|#########6| 25427968/26421880 [00:01<00:00, 26315921.43it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 16023208.49it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 272213.07it/s]
100%|##########| 29515/29515 [00:00<00:00, 270949.40it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|          | 32768/4422102 [00:00<00:14, 304160.00it/s]
  1%|1         | 65536/4422102 [00:00<00:14, 302890.74it/s]
  3%|2         | 131072/4422102 [00:00<00:09, 440396.43it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 624615.73it/s]
 11%|#1        | 491520/4422102 [00:00<00:03, 1270598.77it/s]
 21%|##1       | 950272/4422102 [00:00<00:01, 2277278.48it/s]
 44%|####3     | 1933312/4422102 [00:00<00:00, 4493671.94it/s]
 87%|########6 | 3833856/4422102 [00:00<00:00, 8645529.02it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 5082787.38it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 20486031.30it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```
我们将Dataset作为参数传递给DataLoader。这里我们的数据集可以进行迭代，并支持自动批处理、采样、混洗和多进程数据加载。这里我们定义了一个64的batch size，即dataloader iterable中的每个元素都会返回一个batch 64个特征和标签。
```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```
Out:
```bash
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```