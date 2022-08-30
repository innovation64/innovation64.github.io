---
tags: Pytorch
---
# 数据集和数据加载器
>处理数据样本的代码可能会变得混乱且难以维护；理想情况下，我们希望我们的数据集代码与我们的模型训练代码分离，以获得更好的可读性和模块化。PyTorch 提供了两个数据原语：torch.utils.data.DataLoader允许torch.utils.data.Dataset 您使用预加载的数据集以及您自己的数据。 Dataset存储样本及其对应的标签，并DataLoader在 周围包裹一个可迭代对象Dataset，以便轻松访问样本。

PyTorch 域库提供了许多预加载的数据集（例如 FashionMNIST），这些数据集子类torch.utils.data.Dataset化并实现了特定于特定数据的功能。它们可用于对您的模型进行原型设计和基准测试。你可以在这里找到它们：图像数据集、 文本数据集和 音频数据集
## 加载数据集
下面是如何从 TorchVision 加载Fashion-MNIST数据集的示例。Fashion-MNIST 是 Zalando 文章图像的数据集，由 60,000 个训练示例和 10,000 个测试示例组成。每个示例都包含 28×28 灰度图像和来自 10 个类别之一的相关标签。

我们使用以下参数加载FashionMNIST 数据集：
- root是存储训练/测试数据的路径，

- train指定训练或测试数据集，

- download=True如果数据不可用，则从 Internet 下载数据root。

- transform并target_transform指定特征和标签转换

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```
OUT:
```bash
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 32768/26421880 [00:00<01:27, 301106.05it/s]
  0%|          | 65536/26421880 [00:00<01:28, 299249.14it/s]
  0%|          | 131072/26421880 [00:00<01:00, 434819.33it/s]
  1%|          | 229376/26421880 [00:00<00:42, 616076.86it/s]
  2%|1         | 491520/26421880 [00:00<00:20, 1252390.90it/s]
  4%|3         | 950272/26421880 [00:00<00:11, 2244425.58it/s]
  7%|7         | 1933312/26421880 [00:00<00:05, 4428479.29it/s]
 15%|#4        | 3833856/26421880 [00:00<00:02, 8527824.01it/s]
 26%|##6       | 6979584/26421880 [00:00<00:01, 14773537.06it/s]
 38%|###8      | 10092544/26421880 [00:01<00:00, 18978306.16it/s]
 50%|####9     | 13172736/26421880 [00:01<00:00, 21738082.11it/s]
 62%|######1   | 16285696/26421880 [00:01<00:00, 23693088.85it/s]
 74%|#######3  | 19431424/26421880 [00:01<00:00, 25076357.81it/s]
 85%|########5 | 22511616/26421880 [00:01<00:00, 26054963.69it/s]
 97%|#########6| 25559040/26421880 [00:01<00:00, 26539207.15it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 15932941.16it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 274124.47it/s]
100%|##########| 29515/29515 [00:00<00:00, 272779.90it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|          | 32768/4422102 [00:00<00:14, 302384.63it/s]
  1%|1         | 65536/4422102 [00:00<00:14, 300938.90it/s]
  3%|2         | 131072/4422102 [00:00<00:09, 437453.15it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 620565.75it/s]
 10%|9         | 425984/4422102 [00:00<00:03, 1046322.17it/s]
 20%|##        | 884736/4422102 [00:00<00:01, 2118707.42it/s]
 40%|####      | 1769472/4422102 [00:00<00:00, 4073599.94it/s]
 79%|#######9  | 3506176/4422102 [00:00<00:00, 7850346.61it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 5052165.16it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 23317793.73it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```
## 迭代和可视化数据集
>我们可以把Datasets像列表一样手动索引：training_data[index]. 我们matplotlib用来可视化训练数据中的一些样本

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

```

![https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_001.png)

## 自定义数据集
自定义 Dataset 类必须实现三个函数：__init__、__len__和__getitem__。看看这个实现；FashionMNIST 图像存储在一个目录img_dir中，它们的标签分别存储在一个 CSV 文件annotations_file中。

在接下来的部分中，我们将分解每个函数中发生的事情。

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```
### __init__
__init__ 函数在实例化 Dataset 对象时运行一次。我们初始化包含图像、注释文件和两种转换的目录（在下一节中更详细地介绍）。

labels.csv 文件如下所示：

```bash
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

```

### __len__

__len__ 函数返回我们数据集中的样本数。

example
```python
def __len__(self):
    return len(self.img_labels)
```

### __getitem__
__getitem__ 函数从给定索引处的数据集中加载并返回一个样本idx。基于索引，它识别图像在磁盘上的位置，使用 将其转换为张量read_image，从 csv 数据中检索相应的标签self.img_labels，调用它们的转换函数（如果适用），并返回张量图像和相应的标签一个元组。

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```

## 使用 DataLoaders 为训练准备数据

检索我们数据集的Dataset特征并一次标记一个样本。在训练模型时，我们通常希望以“小批量”的形式传递样本，在每个 epoch 重新洗牌以减少模型过拟合，并使用 Pythonmultiprocessing加速数据检索。

DataLoader是一个可迭代的，它在一个简单的 API 中为我们抽象了这种复杂性。

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## 遍历DataLoader
我们已将该数据集加载到 中，DataLoader并且可以根据需要遍历数据集。下面的每次迭代都会返回一批train_features和train_labels（分别包含batch_size=64特征和标签）。因为我们指定shuffle=True了 ，所以在我们遍历所有批次之后，数据被打乱（为了更细粒度地控制数据加载顺序，请查看Samplers）。

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

```
![https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_002.png](https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_002.png)