---
tags: Pytorch
---
# 构建神经网络
神经网络由对数据执行操作的层/模块组成。torch.nn命名空间提供了构建自己的神经网络所需的所有构建块。PyTorch中的每个模块都是 nn.Module 的子类。神经网络是一个模块本身，它由其他模块（层）组成。这种嵌套结构允许轻松构建和管理复杂的架构。

在接下来的部分中，我们将构建一个神经网络来对 FashionMNIST 数据集中的图像进行分类。

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```
## 获取培训设备
我们希望能够在 GPU 等硬件加速器（如果可用）上训练我们的模型。让我们检查一下 torch.cuda是否可用，否则我们继续使用CPU。

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```
OUT:
```bash
Using cuda device
```
## 定义类
我们通过子类化定义我们的神经网络nn.Module，并在 中初始化神经网络层__init__。每个nn.Module子类都在方法中实现对输入数据的操作forward。

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```
我们创建 的实例NeuralNetwork，并将其移动到device，并打印其结构。
```python
model = NeuralNetwork().to(device)
print(model)
```
OUT:
```bash
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

要使用模型，我们将输入数据传递给它。这将执行模型的forward，以及一些后台操作。不要model.forward()直接回调！

在输入上调用模型会返回一个 10 维张量，其中包含每个类的原始预测值。我们通过nn.Softmax模块的一个实例来获得预测概率。

```
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```
OUT:
```bash
Predicted class: tensor([0], device='cuda:0')
```
## 模型层
让我们分解 FashionMNIST 模型中的层。为了说明这一点，我们将抽取 3 张大小为 28x28 的图像的小批量样本，看看当我们通过网络传递它时会发生什么。
```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```
OUT:
```bash
torch.Size([3, 28, 28])
```
### nn.Flatten
我们初始化nn.Flatten 层以将每个 2D 28x28 图像转换为 784 个像素值的连续数组（保持小批量维度（dim=0））。

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```
OUT:
```bash
torch.Size([3, 784])
```

### nn.Linear
线性层是一个模块，它 使用其存储的权重和偏差对输入应用线性变换。
```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```
OUT:
```bash
torch.Size([3, 20])
```

### nn.ReLU
非线性激活是在模型的输入和输出之间创建复杂映射的原因。它们在线性变换后应用以引入非线性，帮助神经网络学习各种现象。

在这个模型中，我们在线性层之间使用nn.ReLU，但是还有其他激活可以在模型中引入非线性。

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```
OUT:
```bash
Before ReLU: tensor([[ 0.2900, -0.0877,  0.5323,  0.8204,  0.5510, -0.2995, -0.2088,  0.0265,
          0.2053, -0.0942, -0.0232,  0.0479, -0.1911, -0.4061,  0.5482,  0.1485,
         -0.1610, -0.0270,  0.0733, -0.3143],
        [ 0.1451, -0.0142,  0.5657,  0.8793,  0.4691, -0.0262,  0.0396,  0.0484,
          0.3901,  0.0995, -0.1170,  0.1374, -0.1050, -0.3080, -0.2261,  0.1466,
         -0.2280,  0.0584, -0.2568,  0.0326],
        [ 0.1920, -0.0377,  0.8050,  0.8166,  0.4925,  0.3568, -0.1310,  0.0982,
          0.2234,  0.3355, -0.0463,  0.0299, -0.1813, -0.4084,  0.1847, -0.0153,
         -0.0925, -0.0377,  0.0320, -0.1275]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.2900, 0.0000, 0.5323, 0.8204, 0.5510, 0.0000, 0.0000, 0.0265, 0.2053,
         0.0000, 0.0000, 0.0479, 0.0000, 0.0000, 0.5482, 0.1485, 0.0000, 0.0000,
         0.0733, 0.0000],
        [0.1451, 0.0000, 0.5657, 0.8793, 0.4691, 0.0000, 0.0396, 0.0484, 0.3901,
         0.0995, 0.0000, 0.1374, 0.0000, 0.0000, 0.0000, 0.1466, 0.0000, 0.0584,
         0.0000, 0.0326],
        [0.1920, 0.0000, 0.8050, 0.8166, 0.4925, 0.3568, 0.0000, 0.0982, 0.2234,
         0.3355, 0.0000, 0.0299, 0.0000, 0.0000, 0.1847, 0.0000, 0.0000, 0.0000,
         0.0320, 0.0000]], grad_fn=<ReluBackward0>)
```
### nn.Sequential
nn.Sequential是一个有序的模块容器。数据按照定义的顺序通过所有模块。您可以使用顺序容器来组合一个快速网络，例如seq_modules.

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

### nn.Softmax
神经网络的最后一个线性层返回logits - [-infty, infty] 中的原始值 - 被传递给 nn.Softmax模块。logits 被缩放为值 [0, 1]，表示模型对每个类别的预测概率。dim参数指示值必须总和为 1 的维度。

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## 模型参数
神经网络内的许多层都是参数化的，即具有在训练期间优化的相关权重和偏差。子类nn.Module化会自动跟踪模型对象中定义的所有字段，并使用模型parameters()或named_parameters()方法使所有参数都可以访问。

在此示例中，我们遍历每个参数，并打印其大小和其值的预览。

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

```
OUT：
```bash
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0142, -0.0281, -0.0273,  ...,  0.0326,  0.0103, -0.0294],
        [-0.0317, -0.0179,  0.0351,  ..., -0.0033,  0.0236,  0.0174]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0014,  0.0115], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0136,  0.0198,  0.0318,  ..., -0.0344, -0.0054, -0.0205],
        [ 0.0294, -0.0173, -0.0092,  ..., -0.0296, -0.0081,  0.0229]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0353,  0.0373], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0083, -0.0411,  0.0099,  ..., -0.0270,  0.0263,  0.0175],
        [-0.0280, -0.0364, -0.0340,  ...,  0.0365, -0.0095, -0.0361]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([0.0139, 0.0053], device='cuda:0', grad_fn=<SliceBackward0>)
```