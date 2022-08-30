---
tags: Pytorch
---
# 保存和加载模型
在本节中，我们将了解如何通过保存、加载和运行模型预测来保持模型状态。
```python
import torch
import torchvision.models as models
```
## 保存和加载模型权重
PyTorch 模型将学习到的参数存储在内部状态字典中，称为state_dict. 这些可以通过以下torch.save 方法持久化：

```python
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```
OUT:
```bash
/opt/conda/lib/python3.7/site-packages/torchvision/models/_utils.py:209: UserWarning:

The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.

/opt/conda/lib/python3.7/site-packages/torchvision/models/_utils.py:223: UserWarning:

Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.

Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/jenkins/.cache/torch/hub/checkpoints/vgg16-397923af.pth

  0%|          | 0.00/528M [00:00<?, ?B/s]
  2%|2         | 11.2M/528M [00:00<00:04, 117MB/s]
  7%|7         | 37.0M/528M [00:00<00:02, 207MB/s]
 12%|#1        | 62.1M/528M [00:00<00:02, 233MB/s]
 17%|#6        | 88.1M/528M [00:00<00:01, 249MB/s]
 21%|##1       | 112M/528M [00:00<00:01, 251MB/s]
 26%|##5       | 136M/528M [00:00<00:01, 250MB/s]
 31%|###       | 162M/528M [00:00<00:01, 256MB/s]
 35%|###5      | 187M/528M [00:00<00:01, 258MB/s]
 40%|####      | 212M/528M [00:00<00:01, 260MB/s]
 45%|####5     | 238M/528M [00:01<00:01, 262MB/s]
 50%|####9     | 263M/528M [00:01<00:01, 263MB/s]
 55%|#####4    | 288M/528M [00:01<00:00, 262MB/s]
 59%|#####9    | 313M/528M [00:01<00:00, 261MB/s]
 64%|######4   | 339M/528M [00:01<00:00, 265MB/s]
 69%|######9   | 365M/528M [00:01<00:00, 267MB/s]
 74%|#######4  | 391M/528M [00:01<00:00, 268MB/s]
 79%|#######8  | 416M/528M [00:01<00:00, 267MB/s]
 84%|########3 | 442M/528M [00:01<00:00, 267MB/s]
 89%|########8 | 467M/528M [00:01<00:00, 267MB/s]
 93%|#########3| 493M/528M [00:02<00:00, 260MB/s]
 98%|#########8| 518M/528M [00:02<00:00, 261MB/s]
100%|##########| 528M/528M [00:02<00:00, 257MB/s]
```
要加载模型权重，首先需要创建一个相同模型的实例，然后使用load_state_dict()方法加载参数。

```python
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```
OUT:
```bash
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```
>一定要model.eval()在推理之前调用方法，将 dropout 和 batch normalization 层设置为评估模式。不这样做会产生不一致的推理结果。

## 使用形状保存和加载模型
加载模型权重时，我们需要先实例化模型类，因为该类定义了网络的结构。我们可能希望将此类的结构与模型一起保存，在这种情况下，我们可以将model（而不是model.state_dict()）传递给保存函数：

```python
torch.save(model, 'model.pth')
```
然后我们可以像这样加载模型：
```python
model = torch.load('model.pth')
```
>这种方法在序列化模型时使用 Python pickle模块，因此它依赖于在加载模型时可用的实际类定义。
