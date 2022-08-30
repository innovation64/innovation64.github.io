---
tags: Pytorch
---
# 优化模型参数
现在我们有了模型和数据，是时候通过优化数据上的参数来训练、验证和测试我们的模型了。训练模型是一个迭代过程；在每次迭代（称为epoch）中，模型对输出进行猜测，计算猜测中的误差（损失），收集误差关于其参数的导数（如我们在上一节中所见），并优化这些参数使用梯度下降。有关此过程的更详细演练，请查看来自 3Blue1Brown 的有关反向传播的视频。
## 先决条件代码
我们从前面的Datasets & DataLoaders 和Build Model部分加载代码。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

model = NeuralNetwork()
```
OUT:
```bash
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 32768/26421880 [00:00<01:27, 300812.12it/s]
  0%|          | 65536/26421880 [00:00<01:27, 299676.26it/s]
  0%|          | 131072/26421880 [00:00<01:00, 435855.89it/s]
  1%|          | 229376/26421880 [00:00<00:42, 618270.26it/s]
  2%|1         | 491520/26421880 [00:00<00:20, 1256975.31it/s]
  4%|3         | 950272/26421880 [00:00<00:11, 2254336.05it/s]
  7%|7         | 1933312/26421880 [00:00<00:05, 4449070.57it/s]
 15%|#4        | 3833856/26421880 [00:00<00:02, 8558864.69it/s]
 26%|##6       | 6946816/26421880 [00:00<00:01, 14739548.26it/s]
 37%|###6      | 9699328/26421880 [00:01<00:00, 17948117.57it/s]
 48%|####8     | 12812288/26421880 [00:01<00:00, 21148273.09it/s]
 60%|######    | 15958016/26421880 [00:01<00:00, 23377537.29it/s]
 72%|#######1  | 18972672/26421880 [00:01<00:00, 24686191.09it/s]
 84%|########3 | 22118400/26421880 [00:01<00:00, 25887453.20it/s]
 95%|#########5| 25231360/26421880 [00:01<00:00, 26490039.68it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 15996303.22it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 268834.70it/s]
100%|##########| 29515/29515 [00:00<00:00, 267709.18it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|          | 32768/4422102 [00:00<00:14, 306434.54it/s]
  1%|1         | 65536/4422102 [00:00<00:14, 304724.01it/s]
  3%|2         | 131072/4422102 [00:00<00:09, 443021.34it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 628264.87it/s]
 10%|#         | 458752/4422102 [00:00<00:03, 1168691.48it/s]
 21%|##1       | 950272/4422102 [00:00<00:01, 2320415.95it/s]
 43%|####2     | 1900544/4422102 [00:00<00:00, 4437123.54it/s]
 86%|########5 | 3801088/4422102 [00:00<00:00, 8640843.63it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 5112026.92it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 22168662.21it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## 超参数
超参数是可调整的参数，可让您控制模型优化过程。不同的超参数值会影响模型训练和收敛速度（阅读有关超参数调整的更多信息）

我们为训练定义了以下超参数：
- Number of Epochs - 迭代数据集的次数

- Batch Size - 参数更新前通过网络传播的数据样本数

- learning rate- 在每个批次/时期更新模型参数的程度。较小的值会产生较慢的学习速度，而较大的值可能会导致训练期间出现不可预测的行为。

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```
## 优化循环
一旦我们设置了超参数，我们就可以使用优化循环来训练和优化我们的模型。优化循环的每次迭代称为epoch

每个时期包括两个主要部分：
- 训练循环- 迭代训练数据集并尝试收敛到最佳参数。

- 验证/测试循环- 迭代测试数据集以检查模型性能是否正在改善。

让我们简要熟悉一下训练循环中使用的一些概念。继续查看优化循环的完整实现。

## 损失函数
当呈现一些训练数据时，我们未经训练的网络可能不会给出正确的答案。损失函数衡量得到的结果与目标值的相异程度，是我们在训练时要最小化的损失函数。为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实数据标签值进行比较。

常见的损失函数包括用于回归任务的nn.MSELoss（均方误差）和 用于分类的nn.NLLLoss（负对数似然）。 nn.CrossEntropyLoss结合nn.LogSoftmax和nn.NLLLoss。

我们将模型的输出 logits 传递给nn.CrossEntropyLoss，这将对 logits 进行归一化并计算预测误差。

```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

## 优化器
优化是在每个训练步骤中调整模型参数以减少模型误差的过程。优化算法定义了如何执行这个过程（在这个例子中，我们使用随机梯度下降）。所有优化逻辑都封装在optimizer对象中。在这里，我们使用 SGD 优化器；此外，PyTorch 中有许多不同的优化器 可用，例如 ADAM 和 RMSProp，它们可以更好地用于不同类型的模型和数据。

我们通过注册模型需要训练的参数并传入学习率超参数来初始化优化器。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
在训练循环中，优化分三个步骤进行：
- 调用optimizer.zero_grad()以重置模型参数的梯度。默认情况下渐变加起来；为了防止重复计算，我们在每次迭代时明确地将它们归零。

- 通过调用来反向传播预测损失loss.backward()。PyTorch 存储每个参数的损失梯度。

- 一旦我们有了我们的梯度，我们调用optimizer.step()通过在反向传递中收集的梯度来调整参数。

## 全程执行
我们定义train_loop循环优化代码，并test_loop根据我们的测试数据评估模型的性能。

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```
我们初始化损失函数和优化器，并将其传递给train_loop和test_loop。随意增加 epoch 的数量来跟踪模型的改进性能。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

OUT:
```bash
Epoch 1
-------------------------------
loss: 2.312421  [    0/60000]
loss: 2.299128  [ 6400/60000]
loss: 2.280626  [12800/60000]
loss: 2.277924  [19200/60000]
loss: 2.257964  [25600/60000]
loss: 2.230069  [32000/60000]
loss: 2.236777  [38400/60000]
loss: 2.200969  [44800/60000]
loss: 2.206201  [51200/60000]
loss: 2.169108  [57600/60000]
Test Error:
 Accuracy: 37.9%, Avg loss: 2.162097

Epoch 2
-------------------------------
loss: 2.177935  [    0/60000]
loss: 2.160434  [ 6400/60000]
loss: 2.107404  [12800/60000]
loss: 2.123934  [19200/60000]
loss: 2.075139  [25600/60000]
loss: 2.012481  [32000/60000]
loss: 2.039521  [38400/60000]
loss: 1.957680  [44800/60000]
loss: 1.971344  [51200/60000]
loss: 1.897263  [57600/60000]
Test Error:
 Accuracy: 54.8%, Avg loss: 1.891562

Epoch 3
-------------------------------
loss: 1.930147  [    0/60000]
loss: 1.888406  [ 6400/60000]
loss: 1.777649  [12800/60000]
loss: 1.823289  [19200/60000]
loss: 1.717290  [25600/60000]
loss: 1.660339  [32000/60000]
loss: 1.680510  [38400/60000]
loss: 1.578335  [44800/60000]
loss: 1.606417  [51200/60000]
loss: 1.507668  [57600/60000]
Test Error:
 Accuracy: 61.2%, Avg loss: 1.519658

Epoch 4
-------------------------------
loss: 1.590842  [    0/60000]
loss: 1.546384  [ 6400/60000]
loss: 1.403137  [12800/60000]
loss: 1.480332  [19200/60000]
loss: 1.365523  [25600/60000]
loss: 1.355569  [32000/60000]
loss: 1.365460  [38400/60000]
loss: 1.287711  [44800/60000]
loss: 1.324499  [51200/60000]
loss: 1.233377  [57600/60000]
Test Error:
 Accuracy: 63.4%, Avg loss: 1.253872

Epoch 5
-------------------------------
loss: 1.336040  [    0/60000]
loss: 1.310028  [ 6400/60000]
loss: 1.150771  [12800/60000]
loss: 1.260314  [19200/60000]
loss: 1.138190  [25600/60000]
loss: 1.157820  [32000/60000]
loss: 1.175813  [38400/60000]
loss: 1.109522  [44800/60000]
loss: 1.150945  [51200/60000]
loss: 1.074586  [57600/60000]
Test Error:
 Accuracy: 64.8%, Avg loss: 1.091219

Epoch 6
-------------------------------
loss: 1.167744  [    0/60000]
loss: 1.162937  [ 6400/60000]
loss: 0.986825  [12800/60000]
loss: 1.124647  [19200/60000]
loss: 0.997677  [25600/60000]
loss: 1.025049  [32000/60000]
loss: 1.058906  [38400/60000]
loss: 0.996249  [44800/60000]
loss: 1.037670  [51200/60000]
loss: 0.974875  [57600/60000]
Test Error:
 Accuracy: 66.0%, Avg loss: 0.986160

Epoch 7
-------------------------------
loss: 1.050311  [    0/60000]
loss: 1.067898  [ 6400/60000]
loss: 0.874420  [12800/60000]
loss: 1.033823  [19200/60000]
loss: 0.907883  [25600/60000]
loss: 0.930730  [32000/60000]
loss: 0.981958  [38400/60000]
loss: 0.922502  [44800/60000]
loss: 0.958236  [51200/60000]
loss: 0.907394  [57600/60000]
Test Error:
 Accuracy: 67.4%, Avg loss: 0.913863

Epoch 8
-------------------------------
loss: 0.962714  [    0/60000]
loss: 1.001379  [ 6400/60000]
loss: 0.793249  [12800/60000]
loss: 0.968555  [19200/60000]
loss: 0.847230  [25600/60000]
loss: 0.861087  [32000/60000]
loss: 0.927261  [38400/60000]
loss: 0.872564  [44800/60000]
loss: 0.899874  [51200/60000]
loss: 0.857955  [57600/60000]
Test Error:
 Accuracy: 68.4%, Avg loss: 0.861201

Epoch 9
-------------------------------
loss: 0.894283  [    0/60000]
loss: 0.950933  [ 6400/60000]
loss: 0.731788  [12800/60000]
loss: 0.919315  [19200/60000]
loss: 0.803535  [25600/60000]
loss: 0.808337  [32000/60000]
loss: 0.885696  [38400/60000]
loss: 0.837751  [44800/60000]
loss: 0.855697  [51200/60000]
loss: 0.819550  [57600/60000]
Test Error:
 Accuracy: 69.7%, Avg loss: 0.821085

Epoch 10
-------------------------------
loss: 0.839493  [    0/60000]
loss: 0.910036  [ 6400/60000]
loss: 0.683531  [12800/60000]
loss: 0.880390  [19200/60000]
loss: 0.770017  [25600/60000]
loss: 0.767418  [32000/60000]
loss: 0.852382  [38400/60000]
loss: 0.812122  [44800/60000]
loss: 0.821206  [51200/60000]
loss: 0.788199  [57600/60000]
Test Error:
 Accuracy: 70.8%, Avg loss: 0.789140

Done!
```