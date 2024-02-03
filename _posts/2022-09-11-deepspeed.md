---
tags: GLM
---
# Deepspeed
## 新手入门
### 安装

```bash
pip install deepspeed
```
- deepspeed 在 HF Transformer 和 Pytorch Lightning 都有组件，可以直接在这俩基础的项目上直接用， HF的用--deepspeed + config 的方式使用， Pytorch Lightning 的话直接用看Trainer。

-  这破玩意在AMD上用ROCm的镜像

### 咋写

初始化
```python

model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)

```
这玩意保证了混合精度和分布式数据加载标记，简单封装。

这里初始化分布式环境，操作这部分代码

```python
torch.distributed.init_process_group(...)
```

换成

```python

deepspeed.init_distributed()
```

这破玩意默认使用 NCCL，你可以自己重写


### 训练

初始化完后，调 3 个 API 去前向传播，反向传播，和参数更新

```python

for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()

```

- 平均梯度：在分布式数据并行训练中，向后确保在Train_batch_size上训练后在数据并行过程中平均梯度。
- 损失缩放：在FP16/混合精度训练中，deepspeed 会自动处理缩放损失，以避免梯度的精确损失。
- 学习率调度程序：当使用DeepSpeed的学习率调度程序（在DS_Config.json文件中指定）时，DeepSpeed在每个培训步骤中调用调度程序的step（）方法（执行model_engine.step（）时）。当不使用DeepSpeed的学习率调度程序时：


### 模型检查

 用这俩 save_checkpoint  load_checkpoint API

 - ckpt_dir ：检查点储存位置
 - ckpt_id : 唯一ID

 ```python
 #load checkpoint
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
step = client_sd['step']

#advance data loader to ckpt step
dataloader_to_step(data_loader, step + 1)

for step, batch in enumerate(data_loader):

    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()

    #save checkpoint
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)
 ```

为了支持这些项目，save_checkpoint接受客户端状态词典client_sd用于保存。这些项目可以从load_checkpoint作为返回参数检索。在上面的示例中，步骤值作为 client_sd 的一部分存储。

### DeepSpeed Configuration

```json

{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": true
```

### 启动训练

github.com
以下是你要求翻译的内容：

DeepSpeed 安装了入口点 deepspeed 以启动分布式训练。我们以以下假设来说明 DeepSpeed 的一个示例用法：

1. 你已经将 DeepSpeed 集成到你的模型中
2. client_entry.py 是你模型的入口脚本
3. client args 是 argparse 命令行参数
4. ds_config.json 是 DeepSpeed 的配置文件

### 多节点资源配置
适配于 OpenMPI 和 Horovod

```
worker-1 slots=4
worker-2 slots=4
```

## ZeRO++
这破玩意是一个通讯优化策略，因为ZeRO不能高效匹配LLM训练，受限于扩展或带宽。
## 挖坑待补