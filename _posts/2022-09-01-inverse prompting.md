---
title: inverse prompting IPROMPT
data : 2022-9-1 00:20
tags: GLM
---
# inverse prompting
# Controllable Generation from Pre-trained Language Model via Inverse Prompt

## Abstract
our results show that our proposed method substantially outperforms the baselines and that our gengeration quality is close to human performance on  some of the tasks.
- Demo
```
poem gengeration
https://pretrain.aminer.cn/apps/poetry.html

QA
https://models.aminer.cn/os/qa
```

## Introduction
Inverse prompting can be decoupled into three steps
- first:given a piece of generated text , an inverse prompt is constructed using the generated text.
- sencond: the conditional likelihood of the original prompt given the inverse prompt is computed based on the pre-trained language model
- Third ,the conditional likelihood is used as a score in beam search for selecting the best generation candidates.
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220831231006.png)

## Related Work
inverse prompting does not require any gradient update to the original model and is free of  any additional attribute models.
- Open-Domain Long-Form Question-Answering
  - human evaluation
- Traditional Chinese Poem Generation
  - jiuge(no contemporary notion)

## Methodology
### Baseline: Prompting and Beam Search
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220831233114.png)
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220831233218.png)

### Inverse Prompting

## Implementation
### Base Language Model
we train our base Chinese language model using Megatron-LM with Transformer-XL.
### Open-Domain Long-Form Question-Answering
### Open-Domain Poem Generation
### Self Training for Poem Generation

## Experiments