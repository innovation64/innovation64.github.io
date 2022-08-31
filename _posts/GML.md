---
title: GML
data : 2022-9-1 00:20
tags: GML
---
# key point 
- GLM
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220822172553.png)
#### existing pretraining framework
- autoregressive
- autoencoding
- encoder-decoder
  
## 
>we propose a pretraining framework named GML (General Language Model),based on autoregressive blank infilling.

## we propose two improvements,
- span shuffling 
- 2D positional encoding

## Pretraining objective
### Autoregressive Blank Infilling

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220823233026.png)

### Multi-Task Pretraining
- Document-level
- Ssentence-level

## Model Architecture
>GLM uses a single Transformer with several modifications to the architecture

### 2D Positional Encoding

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220823234948.png)

### Finetuning GLM

## Experiments
### pretraining setup
#### data using
- BooksCorpus
- English Wikipedia 
#### multi-task pretraining 
two Large-sized model with a mixture of the blank infilling objective, denoted as GLM(doc) and GLM(sent)
#### Compare with SOTA models
### superGLUE
### Multi-Task Pretraining 
>we evaluate the multi-task model for NLU ,seq2seq,blank infilling, and zero-shot language modeling.

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220824001632.png)

- SuperGLUE 
- Sequence-to-Sequence
- Text Infilling
  
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220824001910.png)

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20220824002021.png)

### Ablation Study
