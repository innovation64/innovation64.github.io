---
tags: Paper
---
# Parameter-Efficient Prompt Tuning Makes Generalized and Calibrated Neural Text Retrievers
## Abstract


提示调整尝试更新预训练模型中的少数任务特定参数。 它在语言理解和生成任务上都取得了与微调完整参数集相当的性能。 在这项工作中，我们研究了神经文本检索器的快速调整问题。 我们为跨域、跨域和跨主题设置的文本检索引入了参数有效的提示调整。 通过广泛的分析，我们表明该策略可以缓解基于微调的检索方法面临的两个问题——参数效率低和泛化性弱。 值得注意的是，它可以显着提高检索模型的域外零样本泛化能力。
通过仅更新 0.1% 的模型参数，即时调优策略可以帮助检索模型获得比更新所有参数的传统方法更好的泛化性能。 最后，为了便于研究检索器的跨主题泛化性，我们策划并发布了一个学术检索数据集，其中包含 87 个主题的 18K 查询结果对，使其成为迄今为止最大的特定主题数据集。 1


# Introduction
parameter-efficiency

generalizability

parameter redundancy

Furthermore, fine-tuning the full parameters
of a pre-trained retriever for multi-lingual (Litschko
et al., 2022) or cross-topic settings can also result
in parameter-inefficiency. 

examine a line of mainstream PE methods
- in-domain
- crossdomain
- cross-topic settings.

## first
PE prompt tuning
can help empower the neural model with better
confidence calibration, which refers to the theoretical
principle that a model’s predicted probabilities
of labels should correspond to the ground-truth
correctness likelihood
## Second
it encourages better performance on queries with
different lengths from in-domain training, demonstrating
PE methods’ generalization capacity to
out-of-domain datasets.

>this work aims to advance the neural text retrievers from three aspects
- problem:

    we propose to leverage PE learning
- Understanding

     its confidence-calibrated prediction and query-length robustness.
- Dataset:

    we construct OAG-QA
# Related Work
- Neural Text Retrieval
- Generalization in Text Retrieval
- Parameter-Efficient Learning

# Challenges in Neural Text Retrieval
- Dense Retriever

    the Noise Contrastive Error (NCE)
- Late-Interaction Retriever
- Parameter Inefficieny
    
     substantial parameter redundancy from two aspects
     - first

        training dual-encoders double the size of the parameters to be tuned.

    - Second

        the cross-lingual (Litschko et al., 2022) and crossdomain
(Thakur et al., 2021) transfer may require
additional full-parameter tuning on each of the
individual tasks and consequently increase the
number of parameters by several times
- Weak Generalizability

    cannot generalize well to zero-shot cross-domain benchmarks

     widely adopted in downstream scenarios

     expensive.

# Parameter-Efficient Transfer Learning

PE learning
aims to achieve comparable performance to finetuning
by tuning only a small portion of parameters
per task
## Parameter-Efficient Learning Methods
- Adapters
- BitFit(self-attention,FFN,Layer Norm Operations)
- Lester et al. &P-Tuning
- Prefix-Tuning & P-Tuning v2 .
# In-Domain Parameter-Efficiency
# Cross-Domain and Cross-Topic Generalizability
we present OAG-QA (Cf. Table
3) which consists of 17,948 unique queries
from 22 scientific disciplines and 87 fine-grained
topics.
# Conclusion
PE learning can achieve comparable performance
to full-parameter fine-tuning in in-domain

Finally, we construct
and release the largest fine-grained topic-specific
academic retrieval dataset OAG-QA,

## Discussion
- first
    a long-standing challenge is that
it converges slower and is relatively more sensitive
to hyper-parameters 
- Second
    dataset requires further exploration.
- Third
    However, many other practical problems also suffer from the challenges of biased training data and generalization
