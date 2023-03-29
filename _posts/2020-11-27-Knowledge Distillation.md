---
tags: GLM
---
# Knowledge Distillation

>在2015年，Hinton等人首次提出神经网络中的知识蒸馏(Knowledge Distillation, KD)技术/概念
### 模型压缩四个技术
- 设计高效小型网络
- 剪枝
- 量化
- 蒸馏
  - from deep and large to shallow and small network
  - from ensembles of classifiers to individual classifier

### 暗知识(dark knowledge)
低概率类别与高概率类别的关系

## KD theory
- 利用大规模数据训练一个教师网络
- 利用大规模数据训练一个学生网络
### loss function
- 一部分是拿教师和学生网络的输出logits计算蒸馏损失/KL散度
- 一部分是拿学生网络的输出和数据标签计算交叉熵损失

### FitNets
FitNets Romero等人的工作[6]不仅利用教师网络的最后输出logits，还利用它的中间隐层参数值(intermediate representations)，训练学生网络，获得又深又细的FitNets。前者是KD的内容，后者是作者提出的hint-based training

![a](https://pic1.zhimg.com/80/v2-891ca916ce38cae61af55bd25f9f1694_720w.webp)

### Paying More Attention to Attention
Zagoruyko和Komodakis提出用注意力去做知识迁移[7]，具体而言，用的是activation-based和gradient-based空间注意力图，如图2所示。activation-based空间注意力图，构造一个映射函数F，其输入是三维激活值张量，输出是二维空间注意力图。这样的映射函数F，作者给了三种，并且都是有效的。gradient-based空间注意力图使用的是Simonyan et al.的工作[8]，即输入敏感图。简单而言，KD希望教师和学生网络的输出概率分布类似，而Paying More Attention to Attention希望教师和学生网络的中间激活和原始RGB被激活区域类似。两种知识迁移的效果可以叠加

![](https://pic4.zhimg.com/80/v2-d1398eed88eb9cd7fbbf5f43365bda07_720w.webp)

### Learning Efficient Object Detection Models
Chen等人的工作[9]使用KD[2]和hint learning[6]两种方法，将教师Faster R-CNN的知识迁移到学生Faster R-CNN上，如图3所示，针对物体检测，作者将原始KD的交叉熵损失改为类别加权交叉熵损失解决分类中的不平衡问题；针对检测框回归，作者使用教师回归损失作为学生回归损失的upper bound；针对backbone的中间表示，作者使用hint learning做特征适应。二阶段检测器比起一阶段检测器和anchor-free检测器较复杂，相信KD在未来会被用于一阶段和anchor-free检测器。这篇文章为物体检测和知识蒸馏的结合提供了实践经验。

![](https://pic4.zhimg.com/80/v2-f8729cc298048b781d5078c31cf646b7_720w.webp)

### Neuron Selectivity Transfer NST
使用Maximum Mean Discrepancy去做中间计算层的分布匹配

### DarkRank
认为KD使用的知识来自单一样本，忽略不同样本间的关系。因此，作者提出新的知识，该知识来自跨样本相似性。验证的任务是person re-identification、image retrieval。

###  A Gift from Knowledge Distillation
Yim等人的工作[12]展示了KD对于以下三种任务有帮助：1、网络快速优化，2、模型压缩，3、迁移学习。作者的知识蒸馏方法是让学生网络的FSP矩阵(the flow of solution procedure)和教师网络的FSP矩阵尽可能相似。FSP矩阵是指一个卷积网络的两层计算层结果(特征图)的关系，如图4所示，用Gramian矩阵去描述这种关系。教师网络的“知识”以数个FSP矩阵的形式被提取出来。最小化教师网络和学生网络的FSP矩阵，知识就从教师蒸馏到学生。作者的实验证明这种知识迁移方法比FitNet的好。

![](https://pic4.zhimg.com/80/v2-bfcb40b4ffaaeca1bf0386a34dd32933_720w.webp)

###  Contrastive Representation Distillation
Tian等人的工作[13]指出原始KD适合网络输出为类别分布的情况，不适合跨模态蒸馏(cross-modal distillation)的情况，如将图像处理网络(处理RGB信息)的表示/特征迁移到深度处理网络(处理depth信息)，因为这种情况是未定义KL散度的。作者利用对比目标函数(contrastive objectives)去捕捉结构化特征知识的相关性和高阶输出依赖性。contrastive learning简而言之就是，学习一个表示，正配对在某metric space上离得近些，负配对离得远些。CRD适用于如图5的三种具体应用场景中，其中模型压缩和集成蒸馏是原始KD适用的任务。这里说一些题外话，熟悉域适应的同学，肯定知道像素级适应(CycleGAN)，特征级适应和输出空间适应(AdaptSegNet)是提升模型适应目标域数据的三个角度。原始KD就是输出空间蒸馏，CRD就是特征蒸馏，两者可以叠加适使用。

![](https://pic1.zhimg.com/80/v2-cb3bba5867ef7935077dc919dbb184f4_720w.webp)

### Teacher Assistant Knowledge Distillation
Mirzadeh S-I等人的工作[14]指出KD并非总是effective。当教师网络与学生网络的模型大小差距太大的时候，KD会失效，学生网络的性能会下降【这一点需要特别注意】。作者在教师网络和学生网络之间，引入助教网络，如图6所示。TAKD的原理，简单而言，教师网络和助教网络之间进行知识蒸馏，助教网络和学生网络之间再进行知识蒸馏，即多步蒸馏(multi-step knowledge distillation )。

![](https://pic3.zhimg.com/80/v2-a489866cbb1a0bfd4ab56cbd78ff5132_720w.webp)

### On the Efficacy of Knowledge Distillation 
Cho和Hariharan的工作[15]关注KD的有效性，得到的结论是：教师网络精度越高，并不意味着学生网络精度越高。这个结论和Mirzadeh S-I等人的工作[14]是一致的。mismatched capacity使得学生网络不能稳定地模仿教师网络。有趣的是，Cho和Hariharan的工作认为上述TAKD的多步蒸馏并非有效，提出应该采取的措施是提前停止教师网络的训练。

###  A Comprehensive Overhaul of Feature Distillation
 Heo等人的工作[16]关注特征蒸馏，这区别于Hinton等人的工作：暗知识或输出蒸馏。隐层特征值/中间表示蒸馏从FitNets开始。这篇关注特征蒸馏的论文迁移两种知识，第一种是after ReLU之后的特征响应大小(magnitude)；第二种是每一个神经元的激活状态(activation status)。以ResNet50为学生网络，ResNet152为教师网络，使用作者的特征蒸馏方法，学生网络ResNet50(student)从76.16提升到78.35，并超越教师网络ResNet152的78.31(Top-1 in ImageNet)。另外，这篇论文在通用分类、检测和分割三大基础任务上进行了实验。Heo等人先前在AAAI2019上提出基于激活边界(activation boundaries)的知识蒸馏[17]

 ###  Distilling Knowledge from a Deep Pose Regressor Network
 Saputra等人的工作[18]对用于回归任务的网络进行知识蒸馏有一定的实践指导价值。

 ### Route Constrained Optimization 
 Jin和Peng等人的工作[19]受课程学习(curriculum learning)启发，并且知道学生和老师之间的gap很大导致蒸馏失败，提出路由约束提示学习(Route Constrained Hint Learning)，如图7所示。简单而言，我们训练教师网络的时候，会有一些中间模型即checkpoints，RCO论文称为anchor points，这些中间模型的性能是不同的。因此，学生网络可以一步一步地根据这些中间模型慢慢学习，从easy-to-hard。另外，这篇论文在开集人脸识别数据集MegaFace上做了实验，以0.8MB参数，在1：10^6任务上取得84.3%准确率。
 
 ![](https://pic2.zhimg.com/80/v2-ea44c630017868431278dd5339884e81_720w.webp)

 ### Architecture-aware Knowledge Distillation
  Liu等人的工作[20]指出给定教师网络，有最好的学生网络，即不仅对教师网络的权重进行蒸馏，而且对教师网络的结构进行蒸馏从而得到学生网络。(Rethinking the Value of Network Pruning这篇文章也指出剪枝得到的网络可以从头开始训练，取得不错的性能，表明搜索/剪枝得到的网络结构是挺重要的。

  ### Towards Understanding Knowledge Distillation

Phuong和Lampert的工作[21]研究线性分类器(层数等于1)和深度线性网络(层数大于2)，回答了学生模型在学习什么？(What Does the Student Learn?)，学生模型学习得多？(How Fast Does the Student Learn?)，并从Data Geometry，Optimiation Bias，Strong Monotonicity(单调性)三个方面解释蒸馏为什么工作？(Why Does Distillation Work)。

### Unifying Distillation and Privileged Information
Lopez-Paz等人的工作[22]联合了两种使得机器之间可以互相学习的技术：蒸馏和特权信息，提出新的框架广义蒸馏(generalized distillation)。是一篇从理论上解释蒸馏的论文。

### Born-Again Neural Networks
BANs不算模型压缩，因为学生模型容量与教师模型容量相等。但是Furlanello等人[23]发现以KD的方式去训练模型容量相等的学生网络，会超越教师网络，称这样的学生网络为大师网络。这一发现真让人惊讶！与Hinton理解暗知识不同，BANs的作者Furlanello等人，认为软目标分布起作用，是因为它是重要性采样权重。

### Apprentice
学徒[24]利用KD技术提升低精度网络的分类性能，结合量化和蒸馏两种技术。引入量化，那么教师网络就是高精度网络，并且带非量化参数，而学徒网络就是低精度网络，带量化参数。

### Model Compression via Distillation and Quantization
希望获得浅层和量化的小型网络，结合量化和蒸馏两种技术。

### Structured Knowledge Distillation
Liu等人的工作[26-27]整合了暗知识的逐像素蒸馏(pixel-wise distillation)、马尔科夫随机场的逐配对/特征块蒸馏(pair-wise distillation)和有条件生成对抗网络的整体蒸馏(holistic distillation)，如图8所示，用于密集预测任务：语义分割、深度估计和物体检测。
![](https://pic4.zhimg.com/80/v2-00fa3e3a6addd7f04001121c4917280f_720w.webp)