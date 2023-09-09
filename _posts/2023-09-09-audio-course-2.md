---
tags: Transformer
---
# Audio course
## 声音应用

Transformers 可以处理的音频任务
- 语言分类
- 自动语音识别
- 语义化分
- 文本转语音

### 使用 pipeline 处理语音分类

继续使用 MIDNS-14，里面有 intent_class。
继续使用澳大利亚子集并上采样

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

使用 aduio-classification pipeline。这里我们需要已经微调好的 intent 分类模型，并且是针对 MINDS-14的。

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
```

pipeline 会自动处理声音数据为一个 NUmpy array。

```python
example = minds[0]
```

直接传给分类器

```python
classifier(example["audio"]["array"])
```

输出

```
[
    {"score": 0.9631525278091431, "label": "pay_bill"},
    {"score": 0.02819698303937912, "label": "freeze"},
    {"score": 0.0032787492964416742, "label": "card_issues"},
    {"score": 0.0019414445850998163, "label": "abroad"},
    {"score": 0.0008378693601116538, "label": "high_value_payment"},
]
```

让我们看看例子的实际标签

```python
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])
```

输出

```
"pay_bill"
```

### 使用 pipeline 处理语音识别。

ASR 通常包括把语音转文字。

automatic-speech-recognition pipeline

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition")
```

下一步拿一个例子并传递原始数据给 pipeline

```python
example = minds[0]
asr(example["audio"]["array"])
```

输出

```
{"text": "I WOULD LIKE TO PAY MY ELECTRICITY BILL USING MY COD CAN YOU PLEASE ASSIST"}
```

和真实脚本比较

```python
example["english_transcription"]
```

输出

```
"I would like to pay my electricity bill using my card can you please assist"
```

只识别错了一个 card。6 

pipeline 默认的英语，换语言传参数给pipeline

以德语为例

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

看下效果

```python
example = minds[0]
example["transcription"]
```

输出

```
"ich möchte gerne Geld auf mein Konto einzahlen"
```

对比

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="maxidl/wav2vec2-large-xlsr-german")
asr(example["audio"]["array"])
```

输出

```
{"text": "ich möchte gerne geld auf mein konto einzallen"}
```

6
看一下好处
- 预训练模型会节省时间
- pipeline会帮你处理格式
- 结果不理想给了基线可以快速微调
- 一键共享，共建社区