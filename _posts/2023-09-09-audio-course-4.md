---
tags: Transformer
---
# Audio course
## 造一个音乐流派分类器

目标

- 找到合适声音分类预训练模型
- 使用 HF 选择合适的数据集
- 微调预训练模型
- 写个 Gradio demo

### 找合适模型
#### 安装 Transformers

```python
pip install git+https://github.com/huggingface/transformers
```

#### keyword spotting(KWS)
口语识别关键字任务

Minds-14 数据集

```python
from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
```

加载检测点"anton-l/xtreme_s_xlsr_300m_minds14"

```python

from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
```

看一下例子

```python
classifier(minds[0]["path"])
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

Speech Commands 数据集

评估简单命令单词上的音频分类模型的口语单词的数据集。该数据集由15类关键字，沉默的类和一个未知类别组成

加载

```python
speech_commands = load_dataset(
    "speech_commands", "v0.02", split="validation", streaming=True
)
sample = next(iter(speech_commands))
```

我们将在 Speech Commands 数据集上微调的官方Audio Spectrogram Transformer  检查点

```python
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2"
)
classifier(sample["audio"])
```

输出    

```
[{'score': 0.9999892711639404, 'label': 'backward'},
 {'score': 1.7504888774055871e-06, 'label': 'happy'},
 {'score': 6.703040185129794e-07, 'label': 'follow'},
 {'score': 5.805884484288981e-07, 'label': 'stop'},
 {'score': 5.614546694232558e-07, 'label': 'up'}]
 ```

 6,看起来 backward 概率最高。确认一下

```python
from IPython.display import Audio

classifier(sample["audio"].copy())
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```

#### 语言识别

语言识别（LID）是从候选语言列表中识别音频样本中使用的语言的任务

FLEURS 数据集

是用于评估102种语言语音识别系统的数据集，其中包括许多被归类为“低资源”

加载
```python
fleurs = load_dataset("google/fleurs", "all", split="validation", streaming=True)
sample = next(iter(fleurs))
```

加载分类模型

```
classifier = pipeline(
    "audio-classification", model="sanchit-gandhi/whisper-medium-fleurs-lang-id"
)
```

丢进去一个预测

```python
classifier(sample["audio"])
```

输出

```
[{'score': 0.9999330043792725, 'label': 'Afrikaans'},
 {'score': 7.093023668858223e-06, 'label': 'Northern-Sotho'},
 {'score': 4.269149485480739e-06, 'label': 'Icelandic'},
 {'score': 3.2661141631251667e-06, 'label': 'Danish'},
 {'score': 3.2580724109720904e-06, 'label': 'Cantonese Chinese'}]
 ```

 #### 零样本语音分类

 支持模型，CLAP

 举个 ESC 数据集的例子

 ```python
 dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
audio_sample = next(iter(dataset))["audio"]["array"]
```

```python
candidate_labels = ["Sound of a dog", "Sound of vacuum cleaner"]
```

找到最适合的候选标签

```python
classifier = pipeline(
    task="zero-shot-audio-classification", model="laion/clap-htsat-unfused"
)
classifier(audio_sample, candidate_labels=candidate_labels)
```

输出

```
[{'score': 0.9997242093086243, 'label': 'Sound of a dog'}, {'score': 0.0002758323971647769, 'label': 'Sound of vacuum cleaner'}]
```

听声音确认一下

```python
Audio(audio_sample, rate=16000)
```

6 ,就是狗叫

### 微调音乐分类模型

这里手把手教你咋微调 只用编码器的transformer 模型去做音乐分类

这里只用轻量小模型配合小数据集，你可以在任意端到端的消费级别GPU上玩，包括colab的T4

#### 数据集
GTZAN 数据集（包含1000首歌，10种，30s切片）

```python
from datasets import load_dataset

gtzan = load_dataset("marsyas/gtzan", "all")
gtzan
```

输出

```
Dataset({
    features: ['file', 'audio', 'genre'],
    num_rows: 999
})
```

这个数据集没有提供验证集，我们得自己建一个。 9/1 分配

```
gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
gtzan
```

输出

```
DatasetDict({
    train: Dataset({
        features: ['file', 'audio', 'genre'],
        num_rows: 899
    })
    test: Dataset({
        features: ['file', 'audio', 'genre'],
        num_rows: 100
    })
})
```

大概看一下

```python
gtzan["train"][0]
```

输出

```
{
    "file": "~/.cache/huggingface/datasets/downloads/extracted/fa06ce46130d3467683100aca945d6deafb642315765a784456e1d81c94715a8/genres/pop/pop.00098.wav",
    "audio": {
        "path": "~/.cache/huggingface/datasets/downloads/extracted/fa06ce46130d3467683100aca945d6deafb642315765a784456e1d81c94715a8/genres/pop/pop.00098.wav",
        "array": array(
            [
                0.10720825,
                0.16122437,
                0.28585815,
                ...,
                -0.22924805,
                -0.20629883,
                -0.11334229,
            ],
            dtype=float32,
        ),
        "sampling_rate": 22050,
    },
    "genre": 7,
}
```

转换一下

```python
id2label_fn = gtzan["train"].features["genre"].int2str
id2label_fn(gtzan["train"][0]["genre"])
```

输出

```
'pop'
```

看起来标签没毛病，再用 Gradio 的API听几个试试

```python
import gradio as gr


def generate_audio():
    example = gtzan["train"].shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label_fn(example["genre"])


with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)
```

#### 选模型

DistilHuBERT

加载数据

```python
from transformers import AutoFeatureExtractor

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)
```

改一下采样率

```python
sampling_rate = feature_extractor.sampling_rate
sampling_rate
```

输出

```
16000
```

```python
from datasets import Audio

gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))
```

查看一下

```
gtzan["train"][0]
```

输出

```
{
    "file": "~/.cache/huggingface/datasets/downloads/extracted/fa06ce46130d3467683100aca945d6deafb642315765a784456e1d81c94715a8/genres/pop/pop.00098.wav",
    "audio": {
        "path": "~/.cache/huggingface/datasets/downloads/extracted/fa06ce46130d3467683100aca945d6deafb642315765a784456e1d81c94715a8/genres/pop/pop.00098.wav",
        "array": array(
            [
                0.0873509,
                0.20183384,
                0.4790867,
                ...,
                -0.18743178,
                -0.23294401,
                -0.13517427,
            ],
            dtype=float32,
        ),
        "sampling_rate": 16000,
    },
    "genre": 7,
}
```
很好，下采样到16kHz了。

看一下特征提取器

```python
import numpy as np

sample = gtzan["train"][0]["audio"]

print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")
```

输出

```
Mean: 0.000185, Variance: 0.0493
```

方差再大点才明显

```python
inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])

print(f"inputs keys: {list(inputs.keys())}")

print(
    f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}"
)
```

输出

```
inputs keys: ['input_values', 'attention_mask']
Mean: -4.53e-09, Variance: 1.0
```

当我们一次处理一批音频输入时，请使用active_mask - 用于告诉模型我们在哪里有不同长度的填充输入。

裁剪一下特征

```python
max_duration = 30.0


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs
```

map()一下，batched设置100

```python
gtzan_encoded = gtzan.map(
    preprocess_function,
    remove_columns=["audio", "file"],
    batched=True,
    batch_size=100,
    num_proc=1,
)
gtzan_encoded
```

输出

```
DatasetDict({
    train: Dataset({
        features: ['genre', 'input_values','attention_mask'],
        num_rows: 899
    })
    test: Dataset({
        features: ['genre', 'input_values','attention_mask'],
        num_rows: 100
    })
})
```

如果 RAM 不够，尝试将2倍降低到50 *`writer_batch_size`：默认为1000。尝试将其减少到500，如果不起作用，请再次将其减少2倍至250倍

简化训练，移除 audio 和 file 列，重命名 genre列为label

```python
gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
```

获取一下标签

```python
id2label = {
    str(i): id2label_fn(i)
    for i in range(len(gtzan_encoded["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}

id2label["7"]
```
```
'pop'
```
#### 微调模型

```python
from transformers import AutoModelForAudioClassification

num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
```

这里建议在训练时直接上传检查点
毕竟提供
- 版本控制
- tensorboard logs
- 模型卡
- 社区交流

在notebook上直接连接很容易
```python
from huggingface_hub import notebook_login

notebook_login()
```

输出

```
Login successful
Your token has been saved to /root/.huggingface/token
```

定义超参数

```python
from transformers import TrainingArguments

model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10

training_args = TrainingArguments(
    f"{model_name}-finetuned-gtzan",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
)
```

设置 push_to_hub=True 来自动上传检查点

定义评估矩阵

```python
import evaluate
import numpy as np

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

搞定 ，来训练吧

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
```

爆显存记得调bs

提交到 leaderboard 排个名

```python
kwargs = {
    "dataset_tags": "marsyas/gtzan",
    "dataset": "GTZAN",
    "model_name": f"{model_name}-finetuned-gtzan",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}
```
```python
trainer.push_to_hub(**kwargs)
```

#### 分享模型
跟找模型过程一样


### 用 Gradio 写个小例子

先加载一下
```python
from transformers import pipeline

model_id = "sanchit-gandhi/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)
```

定义输入路径

```python
def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs
```

发布例子

```python
import gradio as gr

demo = gr.Interface(
    fn=classify_audio, inputs=gr.Audio(type="filepath"), outputs=gr.outputs.Label()
)
demo.launch(debug=True)
```
