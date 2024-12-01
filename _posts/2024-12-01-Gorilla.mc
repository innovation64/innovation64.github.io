---
tags: LLM
---

# 伯克利实验室系列
## Gorilla LLM

>这篇文章是 2023 年 10 月由 Lisa DunLap 撰写

Gorilla LLM 可以编写准确的 API 调用，包括 kubernetes, GCP ,AWS, Azure, OPENAI 和其他

Gorilla 是我们用一种检索器感知训练 （retriever-aware training）进行训练，它可以选择执行用户以自然语言指定任务正确的 API 。

同时我们还引入基于 AST 子树算法测量 LLMs 的幻觉

现在 LLMs 生成精确的 API 是不可能的，因为数量巨大且经常更新

Gorilla 将 LLM 与 API 链接起来，系统会接收：为我构建一个用于医学图像的分类器这样的指令，并输出相应的导入和 API 调用，以及过程的一步一步解释

Gorilla 使用  self-instruct fine-tuning and retrieval ，使LLMs能够从使用其 API 和 API 文档表达的大、重叠和不断变化的工具集中准确选择。 随着 API 生成方法的发展，如何评估成为一个问题，因为许多 API 将具有重叠的功能和细微的限制和约束。因此，我们构建了 APIBench，一个包括 Kubernetes、AWS、GCP、Azure、GitHub、Conda、Curl、Sed、Huggingface、Tensorflow、Torch Hub 等 API 的大型语料库，并开发了一个新颖的评估框架，该框架使用子树匹配来检查功能正确性并衡量幻觉。

使用 APIBench 我们训练了 Gorilla，一个 7B 模型具有文档检索功能

## 跟上不断变化的数据

如何解决 API 频繁变更的问题，Gorilla 可以使用两种模式进行推理，零样本和检索。   在检索模式中，检索器首先从 APIZoo 的最新文档中进行检索，其中 APIZoo 是 LLMs 的数据库，在发送到 Gorilla 之前，API 文档与用户提示以及消息“使用此 API 文档作为参考”连接起来

Gorilla 的输出是要调用的 API。检索器感知推理模式使 Gorilla 能够应对 API 的频繁变化！ 我们开源了 APIZoo，并继续致力于开源社区，今天我们发布了额外的约 20k 个精心文档化的 API，包括 KubeCtl、GCP、Azure、AWS、GitHub 等。我们将继续向 APIZoo 添加更多 API，并欢迎社区贡献！

![https://gorilla.cs.berkeley.edu/assets/img/blog_post_1_inference.jpg](https://gorilla.cs.berkeley.edu/assets/img/blog_post_1_inference.jpg)


## 一见钟情：LLMs与 Retrievers 之间不为人知的纽带

如何协调训练工作参考我们之前的工作内容，RAT 
![](https://gorilla.cs.berkeley.edu/assets/img/blog_post_1_result.png)

## 现实字节：当LLMs看到不存在的事物

在 API 生成的情况下，幻觉可以被定义为生成不存在的 API 调用的模型.这里用 AST 测量

## 使用方法

```python

!pip install transformers[sentencepiece] datasets langchain openai==0.28.1 &> /dev/null
from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI(
    openai_api_base="http://zanino.millennium.berkeley.edu:8000/v1",
    openai_api_key="EMPTY",
    model="gorilla-7b-hf-v1",
    verbose=True
)
example = chat_model.predict("I want to translate from English to Chinese")
print(example)
```

another

```python
# Import Chat completion template and set-up variables
!pip install openai==0.28.1 &> /dev/null
import openai
import urllib.parse

openai.api_key = "EMPTY" # Key is ignored and does not matter
openai.api_base = "http://zanino.millennium.berkeley.edu:8000/v1"
# Alternate mirrors
# openai.api_base = "http://34.132.127.197:8000/v1"

# Report issues
def raise_issue(e, model, prompt):
    issue_title = urllib.parse.quote("[bug] Hosted Gorilla: <Issue>")
    issue_body = urllib.parse.quote(f"Exception: {e}\nFailed model: {model}, for prompt: {prompt}")
    issue_url = f"https://github.com/ShishirPatil/gorilla/issues/new?assignees=&labels=hosted-gorilla&projects=&template=hosted-gorilla-.md&title={issue_title}&body={issue_body}"
    print(f"An exception has occurred: {e} \nPlease raise an issue here: {issue_url}")

# Query Gorilla server
def get_gorilla_response(prompt="I would like to translate from English to French.", model="gorilla-7b-hf-v1"):
  try:
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
  except Exception as e:
    raise_issue(e, model, prompt)

```
## example

Translate huandwrite with hf
```python
# Gorilla `gorilla-mpt-7b-hf-v1` with code snippets
# Translation
prompt = "I would like to translate 'I feel very good today.' from English to Chinese."
print(get_gorilla_response(prompt, model="gorilla-7b-hf-v1"))
```

Object dection with huggingface
```python
# Gorilla `gorilla-7b-hf-v1` with code snippets
# Object Detection
prompt = "I want to build a robot that can detecting objects in an image ‘cat.jpeg’. Input: [‘cat.jpeg’]"
print(get_gorilla_response(prompt, model="gorilla-7b-hf-v1"))
```

让我们尝试从Torch Hub中调用API，以获取相同的提示

```python
# Translation ✍ with Torch Hub
prompt = "I would like to translate from English to Chinese."
print(get_gorilla_response(prompt, model="gorilla-7b-th-v0"))
```
随着大猩猩在MPT和Falcon上进行微调，您可以在商业上使用

```python   
# Gorilla with `gorilla-mpt-7b-hf-v0`
prompt = "I would like to translate from English to Chinese."
print(get_gorilla_response(prompt, model="gorilla-mpt-7b-hf-v0"))
```
