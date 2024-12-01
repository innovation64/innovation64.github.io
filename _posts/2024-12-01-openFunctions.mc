---
tags: LLM
---

# 伯克利实验室系列
## Gorilla OpenFunctions
OpenFunctions 旨在扩展大型语言模型（LLM）的 Chat Completion 功能，以便根据自然语言指令和 API 上下文生成可执行的 API 调用。想象一下，如果LLM能够为 Instagram、DoorDash 等众多服务以及 Google 日历和 Stripe 等工具填充参数，会怎样。即使是那些不太熟悉 API 调用程序和编程的用户，也可以使用该模型生成所需功能的 API 调用。Gorilla OpenFunctions 是我们使用经过精选的 API 文档和从 API 文档生成的问答对进行训练的LLM。我们一直在扩展 Gorilla 范式，并寻求提高有效函数调用生成的质量和准确性。本文探讨了开发一个开源的函数调用替代方案，类似于在专有模型中看到的特性，特别是类似于 OpenAI 的 GPT-4 中的函数调用。我们的解决方案基于 Gorilla 配方，并且仅使用 7B 参数的模型，其准确性出人意料地与 GPT-4 相当。

## 如何使用 OpenFunctions
1. 定义您的函数：提供一个包含自定义函数描述的 JSON 文件。每个函数应包含以下字段： name （API 名称）、 api_call （如何调用此 API）、 description （描述 API 的功能）以及最后， parameters （与 API 调用相关的参数列表）。以下是 API 文档示例，可用于 OpenFunctions。

2. Install the openai client with pip install openai==0.28

```json

    function_documentation = {  
        "name" : "Order Food on Uber",
        "api_call": "uber.eat.order",
        "description": "Order food on uber eat given a list of items and the quantity of items respectively",
        "parameters": 
            [
                {
                    "name": "restaurants", 
                    "description": "The restaurants user wants to order from" 
                }, 
                {
                    "name": "items", 
                    "description": "A list of order user wants to order from restaurants"
                },
                {
                    "name": "quantities", 
                    "description": "A list of quantities corresponding to the items ordered"
                }
            ]
        }
```
3. 请提出您的问题：就像与另一个人交谈一样描述您想要的内容。
I want to order five burgers and six chicken wings from McDonald.

4. 获取您的函数调用：根据您的请求，模型将返回一个 Python 函数调用。
这为开发者和非开发者 alike 提供了可能性，使他们能够在不编写大量代码的情况下利用复杂的功能。

这为开发者和非开发者 alike 提供了可能性，使他们能够在不编写大量代码的情况下利用复杂的功能。
```python
get_gorilla_response(prompt="I want to order five burgers and six chicken wings from McDonald.", 
                         functions=[function_documentation])
```
输出
```python
uber.eat.order(restaurants="McDonald",item=["chicken wings", "burgers"], quantity=[6,5])
```

## OpenFunctions Performance Benchmarking

我们正在将我们的模型与当前最先进的模型 GPT-4-0613 以及 GPT-4 和 GPT-3.5-turbo 的功能调用特性进行比较。我们的测试数据集由 116 个独特的查询和 API 文档对组成，这些对是通过向 GPT-3.5-turbo 输入少量示例并要求模型从不同领域生成 API（包括旅行、金融、会议安排）而精心制作的。

令人惊讶的是，我们发现 GPT-4 和 GPT-3.5 在函数调用方面的表现优于专为函数调用定制的最先进的 GPT-4-Turbo 和 GPT-4 模型。我们的 OpenFunctions 模型紧随其后，差距微小。

为了评估输出质量，我们在模型输出和“黄金”答案之间进行了一一对照检查。从上面的图表中，我们可以看到，与 GPT-4 的函数调用相比，GPT-4 和 GPT-3.5-Turbo 返回的函数调用成功率更高，约为 95%。我们的基于 7B 参数的 llama-v2 模型 OpenFunctions 在 GPT-4 函数调用之后，成功率达到了 87.39%。

以下是 GPT-4 生成不满意结果的两个示例：

| GPT4 | GPT-4 |
| --- | --- |
| ```"Query": "Determine the monthly mortgage payment for a loan
        amount of $200,000, an interest rate of 4%, and a 
        loan term of 30 years.",
"GPT-4 output":
    "{"name": "finance.calculate_mortgage_payment",
    "arguments": "{"loan_amount": 200000,
                    "interest_rate": 4,
                    "loan_term": 30}"
                    }",

"Gold answer": "finance.calculate_mortgage_payment(
                    loan_amount=200000, 
                    interest_rate=0.04, 
                    loan_term=30)"
                                ```| ```"Query": "Order me six pack of potato chips and eight 
        pack of chocolate from target near Berkeley.",
"GPT-4 output":
        "{ "name": "target.get",
            "arguments": "{
                "loc": "Berkeley",
                "item": ["six pack of potato chips", 
                "eight pack of chocolate"],
                "quantity": [1, 1]}
                    }",
"Gold answer": "target.order(
                  loc=Berkeley,
                  item=["potato chips", "chocolate"], 
                  quantity=[6,8])"``` |


从上述例子中我们可以看出，即使是 GPT-4 的功能调用也无法保证在函数参数分析中取得令人满意的结果。在这里，我们对测试数据中成功和失败的比例进行了详细分析：

![](https://gorilla.cs.berkeley.edu/assets/img/blog_post_4_OpenFunctions_Mistake.png)

在调用 OpenAI 的函数调用模型时，如果指令中未提供所需参数，这会导致函数调用模型输出“后续问题”以请求所需参数。这导致上述图表中显示的“不完整”状态。我们在准确性计算中将“不完整”执行视为“成功”，因为模型成功识别了缺失的参数。请注意，这一点在所有评估中都是一致的。我们的 OpenFunctions 模型，以及常规的 GPT-4，由于其聊天完成性质，会用占位符或默认值填充所需参数，从而允许无干扰的生成。


## OpenFunctions 数据组成

我们训练模型的数据集由 14,189 个指令-API 对组成。我们从 3 个来源精心挑选了 API 文档

- Python packages: 
- RapidAPI:
- Command line tools from cloud providers:

对于每个 API 文档，我们生成三个不同的指令-API 对作为我们的训练数据。这个指令和“模型”答案对是通过使用正确使用 API 文档的少量示例来自我生成的，以实现准确的函数调用。我们明确提示模型利用复杂值类型等特性，如果特定 API 具有该特性，则使用更多参数。

## 代码功能调用 API 与 REST API

当整理数据集时，我们观察到 API 调用可以大致分为两大类：

- 代码功能调用 API
- REST APIs REST API

首先，代码功能调用 API 通常出现在 Numpy、Sklearn 等外部 Python 包中。这些 API 定义良好，可以轻松格式化。因此，只需知道“api_name”，例如 numpy.sum() ，以及“arguments”规范，我们就可以构建一个可执行的函数 API。由于其稳定的格式和固定的局部性，在训练这些 API 时需要的数据点相对较少。这体现在我们选择训练数据混合的方式上。

RESTful API 占据了互联网上 API 的重要部分，为大多数服务提供动力。这些 API 通常由第三方托管，提供从金融服务到天气预报的各种功能。通常，RESTful API 的元数据中包含三个参数： url ， header 和 params 。URL 包含 API 端点，头部通常包含认证信息，而 params 包含查询 API 端点的 '信息'。使用 requests.get ，我们可以正确查询端点。然而，REST API 的参数可以存在于不同的位置。例如，参数可以嵌入到 URL 中，例如 gorilla.berkeley.edu/{param1}/{param2} 。另一种表示参数嵌入的方法可以是 gorilla.berkeley.edu/?query=param1 。调用 REST API 的不同方法可能使我们的模型难以处理复杂的 REST API 调用。为此，我们依靠不同的 REST API 来源，如 RapidAPI、Postman API 等，以多样化我们的 API 数据库并生成更准确的 REST API。

## 模型和功能！

我们很高兴发布两款模型： gorilla-openfunctions-v0 和 gorilla-openfunctions-v1 。 gorilla-openfunctions-v0 LLM 是在 7B LLaMA-v2-chat 指令微调模型之上训练的 7B 参数模型。它接受用户提示和单个 API 调用，并返回带有正确参数的函数。 gorilla-openfunctions-v1 是在 7B LLaMA-v2 预训练模型之上训练的 7B 参数模型。 gorilla-openfunctions-v1 是我们的高级模型，接受用户提示和多个 API 调用，并返回带有正确参数的函数。它还支持并行函数！ gorilla-openfunctions-v1 目前处于早期预览阶段，您可以在接下来的几天内期待它变得更好！本博客中所有结果均使用 gorilla-openfunctions-v0 生成。

