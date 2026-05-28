# LLM 推理基础

## 1. 为什么先学推理链路

我们要做的项目是：

```text
LLM Inference Engine Lab
```

这个项目的核心不是“调用一个大模型 API”，而是理解并实现一个大模型推理系统的关键链路：

```text
prompt
-> tokenizer
-> input ids
-> model forward
-> logits
-> decoding
-> generated tokens
-> output text
```

后面要做的 API 服务、benchmark、batching、KV cache、量化、TensorRT、Triton，本质上都是围绕这条链路做工程化和性能优化。

---

## 2. Prompt 是什么

`prompt` 是用户输入给模型的文本。

例子：

```text
Explain what KV cache is in LLM inference.
```

人能直接读懂文本，但模型不能直接处理字符串。模型只能处理数字张量。

所以第一步必须把文本变成数字。

---

## 3. Tokenizer 是什么

`tokenizer` 的作用是把文本切成 token，并把 token 映射成整数 id。

例子：

```text
"Hello world"
```

可能会被切成：

```text
["Hello", " world"]
```

然后映射成：

```text
[15496, 995]
```

不同模型有不同 tokenizer。GPT、BERT、LLaMA、Qwen 的 tokenizer 规则可能不同。

### 为什么 tokenizer 很重要

因为 LLM 的上下文长度、显存占用、推理耗时，都和 token 数量直接相关。

同样一句话：

- token 越多，计算越多；
- prompt 越长，prefill 越慢；
- 生成越长，decode 越久；
- KV cache 占用越多显存。

所以做 LLM Infra，不能只看字符串长度，要看 token 数量。

---

## 4. Input IDs 是什么

`input_ids` 是 tokenizer 输出的 token id 序列。

模型真正接收的是类似这样的张量：

```text
input_ids = [[15496, 995]]
```

形状通常是：

```text
[batch_size, sequence_length]
```

例如：

```text
[1, 128]
```

表示：

- batch size 是 1；
- prompt 长度是 128 个 token。

如果是 batch 推理：

```text
[8, 128]
```

表示一次处理 8 条请求，每条长度约 128 token。

---

## 5. Attention Mask 是什么

batch 推理时，不同 prompt 长度可能不同。

为了凑成一个矩阵，短句子通常会 padding。

例子：

```text
prompt A: [10, 20, 30, 40]
prompt B: [11, 22]
```

padding 后：

```text
prompt A: [10, 20, 30, 40]
prompt B: [11, 22,  0,  0]
```

`attention_mask` 用来告诉模型哪些位置是真 token，哪些位置是 padding。

```text
prompt A mask: [1, 1, 1, 1]
prompt B mask: [1, 1, 0, 0]
```

这对 batch 推理非常重要。

---

## 6. Model Forward 做了什么

LLM 的核心是 Transformer。

输入：

```text
input_ids
attention_mask
```

经过：

```text
embedding
-> transformer layers
-> final normalization
-> lm head
```

输出：

```text
logits
```

简化理解：

```text
模型根据当前 token 序列，预测下一个 token 的概率分布。
```

---

## 7. Logits 是什么

`logits` 是模型输出的原始分数。

假设词表大小是 50,000，模型每次要给这 50,000 个 token 都打一个分数。

如果输入 shape 是：

```text
[batch_size, sequence_length]
```

那么 logits shape 通常是：

```text
[batch_size, sequence_length, vocab_size]
```

生成下一个 token 时，我们通常只关心最后一个位置的 logits：

```text
logits[:, -1, :]
```

它表示：

```text
基于当前整段 prompt，下一个 token 应该是什么。
```

---

## 8. Decoding 是什么

`decoding` 是把 logits 变成下一个 token id 的过程。

常见策略：

### Greedy Decoding

每次选择分数最高的 token。

优点：

- 稳定；
- 可复现；
- 实现简单。

缺点：

- 输出可能死板；
- 容易重复。

### Sampling

按概率随机采样 token。

常见参数：

- `temperature`
- `top_k`
- `top_p`

优点：

- 输出更丰富；
- 更适合创意任务。

缺点：

- 不稳定；
- 更难复现；
- benchmark 时要固定 seed。

---

## 9. 自回归生成是什么

GPT / LLaMA / Qwen 这类模型通常是自回归生成。

意思是：

```text
一次只生成一个 token。
```

流程：

```text
输入 prompt
-> 预测第 1 个新 token
-> 把第 1 个新 token 拼回输入
-> 预测第 2 个新 token
-> 再拼回输入
-> 重复直到达到停止条件
```

停止条件可能是：

- 达到 `max_new_tokens`；
- 生成 EOS token；
- 命中特定 stop words；
- 请求超时。

---

## 10. Prefill 和 Decode

LLM 推理可以分成两个阶段：

```text
prefill
decode
```

### Prefill

prefill 是处理用户完整 prompt 的阶段。

如果 prompt 有 1024 个 token，模型要一次性处理这 1024 个 token。

特点：

- 计算量和 prompt 长度相关；
- prompt 越长，prefill 越慢；
- 主要影响 time to first token。

### Decode

decode 是逐 token 生成新内容的阶段。

每一步生成一个新 token。

特点：

- 和生成长度相关；
- 每生成一个 token 都要跑一次模型；
- 主要影响 tokens/sec 和总响应时间。

---

## 11. KV Cache 是什么

Transformer 里的 attention 会用到 Key / Value。

自回归生成时，每一步都会用到之前 token 的信息。

如果每生成一个新 token，都重新计算所有历史 token 的 Key / Value，会非常浪费。

KV cache 的作用是：

```text
把之前 token 的 Key / Value 缓存起来，下一步直接复用。
```

### KV Cache 的好处

- 减少重复计算；
- 加快 decode；
- 是 LLM 高效推理的核心机制之一。

### KV Cache 的代价

- 占用显存；
- prompt 越长，占用越大；
- batch 越大，占用越大；
- 生成越长，占用越大。

所以 LLM 推理优化经常是在平衡：

```text
速度
显存
batch size
上下文长度
生成长度
```

---

## 12. Batch 推理为什么复杂

普通模型的 batch 比较简单：

```text
一次输入 N 张图片，一次输出 N 个结果。
```

LLM batch 更复杂，因为：

- 每个 prompt 长度不同；
- 每个请求生成长度不同；
- 有些请求先结束，有些请求还在生成；
- KV cache 大小不同；
- decode 是逐 token 进行的。

这也是 vLLM、sglang、Triton dynamic batching 这类系统有价值的原因。

---

## 13. 后续项目中这些概念怎么落地

后续每个模块都会对应到这些概念：

`model_loader.py`

- 加载 tokenizer；
- 加载模型；
- 设置 dtype；
- 设置 device。

`generation.py`

- 处理 prompt；
- 调用 tokenizer；
- 调用 model.generate；
- 控制 max_new_tokens、temperature、top_k、top_p。

`benchmark.py`

- 统计 latency；
- 统计 tokens/sec；
- 统计 prompt token 数；
- 统计 generated token 数。

`batching.py`

- 实验 batch prompt；
- 观察 padding；
- 比较 batch size 对吞吐和延迟的影响。

`quantization.py`

- 对比不同精度；
- 观察显存变化；
- 观察 latency 变化；
- 观察输出质量变化。

---

## 14. 当前阶段需要记住的核心句子

LLM 推理不是“一次输入，一次输出”。

它是：

```text
先 prefill 处理 prompt，再 decode 逐 token 生成。
```

性能优化的核心是理解：

```text
token 数量
batch size
KV cache
显存
latency
throughput
```

