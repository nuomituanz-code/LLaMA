# LLM Inference Engine Lab 项目开发学习路线

## 0. 项目定位

这个路线不再围绕 MNIST、普通 CNN/RNN 小练习展开。

我们要做的是一个真正能对齐 AI Infra / LLM Infra 岗位要求的项目：

```text
LLM Inference Engine Lab
```

中文可以理解为：

```text
大模型推理服务与性能优化实验室
```

它不是为了展示“我会写基础工具函数”，而是在做一个真实系统的过程中，自然维护并提升基础工程能力。

## 1. 对齐岗位要求

你给的岗位要求里，核心关键词是：

- LLM 研发经验；
- 熟悉 GPT / BERT / LLaMA 等模型架构；
- PyTorch / TensorFlow / sglang / vLLM / Transformer；
- CUDA 编程；
- GPU 并行计算；
- 内存管理；
- 性能优化；
- 模型量化；
- 分布式技术：DP / TP / PP / EP；
- C/C++、Python；
- Git；
- NVIDIA GPU 架构；
- TensorRT / cuDNN / NCCL；
- Triton / TFServing / ONNX Runtime；
- 大规模模型部署。

所以我们的项目不能只是“训练一个小模型”。

项目应该围绕下面这条主线：

```text
加载 LLM -> 推理服务 -> 性能 benchmark -> 量化优化 -> CUDA kernel 学习 -> ONNX/TensorRT/Triton 部署 -> 分布式推理理解
```

## 2. 最终项目长什么样

最终你要有一个 GitHub 项目，结构类似：

```text
llm_inference_engine_lab/
  README.md
  docs/
    architecture.md
    benchmark_report.md
    quantization_report.md
    cuda_notes.md
    deployment_report.md
    interview_notes.md
  src/
    llm_lab/
      model_loader.py
      generation.py
      batching.py
      benchmark.py
      quantization.py
      api.py
      schemas.py
      metrics.py
  cuda/
    vector_add/
    layernorm/
    matmul_tiling/
  cpp/
    tokenizer_demo/
    onnx_runtime_demo/
  scripts/
    serve.py
    benchmark_latency.py
    benchmark_throughput.py
    export_onnx.py
    quantize_model.py
  configs/
    tiny_llm.json
    qwen_small.json
    llama_like.json
  tests/
  docker/
    Dockerfile
  triton/
    model_repository/
```

最终 README 要能回答：

- 这个系统解决什么问题？
- 支持哪些模型？
- 如何启动推理服务？
- 如何 benchmark latency / throughput？
- 做过哪些量化实验？
- GPU 显存如何变化？
- 你如何理解 batch、KV cache、prefill、decode？
- 后续如何扩展到 vLLM / TensorRT / Triton？

## 3. 推荐主项目题目

### 项目名称

```text
LLM Inference Engine Lab: 本地大模型推理服务与性能优化系统
```

### 项目目标

构建一个小型但完整的大模型推理系统，支持：

- 加载 Hugging Face LLM；
- 本地文本生成；
- FastAPI 推理服务；
- 请求 schema 校验；
- latency / throughput benchmark；
- batch 推理实验；
- KV cache / prefill / decode 概念实验；
- int8 / int4 量化对比；
- ONNX Runtime 或 TensorRT 路线探索；
- Triton Inference Server 部署实验；
- CUDA kernel 学习实验；
- 分布式推理概念文档。

### 为什么这个项目有意义

它直接贴近岗位中的真实能力：

- 不是“会调包跑模型”，而是能解释推理链路；
- 不是“会写 API”，而是能测量延迟、吞吐和显存；
- 不是“知道量化”，而是能做量化前后对比；
- 不是“听过 CUDA”，而是写过基础 kernel 并理解并行计算；
- 不是“背过分布式术语”，而是能画出 DP/TP/PP/EP 的区别和适用场景。

## 4. 项目分阶段路线

### Phase 1：LLM 基础推理链路

**目标：** 跑通一个小模型的本地推理，理解 tokenizer、model、generate。

**建议模型：**

优先使用小模型，避免一开始被显存和下载卡住：

- `Qwen/Qwen2.5-0.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `distilgpt2`
- 如果网络不稳定，先用本地 mock model 设计接口，再补真实模型。

**要学：**

- Transformer 基础结构；
- tokenizer；
- input ids；
- attention mask；
- logits；
- greedy decoding；
- sampling；
- temperature / top-k / top-p；
- prefill 和 decode。

**要做：**

- `src/llm_lab/model_loader.py`
- `src/llm_lab/generation.py`
- `scripts/generate_text.py`
- `docs/architecture.md`

**交付标准：**

- 能从命令行输入 prompt，输出文本；
- 能解释一次推理过程中数据如何流动；
- 文档里画出 tokenizer -> model -> logits -> decode 的链路。

**简历价值：**

这一步还不能单独写简历，但它是后续所有工作的地基。

---

### Phase 2：LLM 推理 API 服务

**目标：** 把本地 LLM 推理封装成可调用服务。

**要学：**

- FastAPI；
- Pydantic schema；
- 服务启动时加载模型；
- `/health`；
- `/generate`；
- `/generate_stream`；
- error handling；
- request id；
- structured logging。

**要做：**

- `src/llm_lab/api.py`
- `src/llm_lab/schemas.py`
- `scripts/serve.py`
- `tests/test_api.py`

**交付标准：**

- 服务启动后只加载一次模型；
- API 支持 prompt、max_new_tokens、temperature；
- 错误输入会返回清晰错误；
- 有测试；
- README 里有 curl 示例。

**简历价值：**

可以表达为：

> Implemented a FastAPI-based LLM inference service with request validation, configurable generation parameters, health checks, and structured logging.

---

### Phase 3：Latency / Throughput Benchmark

**目标：** 不只是“能跑”，而是能测。

**要学：**

- latency；
- p50 / p95 / p99；
- throughput；
- tokens per second；
- time to first token；
- GPU memory；
- warmup；
- concurrency；
- batch size。

**要做：**

- `src/llm_lab/benchmark.py`
- `scripts/benchmark_latency.py`
- `scripts/benchmark_throughput.py`
- `docs/benchmark_report.md`

**实验：**

- 不同 `max_new_tokens`；
- 不同 batch size；
- 不同 prompt 长度；
- 不同并发数；
- CPU vs GPU，如果有 GPU；
- FP32 / FP16 / BF16，如果硬件支持。

**交付标准：**

- 有 benchmark 表格；
- 有图表；
- 能解释瓶颈在哪里；
- 能说出 batch size 增大时 latency 和 throughput 的变化。

**简历价值：**

> Built a benchmark suite for LLM inference measuring p50/p95 latency, throughput, tokens/sec, time-to-first-token, and GPU memory under different batch sizes and prompt lengths.

---

### Phase 4：Batching 与 KV Cache 实验

**目标：** 理解 LLM 推理为什么和普通模型推理不一样。

**要学：**

- prefill；
- decode；
- KV cache；
- batch inference；
- continuous batching 概念；
- vLLM 的 PagedAttention 思想。

**要做：**

- `src/llm_lab/batching.py`
- `docs/kv_cache_and_batching.md`

**实验：**

- 单请求推理；
- 静态 batch 推理；
- 不同 prompt 长度混合 batch；
- 记录显存和耗时；
- 对比 batch size 对吞吐的影响。

**交付标准：**

- 能画出 prefill/decode 的时间线；
- 能解释 KV cache 为什么省计算但占显存；
- 能解释 vLLM 为什么重要。

**简历价值：**

> Analyzed LLM prefill/decode behavior and KV cache memory trade-offs through batching experiments, documenting how batch size and sequence length affect throughput and memory.

---

### Phase 5：模型量化实验

**目标：** 对齐岗位里的“模型量化技术实际项目落地案例”。

**要学：**

- FP32 / FP16 / BF16；
- int8；
- int4；
- weight-only quantization；
- GPTQ / AWQ / bitsandbytes 基础；
- 量化对显存、速度、质量的影响。

**要做：**

- `src/llm_lab/quantization.py`
- `scripts/quantize_model.py`
- `docs/quantization_report.md`

**实验：**

- 原始模型；
- FP16；
- int8；
- int4，如果环境支持；
- 对比显存；
- 对比 latency；
- 对比输出质量。

**交付标准：**

- 有量化前后表格；
- 有显存变化；
- 有 latency 变化；
- 有若干 prompt 输出对比；
- 能解释量化 trade-off。

**简历价值：**

> Evaluated LLM quantization strategies including FP16 and int8/int4 weight-only quantization, comparing memory usage, latency, and generation quality across representative prompts.

---

### Phase 6：CUDA 编程与 GPU 性能基础

**目标：** 不空谈 CUDA，至少写过基础 kernel，并理解 GPU 并行模型。

**要学：**

- thread；
- block；
- grid；
- global memory；
- shared memory；
- coalesced access；
- occupancy；
- warp；
- memory bandwidth；
- basic profiling。

**要做：**

```text
cuda/vector_add/
cuda/reduction/
cuda/matmul_tiling/
cuda/layernorm/
docs/cuda_notes.md
```

**实验：**

- vector add；
- reduction；
- tiled matmul；
- 简化 layernorm；
- 和 PyTorch 对比速度；
- 记录输入规模变化下的耗时。

**交付标准：**

- 能编译运行 CUDA kernel；
- 能解释 block/thread 映射；
- 能说明为什么 matmul 要 tiling；
- 能解释 shared memory 的作用。

**简历价值：**

> Implemented CUDA kernels for vector addition, reduction, tiled matrix multiplication, and simplified LayerNorm to study GPU parallelism, memory access patterns, and performance bottlenecks.

---

### Phase 7：ONNX Runtime / TensorRT / Triton 部署路线

**目标：** 对齐模型服务化框架和 NVIDIA 推理栈。

**要学：**

- ONNX；
- ONNX Runtime；
- TensorRT 基础；
- Triton Inference Server；
- model repository；
- dynamic batching；
- model config；
- HTTP/gRPC inference。

**要做：**

```text
scripts/export_onnx.py
cpp/onnx_runtime_demo/
triton/model_repository/
docs/deployment_report.md
```

**实验：**

- 导出一个小模型到 ONNX；
- 用 ONNX Runtime 推理；
- 如果环境支持，跑 TensorRT；
- 用 Triton 部署一个小模型；
- 测 dynamic batching。

**交付标准：**

- 能说明 PyTorch eager、ONNX Runtime、TensorRT、Triton 的关系；
- 有部署步骤；
- 有 benchmark 对比；
- 有踩坑记录。

**简历价值：**

> Deployed model inference through ONNX Runtime and Triton Inference Server, comparing eager PyTorch inference with optimized serving paths and documenting dynamic batching behavior.

---

### Phase 8：分布式推理与大规模部署理解

**目标：** 对齐 DP / TP / PP / EP，不要求一开始就搭大集群，但要真正理解。

**要学：**

- Data Parallel；
- Tensor Parallel；
- Pipeline Parallel；
- Expert Parallel；
- NCCL；
- all-reduce；
- all-gather；
- reduce-scatter；
- 多 GPU 推理；
- vLLM / DeepSpeed / Megatron-LM 基础思想。

**要做：**

- `docs/distributed_inference_notes.md`
- `docs/nccl_collectives.md`
- 如果有多 GPU，再做最小 demo。

**交付标准：**

- 能画图解释 DP/TP/PP/EP；
- 能解释 NCCL 在多 GPU 通信里的角色；
- 能解释为什么 LLM 推理需要 tensor parallel；
- 能说明单机多卡和多机多卡的差异。

**简历价值：**

> Studied distributed LLM inference strategies including tensor parallelism, pipeline parallelism, expert parallelism, and NCCL collectives, with architecture diagrams and deployment trade-off analysis.

---

## 5. 学习顺序建议

不要一上来冲 CUDA、TensorRT、分布式。顺序应该是：

```text
LLM 本地推理
-> 推理 API
-> benchmark
-> batching / KV cache
-> quantization
-> CUDA kernel
-> ONNX/TensorRT/Triton
-> distributed inference
```

原因：

- 你先要知道 LLM 推理链路是什么；
- 然后才知道服务化要优化哪里；
- 有 benchmark 才能证明优化有效；
- 理解 batching/KV cache 后，vLLM 才不是黑盒；
- 做过量化后，简历上才有“落地案例”；
- CUDA 是底层能力，不应该脱离推理系统孤立学；
- TensorRT/Triton 是部署路径，应该在有模型和 benchmark 后再做；
- 分布式最后做，因为它依赖前面的系统理解。

## 6. 前 14 天具体计划

### Day 1：项目重置与目标确认

- 删除无意义的 mini training platform 文件；
- 保留原始练习脚本作为学习记录；
- 创建本路线文档；
- 明确项目名：`LLM Inference Engine Lab`。

### Day 2：LLM 推理基础概念

- 学 tokenizer、input ids、logits、decode；
- 写 `docs/llm_inference_basics.md`；
- 用伪代码画出生成流程。

### Day 3：创建项目骨架

- 创建 `src/llm_lab/`；
- 创建 `scripts/` 下的生成脚本；
- 创建 `configs/tiny_llm.json`；
- 先支持 mock model，避免卡在模型下载。

### Day 4：接入真实小模型

- 优先尝试 `distilgpt2` 或小型 Qwen；
- 能从命令行输入 prompt 并输出文本；
- 记录模型大小、加载时间、生成时间。

### Day 5：封装 generation 模块

- 把 tokenizer/model/generate 封装成清晰函数；
- 增加简单测试；
- 文档解释参数：temperature、top-k、top-p、max_new_tokens。

### Day 6：FastAPI 服务

- 增加 `/health`；
- 增加 `/generate`；
- 增加 Pydantic schema；
- 用 curl 调通。

### Day 7：服务测试与错误处理

- 测空 prompt；
- 测 max_new_tokens 非法值；
- 测服务启动；
- 写 API 使用文档。

### Day 8：Latency benchmark

- 写 benchmark 脚本；
- 记录 p50 / p95；
- 记录 time to first token，如果实现流式输出。

### Day 9：Throughput benchmark

- 测不同并发；
- 测 tokens/sec；
- 写初版 benchmark report。

### Day 10：Batch 推理实验

- 支持 batch prompt；
- 对比单条循环 vs batch；
- 记录吞吐变化。

### Day 11：KV cache 文档

- 学 prefill/decode/KV cache；
- 画图；
- 用实验数据解释显存变化。

### Day 12：量化准备

- 学 FP16/int8/int4；
- 调研 bitsandbytes / transformers quantization；
- 写 `docs/quantization_plan.md`。

### Day 13：运行第一个量化实验

- 尝试 FP16；
- 如果环境支持，再尝试 int8；
- 记录显存和延迟。

### Day 14：阶段复盘

- 整理 README；
- 整理 benchmark report；
- 写第一版项目简历 bullet；
- 列出下一阶段 CUDA/Triton 学习计划。

## 7. 简历最终目标

最终项目可以写成：

> Built an LLM inference engine lab with Hugging Face model loading, FastAPI serving, latency/throughput benchmarking, batching experiments, KV cache analysis, quantization evaluation, CUDA kernel studies, and deployment exploration with ONNX Runtime/Triton.

中文版：

> 构建了一个 LLM 推理与性能优化实验项目，覆盖 Hugging Face 模型加载、FastAPI 推理服务、延迟/吞吐 benchmark、batching 实验、KV cache 分析、量化评估、CUDA kernel 学习以及 ONNX Runtime/Triton 部署探索。

这才是能对齐岗位要求的项目。

