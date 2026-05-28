# LLM 量化落地项目复现计划：llama.cpp + Qwen GGUF

## 1. 为什么重做方向

前面自己手写 mock 推理、简单 benchmark，对理解概念有一点帮助，但不够形成招聘要求里的“实际项目落地案例”。

岗位要求明确写了：

```text
熟悉模型量化技术，有实际项目落地案例。
熟悉 C/C++、CUDA、TensorRT、cuDNN、NCCL、Triton、ONNX Runtime 等优先。
熟悉大规模模型部署经验者优先。
```

所以现在项目目标改成：

```text
复现成熟开源 LLM 量化推理项目，并做出自己的实验报告。
```

## 2. 选型：为什么选 llama.cpp

项目：

```text
ggml-org/llama.cpp
```

选择原因：

1. 它是成熟的 C/C++ LLM 推理项目。
2. 支持 GGUF 模型格式。
3. 自带量化工具。
4. 支持 CPU 推理，也支持 CUDA 等后端。
5. 支持 server 模式，能做本地推理服务。
6. Qwen 官方文档中也使用 llama.cpp 演示 GGUF 量化与推理。

这个项目比我们自己写 toy pipeline 更接近岗位实际要求。

## 3. 最终要做出的项目成果

最终不是“我跑了一下 llama.cpp”，而是要形成一个完整的量化落地案例：

```text
Qwen 小模型
-> GGUF 格式
-> Q8_0 / Q4_K_M 量化
-> llama.cpp CPU/GPU 推理
-> 性能 benchmark
-> 模型大小对比
-> tokens/sec 对比
-> 输出质量对比
-> 可选 perplexity 评估
-> 可选 server 部署
```

最后产出：

```text
docs/quantization_reproduction_report.md
```

报告必须包含：

- 复现环境；
- 模型来源；
- 量化方式；
- 模型大小对比；
- 推理速度对比；
- 内存/显存占用；
- 输出样例；
- 量化 trade-off；
- 踩坑记录；
- 面试表达。

## 4. 计划复现的成熟流程

### Phase 1：拉取并构建 llama.cpp

目标：

```text
third_party/llama.cpp
```

构建方式优先级：

1. Windows + CMake + MSVC；
2. 如果本机 CUDA 编译环境可用，尝试 CUDA 后端；
3. 如果 CUDA 构建复杂，先 CPU 构建跑通；
4. 后续再补 CUDA 构建。

验证标准：

```powershell
llama-cli --help
llama-quantize --help
llama-server --help
```

### Phase 2：准备小模型

优先选择小模型，避免一开始被显存和下载卡死。

候选：

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

如果 Hugging Face 下载不稳定：

- 使用镜像；
- 手动下载；
- 使用已经转换好的 GGUF；
- 先用更小模型验证流程。

### Phase 3：跑通 FP16 / Q8_0 / Q4_K_M

重点量化格式：

```text
Q8_0
Q4_K_M
```

实验目标：

- 对比模型文件大小；
- 对比推理速度；
- 对比输出质量；
- 对比内存/显存占用。

### Phase 4：Benchmark

固定 prompt：

```text
请用简洁语言解释什么是 KV cache，以及它为什么能加速 LLM 推理。
```

固定输出长度：

```text
128 tokens
```

记录指标：

- load time；
- prompt eval time；
- eval time；
- tokens/sec；
- total time；
- RAM / VRAM；
- CPU/GPU 参数。

### Phase 5：imatrix 校准量化

进阶目标：

复现 llama.cpp / Qwen 文档中的 imatrix 流程。

目的：

```text
用校准数据改善低 bit 量化质量。
```

要解释清楚：

- imatrix 是什么；
- 为什么普通 Q4 和带 imatrix 的 Q4 质量不同；
- 校准数据如何影响量化。

### Phase 6：server 部署

用 llama.cpp 的 server 模式启动本地推理服务。

目标：

- 本地 HTTP 服务；
- curl 调用；
- 记录请求延迟；
- 和 CLI 推理对比；
- 为后续 FastAPI / Triton 做铺垫。

## 5. 当前阻塞

尝试克隆：

```powershell
git clone https://github.com/ggml-org/llama.cpp.git third_party\llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git third_party\llama.cpp
```

结果：

```text
Failed to connect to github.com port 443
```

说明当前环境无法直连 GitHub。

下一步解决方式：

1. 换网络或代理；
2. 使用 GitHub zip 下载；
3. 使用镜像源；
4. 手动把 llama.cpp 压缩包放到 `third_party/llama.cpp`；
5. 如果你同意，我可以继续尝试可用镜像。

## 6. 这个项目对简历的表达

最终可以写成：

```text
复现并扩展 llama.cpp 的 Qwen GGUF 量化推理流程，完成 FP16、Q8_0、Q4_K_M 多种量化格式对比，评估模型大小、推理速度、内存/显存占用与输出质量，并部署本地 OpenAI-compatible 推理服务。
```

英文版：

```text
Reproduced and extended a llama.cpp-based Qwen GGUF quantization workflow, comparing FP16, Q8_0 and Q4_K_M models on file size, inference latency, tokens/sec, memory usage and output quality, with a local inference server deployment.
```

这才接近岗位要求中的：

```text
模型量化技术实际项目落地案例
```

## 7. 下一步

优先解决 llama.cpp 获取问题。

一旦拿到源码，立即执行：

1. 构建 llama.cpp；
2. 下载或准备 Qwen GGUF 小模型；
3. 跑通 baseline 推理；
4. 跑通 Q8_0 / Q4_K_M 量化；
5. 写第一版量化对比报告。

