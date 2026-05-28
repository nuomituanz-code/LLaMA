# LLaMA / LLM Quantization Reproduction Lab

这个仓库用于复现和整理一个面向 AI Infra / LLM Infra 岗位的项目：

```text
llama.cpp + Qwen GGUF 量化推理复现
```

目标不是做 toy demo，而是形成一个能写进简历的模型量化落地案例。

## 项目目标

围绕成熟开源项目 `llama.cpp`，复现 LLM 量化与本地推理流程：

- 构建 `llama.cpp`
- 准备 Qwen GGUF 小模型
- 对比 FP16 / Q8_0 / Q4_K_M
- 记录模型大小、推理速度、内存/显存占用
- 跑通本地推理服务
- 输出量化复现实验报告

## 当前文档

- [LLM 量化复现计划](docs/llm_quantization_reproduction_plan.md)
- [学习笔记](docs/week_notes.md)
- [AI Infra 学习路线](docs/ai_infra_engineer_learning_plan.md)

## 本地目录说明

```text
docs/       项目计划、学习笔记、实验报告
cuda/       后续 CUDA kernel 学习实验
cpp/        后续 C/C++ 推理相关实验
triton/     后续 Triton Inference Server 部署实验
third_party/ 本地克隆的第三方项目，不提交到 Git
```

## 当前状态

`llama.cpp` 已通过镜像源克隆到本地：

```text
third_party/llama.cpp
```

下一步是构建 `llama.cpp`，确认以下工具可用：

```text
llama-cli
llama-quantize
llama-server
llama-bench
```

