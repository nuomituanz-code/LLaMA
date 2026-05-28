# AI Infra 工程师学习与项目计划

> 这份计划按 Superpowers 的方式组织：目标清楚、任务具体、项目优先、每阶段都有可验证成果，并且始终服务于找工作。

**总目标：** 帮你从刚入门的小白，逐步成长为可以求职的 AI Infra 工程师。

**目标岗位：** AI Infra 工程师、LLM 平台工程师、ML Systems 工程师、MLOps 工程师、模型推理工程师、AI 方向平台工程师。

**核心策略：** 不走“只看课不做项目”的路线。每学一个知识点，都要落到代码、测试、部署、性能数据、文档和简历项目上。

---

## 1. AI Infra 工程师到底要会什么

AI Infra 工程师不是只会训练模型的人，而是负责让 AI 系统变得稳定、快速、可复现、可部署、可监控、可扩展、成本可控的人。

你需要逐步掌握 5 类能力：

1. **Python 工程基础**
   - 写清晰、可维护的 Python 代码。
   - 会使用 Git、虚拟环境、测试、日志、配置文件、命令行工具。
   - 能看懂报错，并且有条理地调试问题。

2. **深度学习系统基础**
   - 理解 tensor、自动求导、Dataset、DataLoader、训练循环、GPU、显存、batch、checkpoint。
   - 能比较熟练地使用 PyTorch。

3. **模型服务与推理系统**
   - 把训练好的模型封装成 API 服务。
   - 支持批量推理、流式输出、缓存、队列、异步处理、健康检查。
   - 能测量延迟、吞吐、内存占用和成本。

4. **MLOps 与平台基础设施**
   - 用 Docker 打包模型和服务。
   - 会做实验记录、模型版本管理、CI 自动测试、部署脚本、监控。
   - 理解 Kubernetes 的基本部署和排错方式。

5. **LLM / RAG 基础设施**
   - 会构建文档检索、向量索引、重排序、提示词模板、评测数据集和可观测性。
   - 能评估质量、延迟和成本。

---

## 2. 每周学习工作流

建议节奏：

- **每周 5 天，每天 2 到 4 小时。**
- **60% 时间做项目，25% 时间补知识，15% 时间写笔记和复盘。**
- 每次学习都要产出一个具体东西：
  - 一个通过的测试；
  - 一个能运行的脚本；
  - 一份 benchmark 性能结果；
  - 一个 bug 修复记录；
  - 一张架构图；
  - 一篇简短技术笔记。

每天的循环：

1. 选一个很小的任务。
2. 写清楚什么叫完成。
3. 做出最小可运行版本。
4. 运行并保存结果证据。
5. 写一句复盘：改了什么、哪里失败、学到了什么。

每周复盘：

- 更新 `docs/week_notes.md`。
- 找出下周最需要补的一个短板。
- 如果项目已经能跑，把代码推到 GitHub。
- 把本周成果改写成一条简历 bullet。

---

## 3. 阶段路线图

### 阶段 0：环境搭建与学习纪律

**预计时间：** 3 到 5 天

**目标：** 让你的开发环境稳定，并建立可重复的学习流程。

**要学：**

- Python 项目结构。
- `venv` 或 conda 环境。
- Git 基础：commit、branch、diff、log。
- 从终端运行 Python 脚本。
- 阅读 Python 报错栈。

**要做：**

把你现在的 PyTorch 练习代码整理成一个小型学习项目：

```text
pytorch_learning/
  src/
    basics/
    models/
    training/
  tests/
  docs/
  notebooks/
  scripts/
  requirements.txt
  README.md
```

**交付物：**

- `README.md`：说明如何安装依赖、如何运行示例。
- `docs/week_notes.md`：每周学习笔记。
- 至少 5 次小 commit。

**检查点：**

- 你能自己创建环境、安装依赖、运行 Python 文件，并解释一次 Git diff。

---

### 阶段 1：Python 工程基础

**预计时间：** 2 周

**目标：** 从“只会写单个脚本”，升级到“能写可维护的 Python 模块”。

**要学：**

- 函数、类、dataclass、类型标注。
- 使用 `pathlib` 处理路径。
- 使用 logging，而不是到处 `print`。
- 用 YAML 或 JSON 管理配置。
- 用 `pytest` 写单元测试。
- 用 `argparse` 做命令行参数。

**项目 1：训练实验工具包**

做一个可复用的小工具包，用来支持后面的训练实验。

功能包括：

- 加载配置文件；
- 设置随机种子；
- 创建输出目录；
- 保存 metrics；
- 保存 checkpoint；
- 绘制训练曲线；
- 记录当前设备信息。

**建议创建文件：**

```text
src/mlinfra/config.py
src/mlinfra/seed.py
src/mlinfra/checkpoint.py
src/mlinfra/metrics.py
src/mlinfra/device.py
tests/test_config.py
tests/test_checkpoint.py
```

**交付物：**

- 8 到 12 个单元测试。
- 命令行入口：`python scripts/train_mnist.py --config configs/mnist.yaml`。
- 项目文档：`docs/project_1_training_toolkit.md`。

**简历 bullet 目标：**

- 构建了一个可复用的 PyTorch 实验工具包，支持配置加载、checkpoint 保存、metrics 持久化、随机种子复现和单元测试。

---

### 阶段 2：PyTorch 训练系统

**预计时间：** 3 周

**目标：** 理解完整训练流程，而不是只会写模型结构。

**要学：**

- Tensor shape。
- Dataset 和 DataLoader。
- 训练循环设计。
- 验证循环。
- checkpoint 保存和恢复训练。
- CPU 与 GPU 的区别。
- 混合精度基础。
- 小数据集过拟合与调试方法。

**项目 2：可复现的视觉模型训练流水线**

做一个干净的 MNIST 或 CIFAR-10 训练项目。

必须包含：

- train / validation / test 划分；
- 通过配置文件控制超参数；
- checkpoint 保存与恢复训练；
- metrics 保存为 JSON 或 CSV；
- loss 和 accuracy 曲线；
- confusion matrix；
- 固定随机种子的可复现模式；
- 针对数据加载和 checkpoint 恢复的测试；
- 最终实验报告。

**里程碑：**

1. 训练出 baseline 模型。
2. 加入验证集和 checkpoint。
3. 加入恢复训练。
4. 加入 metrics 和图表。
5. 加入测试。
6. 写最终报告。

**交付物：**

- `scripts/train_vision.py`
- `scripts/evaluate_vision.py`
- `configs/vision_baseline.yaml`
- `docs/project_2_vision_training_pipeline.md`
- `outputs/example_run/metrics.json`

**检查点：**

- 你能解释为什么要有验证集、checkpoint 恢复训练怎么做、shape 报错怎么调试。

**简历 bullet 目标：**

- 开发了一个可复现的 PyTorch 图像分类训练流水线，支持配置驱动训练、checkpoint 恢复、指标追踪、评估报告和自动化测试。

---

### 阶段 3：模型服务与 API 基础设施

**预计时间：** 3 周

**目标：** 把训练好的模型变成可靠的在线服务。

**要学：**

- FastAPI 基础。
- 用 Pydantic 定义请求和响应格式。
- 服务启动时加载模型。
- 健康检查。
- 批量推理。
- 异步请求处理基础。
- 延迟测量。
- Docker 基础。

**项目 3：模型推理服务**

把你训练好的视觉模型封装成 API。

必须包含：

- `/health`
- `/predict`
- `/batch_predict`
- 模型版本接口；
- 输入校验；
- 结构化日志；
- 延迟日志；
- Dockerfile；
- API 测试；
- benchmark 脚本。

**建议创建文件：**

```text
src/serving/app.py
src/serving/model_loader.py
src/serving/schemas.py
src/serving/preprocess.py
tests/test_serving_api.py
scripts/benchmark_inference.py
Dockerfile
```

**需要测的指标：**

- p50 延迟；
- p95 延迟；
- 每秒请求数；
- 内存占用；
- 冷启动时间。

**交付物：**

- 本地可运行的 FastAPI 服务。
- 本地可运行的 Docker 容器。
- benchmark 报告：`docs/project_3_inference_service.md`。

**简历 bullet 目标：**

- 构建并容器化了一个 FastAPI 模型推理服务，支持批量预测、健康检查、结构化日志、自动化 API 测试和延迟 benchmark。

---

### 阶段 4：MLOps 流水线与可观测性

**预计时间：** 4 周

**目标：** 建出公司里常见的工作流：训练、注册、服务、监控、复现。

**要学：**

- Docker Compose。
- MLflow 或轻量级实验记录系统。
- GitHub Actions CI。
- 服务 metrics。
- Prometheus / Grafana 基础。
- 数据和模型版本管理概念。
- 部署检查清单。

**项目 4：迷你 MLOps 平台**

为一个模型搭建本地 AI 平台：

- 训练任务；
- 模型 artifact 目录；
- 模型 metadata 文件；
- 模型 registry 表；
- 推理服务；
- metrics 接口；
- dashboard；
- CI 测试。

**架构：**

```text
训练脚本 -> artifact 存储 -> registry 元数据 -> 推理 API -> metrics -> dashboard
```

**必须支持：**

- 每个模型版本都记录：
  - version id；
  - 训练配置；
  - 训练指标；
  - artifact 路径；
  - 创建时间；
  - 如果有 Git 仓库，记录 git commit。
- 推理服务可以加载指定模型版本。
- metrics 接口暴露请求数、错误数、延迟。
- CI 自动运行单元测试和 API 测试。

**交付物：**

- `docker-compose.yml`
- `src/registry/`
- `src/monitoring/`
- `.github/workflows/test.yml`
- `docs/project_4_mini_mlops_platform.md`

**简历 bullet 目标：**

- 设计了一个本地 MLOps 平台，串联训练、模型版本元数据、容器化推理服务、CI 测试和 Prometheus 兼容的推理 metrics。

---

### 阶段 5：LLM 与 RAG 基础设施

**预计时间：** 4 周

**目标：** 掌握实用的 LLM 基础设施能力：检索、评测、服务、成本和延迟控制。

**要学：**

- Tokenization 基础。
- Embedding。
- 向量搜索。
- 文档切分策略。
- RAG 流水线设计。
- Rerank 重排序。
- 评测数据集。
- Prompt 和版本管理。
- 流式 API 响应。

**项目 5：生产风格 RAG 服务**

基于你自己的文档搭建一个 RAG 系统。

必须包含：

- 文档导入；
- 文档切分；
- embedding 生成；
- 向量索引；
- 检索接口；
- 问答接口；
- 来源引用；
- 流式响应；
- 评测脚本；
- 延迟和成本报告。

**推荐技术栈：**

- FastAPI；
- SQLite 保存 metadata；
- FAISS 或 Chroma 做向量搜索；
- sentence-transformers 做本地 embedding；
- 如果可用，可以接 OpenAI 兼容的 LLM API；
- pytest 测核心逻辑。

**建议创建文件：**

```text
src/rag/ingest.py
src/rag/chunking.py
src/rag/embeddings.py
src/rag/vector_store.py
src/rag/retriever.py
src/rag/api.py
src/rag/eval.py
tests/test_chunking.py
tests/test_retriever.py
scripts/ingest_docs.py
scripts/evaluate_rag.py
```

**交付物：**

- RAG API。
- 包含 30 到 50 个问题的评测数据集。
- 对比不同 chunk size、top-k 和检索质量的报告。
- `docs/project_5_rag_infra_service.md`。

**简历 bullet 目标：**

- 构建了一个 RAG 基础设施服务，支持文档导入、向量索引、检索评测、来源引用、流式响应和延迟分析。

---

### 阶段 6：分布式与 GPU 基础设施

**预计时间：** 4 到 6 周

**目标：** 理解严肃 AI 工作负载背后的基础设施层。

**要学：**

- CUDA 和 GPU 显存概念。
- batch size、延迟、吞吐之间的关系。
- PyTorch profiler。
- DataLoader 性能。
- 分布式训练概念：DDP、rank、world size、all-reduce。
- Kubernetes 基础：pod、deployment、service、config map、secret、日志。
- GPU 调度基本概念。

**项目 6：推理性能实验室**

做一个模型推理 benchmark 套件。

必须做的实验：

- 如果有 GPU，比较 CPU 和 GPU；
- 单请求和批量请求；
- 不同 batch size；
- FP32 和混合精度；
- 不同并发数；
- 模型 warmup；
- 内存占用。

**交付物：**

- benchmark 脚本；
- 结果表格；
- 图表；
- 性能分析报告；
- 优化建议。

**项目 7：Kubernetes AI 服务部署**

把你的推理服务部署到本地 Kubernetes 环境，例如 Docker Desktop Kubernetes、kind 或 minikube。

必须包含：

- deployment YAML；
- service YAML；
- config map；
- health probes；
- resource requests 和 limits；
- rolling update；
- 日志查看说明。

**交付物：**

- `k8s/deployment.yaml`
- `k8s/service.yaml`
- `k8s/configmap.yaml`
- `docs/project_7_k8s_ai_service.md`

**简历 bullet 目标：**

- 对 AI 推理服务进行了 batch 和并发性能 benchmark，并将服务部署到 Kubernetes，配置了健康检查、资源限制和可复现部署清单。

---

## 4. 最终作品集 Capstone

**预计时间：** 4 周

**目标：** 把前面学到的东西合成一个强有力的求职项目。

**Capstone 项目：训练、模型注册、推理服务、监控和 RAG 一体化 AI 平台**

它应该像一个小型公司内部平台：

- 训练模型；
- 注册模型 artifact；
- 部署模型服务；
- 暴露 metrics；
- 运行 benchmark；
- 提供 RAG 接口；
- 写清楚架构和技术取舍。

**项目结构建议：**

```text
README.md
docs/
  architecture.md
  api.md
  deployment.md
  benchmark_report.md
  interview_notes.md
src/
tests/
configs/
scripts/
k8s/
docker-compose.yml
Dockerfile
```

**README 必须包含：**

- 架构图；
- 快速启动方式；
- 截图或终端运行示例；
- API 示例；
- benchmark 总结；
- 测试运行说明；
- 后续可改进点。

**最终简历 bullet：**

- 构建了一个端到端 AI 基础设施平台，覆盖 PyTorch 训练、模型 registry 元数据、FastAPI 推理服务、Docker 部署、Prometheus 风格 metrics、benchmark 自动化和 Kubernetes 部署清单。
- 实现了一个 RAG 服务，支持文档导入、本地 embedding、向量搜索、检索评测、来源引用和流式 API 响应。
- 为 checkpoint、配置管理、模型加载、推理 API 和检索逻辑添加了 CI 覆盖的 Python 测试。

---

## 5. 面试准备计划

从阶段 3 之后开始面试准备，然后每周持续推进。

### 必须掌握的主题

**Python：**

- generator；
- decorator；
- context manager；
- async 基础；
- typing；
- packaging。

**机器学习：**

- train / validation / test 划分；
- 过拟合；
- loss function；
- 梯度下降；
- checkpoint；
- metrics。

**PyTorch：**

- tensor；
- autograd；
- module；
- dataloader；
- GPU 数据迁移；
- `model.train()` 和 `model.eval()` 的区别。

**模型服务：**

- REST API；
- batch；
- latency；
- throughput；
- cold start；
- health check；
- logging。

**基础设施：**

- Docker；
- Kubernetes 基础；
- CI；
- monitoring；
- deployment rollback；
- resource limits。

**LLM / RAG：**

- embedding；
- chunking；
- vector search；
- top-k retrieval；
- rerank；
- hallucination；
- evaluation。

### 每周面试训练

- 2 道 coding 题。
- 1 道系统设计题。
- 1 次项目深挖演练。
- 1 个 STAR 格式的 debugging 故事。

项目深挖问题示例：

- 为什么选择 FastAPI？
- checkpoint 恢复训练是怎么工作的？
- 你怎么测延迟？
- 如果模型加载失败会发生什么？
- 如果流量扩大 10 倍，你怎么处理？
- 如何降低推理成本？
- 如何评估 RAG 系统？
- 生产环境里你会监控什么？

---

## 6. 推荐学习资料

资料只是工具，不要用“看资料”逃避做项目。

### Python 与工程基础

- Python 官方教程。
- `pytest` 官方文档。
- FastAPI 官方文档。
- 遇到具体问题时查 Real Python。

### PyTorch

- PyTorch 官方 tutorials。
- PyTorch recipes。
- Hugging Face course 里关于 datasets 和 transformers 的部分。

### MLOps 与基础设施

- Docker 官方 getting started。
- Kubernetes 官方 basics tutorial。
- MLflow 文档。
- Prometheus Python client 文档。

### LLM / RAG

- Hugging Face sentence-transformers 文档。
- FAISS 文档。
- 如果使用托管 LLM，则查看 OpenAI 或 OpenAI 兼容 API 文档。
- 可信工程博客里的 RAG evaluation 文章。

规则：看资料 60 到 90 分钟后，必须马上动手做东西。

---

## 7. 前 14 天具体安排

### 第 1 天

- 创建项目结构。
- 创建环境。
- 写 `README.md` 快速启动说明。
- 运行一个现有 PyTorch 脚本。
- 写 `docs/week_notes.md`。

### 第 2 天

- 学 `pytest`。
- 给简单工具函数写 2 个测试。
- 创建 `src/mlinfra/seed.py`。
- commit。

### 第 3 天

- 创建 config loader。
- 写“配置文件不存在”和“配置文件合法”的测试。
- 添加一个 YAML 示例配置。
- commit。

### 第 4 天

- 创建 checkpoint 保存和加载工具。
- 用一个很小的 PyTorch 模型写测试。
- commit。

### 第 5 天

- 重构一个当前脚本，让它使用 config、seed、checkpoint 工具。
- 端到端运行一次。
- 把结果写进 `docs/week_notes.md`。

### 第 6 天

- 加入 metrics 保存到 JSON / CSV。
- 加入训练曲线绘制。
- commit。

### 第 7 天

- 做周复盘。
- 写 `docs/project_1_training_toolkit.md`。
- 草拟第一条简历 bullet。

### 第 8 天

- 构建干净的 MNIST DataLoader 模块。
- 测试 tensor shape 和 label 范围。

### 第 9 天

- 构建 baseline CNN 模型。
- 训练 1 个 epoch。
- 保存 metrics。

### 第 10 天

- 加入 validation loop。
- 记录 best validation accuracy。

### 第 11 天

- 加入 checkpoint resume。
- 测试恢复训练行为。

### 第 12 天

- 加入 evaluation 脚本。
- 生成 confusion matrix。

### 第 13 天

- 整理文档和 README。
- 运行全部测试。

### 第 14 天

- 复盘自己能不能把项目讲清楚。
- 记录薄弱点。
- 制定阶段 2 的完成计划。

---

## 8. 每个项目的质量标准

一个项目只有满足下面条件，才适合放进作品集或简历：

- 有可运行的 setup 说明；
- 有测试；
- 有有意义的错误处理；
- 有日志；
- 有配置文件；
- 有可复现命令；
- 有 benchmark 或 evaluation 结果；
- 有架构说明；
- 有限制和后续改进；
- commit 历史清楚。

不要把半成品 demo 放到简历上。小而完整的系统，比大而混乱的系统更有说服力。

---

## 9. 我会怎么带你

每个项目我们都按这个流程走：

1. 定义最小有价值版本。
2. 写短规格说明。
3. 拆成具体任务。
4. 带测试实现。
5. 用证据调试。
6. 做 benchmark 或 evaluation。
7. 写项目文档。
8. 把结果转成简历 bullet 和面试故事。

你卡住时，把下面信息发给我：

- 你运行的命令；
- 完整报错；
- 你改过的文件；
- 你原本期待的结果；
- 实际发生的结果。

然后我们按步骤定位问题，不靠瞎猜。

---

## 10. 下一步行动

先从阶段 0 和阶段 1 开始。

第一个具体任务是：

1. 把当前仓库整理成干净的 Python 项目结构。
2. 添加 `README.md`。
3. 添加 `src/mlinfra/seed.py`。
4. 添加 `tests/test_seed.py`。
5. 运行 `pytest`。

完成后，把一个现有学习脚本移动到新结构里，并让它变得可复现。

