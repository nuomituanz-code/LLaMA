# LLM Inference Engine Lab 学习笔记

从今天开始，旧的 MNIST / Mini Training Platform 学习记录清空。

新的学习路线只围绕一个更有招聘价值的项目推进：

```text
LLM Inference Engine Lab
大模型推理服务与性能优化实验室
```

记录规则：

- 每一步都写清楚“做了什么”。
- 每一步都写清楚“有什么用”。
- 每一步都写清楚“产出了什么文件或目录”。
- 不为了展示基础能力而做空项目，基础能力必须服务于真实项目。

---

## Step 1：项目重置与目标确认

### 做了什么

1. 删除了前面无意义的 Mini Training Platform 相关文件。
2. 保留原始 PyTorch 练习脚本作为历史学习材料：
   - `RNNsimple.py`
   - `CNNsimple.PY`
3. 确认旧项目文件已经不存在：
   - `docs/project_mini_ai_training_platform.md`
   - `scripts/train_mnist_baseline.py`
4. 创建新的 LLM Infra 项目骨架。

### 为什么有用

前面的 MNIST / CNN / RNN 小练习不能支撑 AI Infra / LLM Infra 岗位简历。

你给的岗位要求强调的是：

- LLM 推理；
- PyTorch / Transformer / vLLM / sglang；
- CUDA；
- GPU 并行计算；
- 内存管理；
- 性能优化；
- 模型量化；
- 分布式 DP / TP / PP / EP；
- TensorRT / cuDNN / NCCL；
- Triton / ONNX Runtime；
- 大规模模型部署。

所以项目必须围绕“大模型推理和性能优化”来做，而不是围绕普通图像分类练习来做。

这一步的价值是：先把方向纠正，避免继续在低价值项目上投入时间。

### 产出的目录

```text
src/llm_lab/
cuda/
cpp/
triton/
docs/llm_inference_engine_lab/
configs/
scripts/
tests/
```

### 每个目录的作用

`src/llm_lab/`

用于放 Python 侧的大模型推理系统代码，例如模型加载、文本生成、batching、benchmark、API 服务。

`cuda/`

用于放 CUDA 学习实验，例如 vector add、reduction、matmul tiling、layernorm。这部分对齐岗位里的 CUDA 编程、GPU 并行计算、内存管理和性能优化要求。

`cpp/`

用于放 C/C++ 实验，例如 tokenizer demo、ONNX Runtime C++ 推理 demo。这部分对齐岗位里的 C/C++ 和高性能代码能力。

`triton/`

用于放 Triton Inference Server 的模型仓库和部署配置。这部分对齐岗位里的模型服务化框架和大规模模型部署要求。

`docs/llm_inference_engine_lab/`

用于写项目文档，例如架构图、benchmark 报告、量化报告、CUDA 学习笔记、部署报告和面试笔记。

`configs/`

用于保存模型和实验配置，例如模型名称、max tokens、batch size、量化方式、benchmark 参数。

`scripts/`

用于放可执行脚本，例如启动服务、运行推理、跑 benchmark、导出 ONNX、运行量化实验。

`tests/`

用于放测试代码。后续每写一个关键模块，都要补最小测试，保证项目不是一次性脚本。

### 当前项目主线

```text
LLM 本地推理
-> 推理 API 服务
-> latency / throughput benchmark
-> batching 与 KV cache 实验
-> 量化实验
-> CUDA kernel 学习
-> ONNX Runtime / TensorRT / Triton 部署
-> 分布式推理理解
```

### 下一步

下一步做 `docs/llm_inference_engine_lab/llm_inference_basics.md`。

目标是先把 LLM 推理链路讲清楚：

- tokenizer 是什么；
- input ids 是什么；
- logits 是什么；
- decoding 是什么；
- greedy / sampling 有什么区别；
- prefill 和 decode 是什么；
- KV cache 为什么重要。

这一步做完以后，再开始写第一个真实推理脚本。

---

## Step 2：理解 LLM 推理基础链路

### 做了什么

创建了文档：

```text
docs/llm_inference_engine_lab/llm_inference_basics.md
```

这份文档系统整理了 LLM 推理的基础链路：

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

还整理了后续性能优化必须理解的概念：

- tokenizer；
- input ids；
- attention mask；
- logits；
- greedy decoding；
- sampling；
- temperature / top-k / top-p；
- 自回归生成；
- prefill；
- decode；
- KV cache；
- batch 推理为什么复杂。

### 为什么有用

我们后面要做的是 LLM 推理系统，不是简单调用 API。

如果不理解这条链路，后面这些内容都会变成黑盒：

- 为什么 prompt 长度影响首 token 延迟；
- 为什么生成越长总耗时越久；
- 为什么 KV cache 能加速但会占显存；
- 为什么 batch 推理能提高吞吐但可能增加延迟；
- 为什么 vLLM / sglang 这类推理框架有价值；
- 为什么 benchmark 不能只测一个总耗时。

这一步的价值是：先建立推理系统的概念地图。后面写代码、做 benchmark、做量化、学 CUDA，都能知道自己在优化链路里的哪一部分。

### 产出的文件

```text
docs/llm_inference_engine_lab/llm_inference_basics.md
```

### 当前阶段要记住的核心句子

LLM 推理不是“一次输入，一次输出”。

它是：

```text
先 prefill 处理 prompt，再 decode 逐 token 生成。
```

后续所有性能优化都绕不开：

```text
token 数量
batch size
KV cache
显存
latency
throughput
```

### 下一步

下一步开始做第一个可运行脚本：

```text
scripts/generate_text.py
```

目标：

- 先支持 mock tokenizer / mock model，避免卡在模型下载；
- 跑通 prompt -> token ids -> logits -> decode -> output text 的最小链路；
- 再替换成真实 Hugging Face 小模型。

---

## Step 3：实现最小 LLM 推理链路

### 你需要先看哪里

在做这一步之前，需要重点回看：

```text
docs/llm_inference_engine_lab/llm_inference_basics.md
```

重点看这些章节：

- 第 3 节：Tokenizer 是什么；
- 第 4 节：Input IDs 是什么；
- 第 7 节：Logits 是什么；
- 第 8 节：Decoding 是什么；
- 第 9 节：自回归生成是什么。

然后看今天新增的代码：

```text
src/llm_lab/mock_generation.py
scripts/generate_text.py
tests/test_mock_generation.py
```

### 看完要得到什么收获

你要能用自己的话解释：

1. 为什么模型不能直接处理字符串；
2. tokenizer 为什么要把文本变成 token id；
3. logits 为什么可以理解为“每个 token 的候选分数”；
4. greedy decoding 为什么是选择最高分 token；
5. 自回归生成为什么是“一次生成一个 token”；
6. tokenizer 词表不完整时会发生什么问题。

### 做了什么

创建了一个不依赖真实大模型的最小推理链路。

文件：

```text
src/llm_lab/mock_generation.py
scripts/generate_text.py
tests/test_mock_generation.py
```

它跑通了下面这条链路：

```text
prompt
-> MockTokenizer.encode()
-> input ids
-> MockCausalLM.next_token_logits()
-> greedy_decode()
-> generated token ids
-> MockTokenizer.decode()
-> output text
```

### 为什么有用

这一步不是为了做一个假模型项目，而是为了在没有下载真实模型之前，先把推理系统的控制流程写清楚。

真实 Hugging Face 模型后面会替换掉：

```text
MockTokenizer
MockCausalLM
```

但主流程不会变：

```text
文本输入 -> tokenizer -> 模型输出 logits -> decode -> 文本输出
```

这能避免一开始就被模型下载、显存、依赖问题卡住。

### 运行命令

运行测试：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe -m pytest tests\test_mock_generation.py -v --basetemp .pytest-tmp
```

运行脚本：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe scripts\generate_text.py --prompt hello --max-new-tokens 20
```

### 运行结果

脚本输出：

```text
Prompt: hello
Input token ids: [7, 5, 10, 10, 13]
Generated token ids: [2, 16, 7, 8, 15, 2, 8, 15, 2, 3, 2, 11, 13, 4, 9, 2, 10, 10, 11]
Output text:  this is a mock llm
```

测试结果：

```text
4 passed in 0.03s
```

### 遇到的问题

第一次测试失败，因为 mock tokenizer 的词表里没有字母 `k`。

结果 `"mock"` 被编码/解码成了类似：

```text
moc  llm
```

修复方式：

在 `MockTokenizer.vocab` 里加入：

```text
k
```

### 这个问题有什么启发

这不是小问题，它对应真实 LLM 项目里的一个关键点：

```text
tokenizer 必须和模型权重匹配。
```

如果 tokenizer 版本不对，token id 可能变化，模型理解的输入就会错。

真实项目里，不能随便替换 tokenizer，也不能只保存模型权重而不记录 tokenizer 和模型版本。

### 代码质量修正：显式声明 tokenizer 映射表

后续查看 `mock_generation.py` 时，编辑器提示：

```text
token_to_id
```

相关调用可能报错或被静态检查器标红。

原因：

`token_to_id` 和 `id_to_token` 原来是在 `__post_init__()` 里通过 `object.__setattr__()` 动态创建的。

Python 运行时能识别，但 VS Code / Pylance 这类静态检查工具不一定能知道这个属性存在。

修复方式：

在 dataclass 里显式声明字段：

```python
token_to_id: dict[str, int] = field(init=False)
id_to_token: dict[int, str] = field(init=False)
```

这样做的好处：

- 编辑器能正确识别属性；
- 类型提示更清楚；
- 团队协作时代码更容易读；
- 后续接真实 tokenizer 时，也更容易维护接口。

修复后重新运行测试：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe -m pytest tests\test_mock_generation.py -v --basetemp .pytest-tmp
```

结果：

```text
4 passed in 0.03s
```

### 下一步

下一步做：

```text
configs/mock_llm.json
```

并改造 `scripts/generate_text.py`：

- 从配置文件读取默认 prompt；
- 从配置文件读取 `max_new_tokens`；
- 为以后切换真实 Hugging Face 模型预留配置字段。

这样项目会逐步从“脚本”走向“可配置推理系统”。

---

## Step 4：把推理脚本配置化

### 你需要先看哪里

先看当前脚本：

```text
scripts/generate_text.py
```

重点看它原来的问题：

- 默认 prompt 写在脚本里；
- 默认 `max_new_tokens` 写在脚本里；
- 后面要切换真实模型时，还得继续改代码。

然后看新增配置：

```text
configs/mock_llm.json
```

再看配置读取逻辑：

```text
src/llm_lab/config.py
```

### 看完要得到什么收获

你要理解：

1. 配置文件不是为了“显得工程化”，而是为了让实验参数和代码分离。
2. 后续切换模型、生成参数、benchmark 参数时，不应该频繁改 Python 源码。
3. 命令行参数可以覆盖配置文件，这样适合临时实验。
4. 配置读取必须有错误处理，例如文件不存在、JSON 格式错误、参数非法。

### 做了什么

新增配置文件：

```text
configs/mock_llm.json
```

新增配置读取模块：

```text
src/llm_lab/config.py
```

新增测试：

```text
tests/test_config.py
```

改造脚本：

```text
scripts/generate_text.py
```

现在脚本支持：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe scripts\generate_text.py --config configs\mock_llm.json
```

也支持命令行覆盖配置：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe scripts\generate_text.py --config configs\mock_llm.json --prompt infra --max-new-tokens 5
```

### 为什么有用

真实 LLM Infra 项目里，经常要对比很多实验：

- 不同模型；
- 不同 `max_new_tokens`；
- 不同 temperature；
- 不同 batch size；
- 不同量化方式；
- 不同 benchmark 参数。

如果每次都改代码，项目会很乱，也很难复现。

配置化之后，后续可以做到：

```text
同一份代码 + 不同配置 = 不同实验
```

这就是后面做 benchmark、量化对比、服务部署的基础。

### 运行结果

运行测试：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe -m pytest tests\test_config.py tests\test_mock_generation.py -v --basetemp .pytest-tmp
```

结果：

```text
11 passed in 0.09s
```

使用配置文件运行：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe scripts\generate_text.py --config configs\mock_llm.json
```

输出：

```text
Prompt: hello
Input token ids: [7, 5, 10, 10, 13]
Generated token ids: [2, 16, 7, 8, 15, 2, 8, 15, 2, 3, 2, 11, 13, 4, 9, 2, 10, 10, 11]
Output text:  this is a mock llm
```

使用命令行覆盖配置：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe scripts\generate_text.py --config configs\mock_llm.json --prompt infra --max-new-tokens 5
```

输出：

```text
Prompt: infra
Input token ids: [8, 12, 6, 14, 3]
Generated token ids: [2, 16, 7, 8, 15]
Output text:  this
```

### 当前收获

现在项目已经从：

```text
能跑的 mock 推理脚本
```

升级为：

```text
可配置的 mock 推理实验
```

这一步虽然还没有接真实模型，但已经在为真实模型接入做准备。

### 下一步

下一步接入真实 Hugging Face 小模型。

你需要先看：

- Hugging Face `AutoTokenizer`
- Hugging Face `AutoModelForCausalLM`
- `model.generate()`

目标：

- 新增 `src/llm_lab/hf_generation.py`；
- 支持从配置文件读取模型名；
- 优先尝试小模型；
- 如果下载失败，保留 mock pipeline 作为 fallback。

---

## Step 5：接入真实 Hugging Face 小模型

### 你需要先看哪里

先看 Hugging Face Transformers 的三个核心接口：

```text
AutoTokenizer.from_pretrained()
AutoModelForCausalLM.from_pretrained()
model.generate()
```

再看今天新增的代码：

```text
src/llm_lab/hf_generation.py
scripts/generate_hf_text.py
configs/hf_tiny_gpt2.json
tests/test_hf_generation.py
```

### 看完要得到什么收获

你要理解：

1. `AutoTokenizer.from_pretrained()` 用模型名或本地目录加载 tokenizer。
2. `AutoModelForCausalLM.from_pretrained()` 加载因果语言模型，适合 GPT / LLaMA / Qwen 这类自回归生成模型。
3. `model.generate()` 封装了自回归生成循环。
4. 真实模型推理时，input ids 和 attention mask 要移动到同一个 device。
5. GPU 可用时，可以把模型放到 CUDA 上推理。
6. 模型输出可能包含任意 Unicode 字符，Windows 控制台可能打印失败。

### 做了什么

安装依赖：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe -m pip install transformers accelerate safetensors
```

新增 Hugging Face 推理模块：

```text
src/llm_lab/hf_generation.py
```

新增真实模型配置：

```text
configs/hf_tiny_gpt2.json
```

新增运行脚本：

```text
scripts/generate_hf_text.py
```

新增测试：

```text
tests/test_hf_generation.py
```

### 为什么有用

前面 mock pipeline 是为了理解流程。

这一步开始接真实生态：

```text
Hugging Face tokenizer
Hugging Face model
model.generate()
CUDA device
真实模型权重格式
模型下载与缓存
```

这一步的价值是让项目从“模拟推理链路”进入“真实 LLM 推理链路”。

### 遇到的问题 1：tiny-gpt2 权重格式受 PyTorch 安全限制影响

一开始尝试：

```text
sshleifer/tiny-gpt2
```

失败原因：

```text
transformers 要求 torch >= 2.6 才能加载旧的 torch.load 权重格式
```

这是因为 `torch.load` 相关安全漏洞限制。

解决方式：

不盲目升级 PyTorch，而是换成带 `safetensors` 的 tiny 模型：

```text
tiny-random/qwen2.5
```

这个选择更符合工程原则：优先选择更安全的权重格式。

### 遇到的问题 2：Windows 控制台 GBK 编码无法打印模型输出

真实模型输出中包含阿拉伯语、韩文等字符。

Windows 控制台默认 GBK 编码无法打印某些 Unicode 字符，报错：

```text
UnicodeEncodeError
```

解决方式：

在 `scripts/generate_hf_text.py` 中加入 `safe_text()`，用 `backslashreplace` 方式安全打印。

### 运行命令

运行真实模型推理：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe scripts\generate_hf_text.py --config configs\hf_tiny_gpt2.json
```

运行测试：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe -m pytest tests\test_hf_generation.py tests\test_config.py tests\test_mock_generation.py -v --basetemp .pytest-tmp
```

### 运行结果

真实模型推理输出：

```text
Model: tiny-random/qwen2.5
Device: cuda
Prompt: LLM inference is
Input tokens: 4
Generated tokens: 16
Output text: LLM inference is_packageetsy electoral ...
```

测试结果：

```text
14 passed in 5.10s
```

### 当前收获

现在项目已经有两条推理链路：

```text
mock pipeline：用于理解和测试控制流程
Hugging Face pipeline：用于真实模型推理
```

这很重要，因为后续做 benchmark、batching、量化时，mock pipeline 可以保留为快速测试，真实 HF pipeline 用来做实际实验。

### 代码质量修正：让编辑器识别 Hugging Face 模型接口

后续查看 `hf_generation.py` 时，编辑器在这里提示报错：

```python
model.to(device)
```

原因：

`AutoModelForCausalLM.from_pretrained()` 返回的是 Hugging Face 动态模型类型。Python 运行时知道它有 `.to()`、`.eval()`、`.generate()`，但静态检查器不一定能准确推断。

修复方式：

新增两个接口协议：

```python
TokenizerLike
CausalLMLike
```

其中 `CausalLMLike` 显式描述当前代码需要的模型能力：

- `to()`
- `eval()`
- `generate()`

这样做的好处：

- 编辑器更容易理解模型对象；
- 类型提示更清楚；
- 代码读者能直接看到我们依赖哪些接口；
- 后续如果换 vLLM / sglang / 自定义模型包装类，也更容易保持接口一致。

修复后验证：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe -m pytest tests\test_hf_generation.py tests\test_config.py tests\test_mock_generation.py -v --basetemp .pytest-tmp
```

结果：

```text
14 passed in 5.12s
```

### 下一步

下一步做最小 benchmark。

你需要先看：

- latency 是什么；
- tokens/sec 是什么；
- 为什么要区分 input token 数和 generated token 数；
- 为什么第一次运行要考虑模型加载时间和 warmup。

目标：

- 新增 `src/llm_lab/benchmark.py`；
- 新增 `scripts/benchmark_hf_generation.py`；
- 测量一次 Hugging Face 推理的耗时、输入 token 数、生成 token 数和 tokens/sec。

---

## Step 6：做最小 LLM 推理 Benchmark

### 你需要先看哪里

先回看：

```text
src/llm_lab/hf_generation.py
```

重点看两个函数：

```python
load_hf_causal_lm()
generate_with_loaded_hf_model()
```

你要注意一个关键拆分：

```text
模型加载
```

和

```text
单次生成请求
```

不是一回事。

真实在线推理服务通常是在服务启动时加载模型，用户请求来了之后只执行 tokenizer、forward/generate、decode。所以 benchmark 不能每次都把模型加载时间算进去，否则指标会严重失真。

然后看：

```text
src/llm_lab/benchmark.py
scripts/benchmark_hf_generation.py
```

### 看完要得到什么收获

你要理解：

1. `latency` 是一次请求从开始到结束的耗时。
2. `tokens/sec` 表示单位时间能生成多少 token，是 LLM decode 性能的重要指标。
3. `input token` 和 `generated token` 要分开统计，因为 prefill 和 decode 阶段的计算特征不同。
4. CUDA 是异步执行的，计时时要用 `torch.cuda.synchronize()` 等 GPU 完成，否则时间可能不准。
5. benchmark 前要 warmup，因为第一次运行可能包含 CUDA 初始化、缓存、kernel 选择等额外开销。

### 这个模块解决什么问题

在 Step 5 里，真实 Hugging Face 模型已经能生成文本。

但“能生成”没有回答岗位里真正关心的问题：

```text
这个推理系统有多快？
吞吐是多少？
生成更多 token 会不会变慢？
GPU 上一次请求的延迟是多少？
优化有没有效果？
```

`benchmark.py` 的作用就是给推理链路加测量仪表。

没有 benchmark，就没法谈性能优化。

### 工作原理是什么

benchmark 的流程是：

```text
1. 先加载 tokenizer 和 model
2. 把模型移动到 CPU 或 CUDA
3. warmup 若干次，不计入正式结果
4. 正式运行 N 次 generate
5. 每次 generate 前后记录时间
6. 如果是 CUDA，计时前后都 synchronize
7. 统计平均延迟、最小延迟、最大延迟
8. 根据生成 token 数和总耗时计算 tokens/sec
```

这里最关键的是：

```text
模型只加载一次，benchmark 统计的是生成阶段。
```

这更接近真实在线推理服务。

### 为什么对找工作有帮助

你给的岗位要求里有：

- GPU 并行计算；
- 内存管理；
- 性能优化；
- 模型服务化；
- 大规模模型部署。

这些都不能只靠“我会调用模型”证明。

你需要能拿出数据说话：

```text
这个模型在这个 GPU 上，prompt 有多少 token，生成多少 token，平均延迟多少，tokens/sec 多少。
```

这就是 AI Infra 面试里非常有价值的表达方式。

你以后可以这样讲：

> 我不是只跑通了 Hugging Face generate，而是把模型加载和请求生成阶段拆开，加入 warmup 和 CUDA synchronize，统计每次生成的 latency 和 tokens/sec，用于后续比较 batch、量化和不同推理后端的性能。

这比“我会用 transformers”强很多。

### 做了什么

我们把 `hf_generation.py` 做了一次结构调整：

```text
load_hf_causal_lm()
```

只负责加载 tokenizer 和 model。

```text
generate_with_loaded_hf_model()
```

负责使用已经加载好的模型执行一次生成。

这样 benchmark 可以避免重复加载模型。

然后实现了：

```text
benchmark_generation()
```

它负责 warmup、计时、CUDA 同步和 tokens/sec 统计。

### 运行命令

运行测试：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe -m pytest tests\test_benchmark.py tests\test_hf_generation.py tests\test_config.py tests\test_mock_generation.py -v --basetemp .pytest-tmp
```

运行 benchmark：

```powershell
d:\pytorch_learning\.conda-envs\pytorch\python.exe scripts\benchmark_hf_generation.py --config configs\hf_tiny_gpt2.json --max-new-tokens 8 --warmup-runs 1 --benchmark-runs 3
```

### 运行结果

benchmark 输出：

```text
Model: tiny-random/qwen2.5
Device: cuda
Prompt: LLM inference is
Input tokens: 4
Generated tokens per run: 8
Runs: 3
Average latency: 0.0910s
Min latency: 0.0874s
Max latency: 0.0929s
Tokens/sec: 87.93
```

测试结果：

```text
15 passed in 5.27s
```

### 当前结果怎么理解

这次用的是 `tiny-random/qwen2.5`。

它是极小随机模型，所以：

- 输出质量没有意义；
- benchmark 数字不能代表真实 Qwen / LLaMA；
- 但链路是真实的 Hugging Face + CUDA + generate。

这一步的意义是建立 benchmark 方法。

以后换成真实模型后，同一套 benchmark 可以继续用。

### 下一步

下一步做 benchmark 报告和多参数实验。

目标：

- 对比 `max_new_tokens=1/4/8/16`；
- 记录 latency 和 tokens/sec；
- 生成 `docs/llm_inference_engine_lab/benchmark_report.md`；
- 学会从数据里解释为什么生成 token 越多，总耗时越长。

---

## Step 7：生成第一份可展示的 Benchmark 报告

### 本步目标

提高学习效率，不再只写解释性文档，而是产出能放进项目经历的实验结果。

本步直接回答一个工程问题：

```text
max_new_tokens 变大时，LLM 推理延迟和 tokens/sec 怎么变化？
```

### 做了什么

写了一个参数扫描脚本：

```text
scripts/benchmark_max_tokens.py
```

它会固定模型和 prompt，自动测试：

```text
max_new_tokens = 1, 4, 8, 16
```

然后生成报告：

```text
docs/llm_inference_engine_lab/benchmark_report.md
```

### 为什么这对找工作有帮助

岗位要求里写的“性能优化”“模型服务化”“GPU 推理”，都需要数据支撑。

这一步的价值是：

```text
你开始拥有自己的推理性能实验数据，而不是只说我会调用模型。
```

面试表达可以这样说：

> 我做了一个 LLM 推理 benchmark 脚本，固定模型和 prompt，对比不同生成长度下的 latency 和 tokens/sec。这样后续做 batch、量化、TensorRT 或 vLLM 对比时，可以用同一套实验方法衡量优化是否有效。

### 实验结果

```text
| max_new_tokens | input_tokens | avg_latency_s | min_latency_s | max_latency_s | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.0159 | 0.0152 | 0.0163 | 63.04 |
| 4 | 4 | 0.0383 | 0.0379 | 0.0388 | 104.56 |
| 8 | 4 | 0.0801 | 0.0738 | 0.0869 | 99.91 |
| 16 | 4 | 0.1550 | 0.1436 | 0.1768 | 103.23 |
```

### 当前结论

- 生成 token 越多，总延迟越高。
- `tokens/sec` 比总耗时更适合比较 decode 性能。
- 当前 tiny random 模型只用于验证方法，不代表真实业务模型性能。

### 验证

运行测试：

```text
15 passed in 5.19s
```

### 下一步

继续提高效率，下一步直接做：

```text
CPU vs GPU benchmark
```

目标是生成第二张表：

```text
同一个模型、同一个 prompt、同一个 max_new_tokens，在 CPU 和 GPU 上的 latency / tokens/sec 对比。
```

这个比继续写概念文档更有求职价值。

---

## Step 8：方向纠偏到成熟量化项目复现

### 结论

前面的 mock 推理和 tiny benchmark 仍然不够支撑岗位要求。

岗位真正需要的是：

```text
模型量化技术有实际项目落地案例
```

所以现在停止继续堆 toy benchmark，改为复现成熟开源项目：

```text
llama.cpp + Qwen GGUF 量化推理
```

### 删除了什么

删除当前低价值代码：

```text
src/llm_lab/
tests/test_*.py
scripts/generate_*.py
scripts/benchmark_*.py
configs/mock_llm.json
configs/hf_tiny_gpt2.json
docs/llm_inference_engine_lab/benchmark_report.md
```

缓存目录可能因 Windows 权限残留，但已被隐藏和 ignore。

### 为什么 llama.cpp 更有意义

llama.cpp 是成熟 C/C++ LLM 推理项目，支持：

- GGUF；
- 多种量化格式；
- CPU/GPU 推理；
- CUDA 后端；
- server 模式；
- benchmark；
- 真实模型部署。

这比自己写 toy pipeline 更接近岗位要求。

### 当前阻塞

尝试克隆失败：

```text
Failed to connect to github.com port 443
```

说明当前环境 GitHub 直连不可用。

### 产出

创建新的复现计划：

```text
docs/llm_quantization_reproduction_plan.md
```

### 下一步

解决 llama.cpp 源码获取问题：

- 换网络；
- 使用代理；
- 使用 GitHub zip；
- 使用镜像源；
- 手动放置源码到 `third_party/llama.cpp`。

拿到源码后，马上开始构建和 Qwen GGUF 量化复现。

### 重新拉取结果

再次尝试官方 GitHub：

```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git third_party\llama.cpp
```

仍然失败，原因是无法连接 `github.com:443`。

随后使用镜像源：

```powershell
git clone --depth 1 https://gh-proxy.com/https://github.com/ggml-org/llama.cpp.git third_party\llama.cpp
```

结果：成功。

源码位置：

```text
third_party/llama.cpp
```

确认存在关键文件：

```text
third_party/llama.cpp/CMakeLists.txt
third_party/llama.cpp/convert_hf_to_gguf.py
third_party/llama.cpp/tools/
third_party/llama.cpp/examples/
third_party/llama.cpp/src/
```

### 当前注意事项

执行 `git status` 时出现：

```text
detected dubious ownership
```

原因是仓库文件所有者和当前沙箱用户不同。这是 Git 的安全保护，不影响源码构建。

如果后续需要在这个仓库里执行 git 命令，可以加 safe.directory。

### 下一步

构建 llama.cpp。

优先顺序：

1. 先 CPU 构建，确保工具链跑通；
2. 再检查是否能 CUDA 构建；
3. 构建成功后确认这些工具可用：
   - `llama-cli`
   - `llama-quantize`
   - `llama-server`
   - `llama-bench`

## Step 9：尝试 CUDA 构建 llama.cpp，并定位工具链不兼容问题

### 这一步在做什么

我们不是在写一个玩具脚本，而是在复现真实 LLM 推理项目 `llama.cpp` 的 CUDA 后端构建。

CUDA 构建的意义是：让模型推理时可以把一部分计算放到 NVIDIA GPU 上执行。对 LLM Infra 岗位来说，这一步非常重要，因为岗位不会只问“会不会 Python”，还会问你是否理解：

- CUDA Toolkit、显卡驱动、MSVC 编译器之间的版本匹配；
- CMake / Ninja 如何组织 C++ 和 CUDA 项目的构建；
- 为什么同一个项目在不同机器上会因为工具链版本失败；
- 如何阅读编译错误并判断是源码问题还是环境问题。

这些能力就是“大模型部署”和“GPU 推理工程”的基础。

### 当前机器环境

我们检查到当前机器的关键环境是：

```text
GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4GB 显存
显卡驱动：546.92
nvidia-smi 显示 CUDA Version：12.3
nvcc：CUDA 12.3.107
CMake：Visual Studio 自带 CMake
Ninja：已安装到项目 conda 环境
MSVC：14.42 和 14.50
```

### 尝试 1：直接用 CUDA 12.3 + 新 MSVC 构建

结果失败。

核心错误是：

```text
unsupported Microsoft Visual Studio version
```

这说明 CUDA 12.3 的 `nvcc` 不认识当前 Visual Studio 2026 的环境。这里的重点不是“命令写错了”，而是 CUDA 编译器会检查宿主 C++ 编译器版本，如果版本不在它支持范围内，就会拒绝编译。

### 尝试 2：切换到旧一点的 MSVC 14.42

我们显式使用了机器上已有的 MSVC 14.42：

```text
C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.42.34433
```

结果仍然失败，但错误更精确：

```text
error STL1002: Unexpected compiler version, expected CUDA 12.4 or newer.
```

这个错误来自 MSVC 的 C++ 标准库头文件 `yvals_core.h`。意思是：当前这套 C++ 标准库认为 CUDA 12.3 太老，至少需要 CUDA 12.4。

### 当前结论

这次 CUDA 构建失败的根因是工具链版本不兼容：

```text
当前显卡驱动显示最高 CUDA 12.3
当前 MSVC / C++ 标准库要求 CUDA 12.4 或更新
所以 CUDA 12.3 + 当前 Visual Studio 环境无法完成 llama.cpp CUDA 编译
```

这不是 llama.cpp 源码问题，也不是我们项目设计问题，而是 AI Infra 工程里很常见的“底层工具链对齐”问题。

### 这一步对找工作的价值

这一步以后可以在简历或面试里这样表达：

```text
复现 llama.cpp CUDA 后端构建流程，排查 Windows 环境下 CUDA Toolkit、NVIDIA Driver、MSVC、CMake/Ninja 的版本兼容问题；通过编译日志定位到 CUDA 12.3 与当前 MSVC STL 不兼容，明确需要升级 CUDA/驱动或切换 VS2022 兼容工具链。
```

这比“跑了一个 MNIST”有价值得多，因为它对应的是实际部署大模型时一定会遇到的工程问题。

### 下一步选择

为了继续走 CUDA 构建，有两个实际方案：

1. 升级 NVIDIA 显卡驱动，让机器支持 CUDA 12.4 或更高版本，然后在项目 conda 环境里安装 CUDA 12.4+ 的 `nvcc`。
2. 保持 CUDA 12.3，但安装更旧、更兼容的 VS2022 MSVC 工具集，例如 MSVC 14.36 / 14.39。

我建议优先走方案 1，因为后续做 LLM 推理、量化、GPU benchmark 时，较新的 CUDA 环境更常见，也更接近招聘岗位里的真实工程环境。

## Step 10：安装兼容 CUDA 12.3 的 MSVC 14.39，并继续推进 CUDA 构建

### 这一步在做什么

上一轮 CUDA 构建失败的原因不是 llama.cpp 写错了，而是工具链版本不匹配：

```text
CUDA 12.3 + VS2026 / 新 MSVC 标准库不兼容
```

因为 NVIDIA 驱动暂时不能升级，所以我们不能简单升级到 CUDA 12.4。正确方案是保留 CUDA 12.3，然后安装一个更旧、更适合 CUDA 12.3 的 MSVC 工具集。

### 做了什么

我们安装了 Visual Studio 2022 Build Tools，并确认存在：

```text
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
```

关键 MSVC 版本：

```text
MSVC 14.39.33519
cl.exe:
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe
```

这一步的意义是：让 CUDA 12.3 的 `nvcc` 能找到一个更老、更兼容的 C++ 宿主编译器。

### 构建推进结果

重新配置 llama.cpp CUDA 构建时，CMake 已经可以识别：

```text
C compiler: MSVC 19.39.33523.0
CXX compiler: MSVC 19.39.33523.0
CUDA compiler: NVIDIA 12.3.107 with host compiler MSVC 19.39.33523.0
CUDA architecture: sm_86
```

这说明之前最大的障碍已经解决了：CUDA 编译器现在能和 MSVC 14.39 正常配合。

### 新发现的问题

构建进入 CUDA 源文件编译阶段后，出现新的问题：

```text
cuda_fp16.hpp: fatal error C1083: 无法打开包括文件: “nv/target”
```

排查发现 `nv/target` 实际在：

```text
d:\pytorch_learning\.conda-envs\pytorch\Library\include\targets\x64\nv\target
```

说明当前 conda 环境里的 CUDA 包不是完整统一的一套：

```text
cuda-nvcc: 12.3.107
cuda-cudart / cuda-cudart-dev: 12.1
cuda-cccl: 12.9
pytorch-cuda: 12.1
```

这就是典型的 CUDA 头文件 / 运行时 / PyTorch 依赖混装问题。

### 为什么不能直接把所有 CUDA 包改成 12.3

尝试统一安装 CUDA 12.3 的 `cuda-cudart` 和 `cuda-cudart-dev` 时，conda 报告冲突：

```text
pytorch-cuda 12.1 requires cuda-cudart >=12.1,<12.2
```

也就是说，当前环境里的 PyTorch 强依赖 CUDA runtime 12.1。如果强行改成 12.3，可能会破坏 PyTorch。

### 这一步对找工作的价值

这一步非常接近真实 AI Infra 工作：

- 不是只会运行模型，而是能处理 C++ / CUDA / Python 环境冲突；
- 能读懂 CMake、Ninja、nvcc、MSVC 的错误；
- 能判断是编译器版本问题、SDK 路径问题，还是 CUDA 包混装问题；
- 能把 PyTorch 训练环境和 llama.cpp 编译环境隔离开。

面试里可以这样说：

```text
在 Windows 环境复现 llama.cpp CUDA 构建时，定位并解决 CUDA 12.3 与新 MSVC 不兼容问题；安装 VS2022 Build Tools/MSVC 14.39 后，成功让 CMake 识别 CUDA 12.3 + MSVC 19.39 工具链，并继续排查 conda CUDA 头文件混装导致的编译错误。
```

### 下一步

不要继续污染当前 PyTorch 环境。下一步应该新建一个独立的 llama.cpp 构建环境，例如：

```text
.conda-envs\llama-build
```

这个环境只负责：

- CUDA 12.3 编译工具链；
- CMake / Ninja；
- llama.cpp 构建；
- 不安装 PyTorch，避免 `pytorch-cuda 12.1` 锁死 CUDA runtime。

这样项目结构会更像真实工程：训练环境和推理引擎构建环境分离。
