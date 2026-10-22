# DeepInfer

面向 **Qwen3.5-4B** 的推理框架，基于
[DeepInfer](https://github.com/deepseek-ai/DeepInfer) 精简实现。

```text
LLM → Engine → Scheduler → ModelRunner → Qwen3.5 + HybridCache
            → sample
```

支持分块 prefill、增量 decode、请求轮转、混合缓存与 BF16 / 可选 NF4。
模型计算、缓存、调度和采样由项目实现；Transformers 只用于 tokenizer。

## 运行

```bash
uv run deepinfer --model /path/Qwen3.5-4B '解释一下 KV cache。'
```

首次运行会自动创建 `.venv` 并安装依赖。权重需自行准备，框架只读取本地目录。

需要 NF4 量化时：

```bash
uv run --extra nf4 deepinfer --model /path/Qwen3.5-4B --quantization nf4 '解释一下 KV cache。'
```

Python API：

```python
from deepinfer import LLM, SamplingParams

llm = LLM("/path/Qwen3.5-4B", quantization="none")
outputs = llm.generate(["解释 prefill。", "解释 decode。"], SamplingParams(max_tokens=64))
print(outputs[0].text)
```

当前面向单设备、纯文本；多请求交错执行，未实现张量批处理和分页缓存。
源码按入口、引擎、调度、采样、缓存、模型、加载分为七个模块。

