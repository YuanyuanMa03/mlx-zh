# Python API 使用

MLX-LM 提供了完整的 Python API，让你可以灵活地将大语言模型集成到你的应用中。

## 基础用法

### 加载模型

```python
from mlx_lm import load, generate

# 加载模型和分词器
model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

# 查看模型信息
print(f"Model loaded: {type(model)}")
print(f"Tokenizer loaded: {type(tokenizer)}")
```

### 简单文本生成

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

# 直接生成
response = generate(
    model,
    tokenizer,
    prompt="你好，请介绍一下你自己",
    verbose=True  # 打印生成过程
)

print(response)
```

### 聊天格式

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

# 使用聊天模板
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "什么是机器学习？"}
]

# 应用聊天模板
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True
)

# 生成回复
response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)
```

## 流式生成

流式生成可以实时显示生成的文本，适合交互式应用。

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

# 准备提示词
prompt = "讲一个关于科技的有趣故事"

# 流式生成
for response in stream_generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512
):
    print(response.text, end="", flush=True)

print()  # 换行
```

### 带进度的流式生成

```python
from mlx_lm import load, stream_generate
import time

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

prompt = "解释深度学习的基本原理"

start_time = time.time()
tokens_generated = 0

for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
    tokens_generated += len(response.text)

elapsed = time.time() - start_time
print(f"\n\n生成 {tokens_generated} tokens，耗时 {elapsed:.2f} 秒")
print(f"速度: {tokens_generated/elapsed:.2f} tokens/秒")
```

## 参数控制

### 采样参数

MLX-LM 使用 `make_sampler` 来控制采样参数：

```python
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

# 创建采样器
sampler = make_sampler(
    temp=0.7,        # 温度：0=贪婪采样，>0=随机采样
    top_p=0.9,       # nucleus sampling
    top_k=50,        # top-k sampling
)

# 使用采样器生成
response = generate(
    model,
    tokenizer,
    prompt="写一首关于春天的诗",
    sampler=sampler,
    max_tokens=200,
    verbose=True
)
```

### 采样参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temp` | float | 0.0 | 温度：0=贪婪采样，>0 越高越随机 |
| `top_p` | float | 0.0 | Nucleus sampling 阈值 (0-1) |
| `top_k` | int | 0 | Top-k sampling (0=禁用) |
| `min_p` | float | 0.0 | 最小概率阈值 |
| `min_tokens_to_keep` | int | 1 | min_p 保留的最小 token 数 |

### 随机种子

```python
import mlx.core as mx

# 设置随机种子
mx.random.seed(42)

response = generate(model, tokenizer, prompt="你好", max_tokens=100)
```

## 批量生成

```python
from mlx_lm import load, generate
from typing import List

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

# 多个提示词
prompts: List[str] = [
    "什么是人工智能？",
    "解释机器学习",
    "深度学习的应用"
]

# 批量生成
responses = []
for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=300
    )
    responses.append(response)

# 输出结果
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}\nA: {response}\n{'-'*50}\n")
```

## 使用自定义配置

```python
from mlx_lm import load

# 使用自定义配置加载模型
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={
        "trust_remote_code": True,
        "eos_token": "<|endoftext|>"
    }
)
```

## 模型转换

```python
from mlx_lm import convert

# 转换并量化模型
repo = "Qwen/Qwen2-7B-Instruct"
upload_repo = "your-username/qwen2-7b-4bit"

convert(
    repo,
    quantize=True,           # 启用量化
    upload_repo=upload_repo, # 上传到 HF
    qlora=False,            # 不使用 QLoRA
    branch="main"           # 分支
)
```

## 实用示例

### 问答助手

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

class QAAssistant:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load(model_path)
        self.sampler = make_sampler(temp=0.7, top_p=0.9)

    def ask(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            sampler=self.sampler,
            max_tokens=500
        )
        return response

# 使用
assistant = QAAssistant("mlx-community/Qwen2-7B-Instruct-4bit")
answer = assistant.ask("什么是量子计算？")
print(answer)
```

### 文本摘要

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

# 创建较低温度的采样器（更确定性）
sampler = make_sampler(temp=0.3, top_p=0.9)

def summarize_text(text: str, max_length: int = 200) -> str:
    prompt = f"请用简练的语言总结以下文本：\n\n{text}"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    summary = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        sampler=sampler,
        max_tokens=max_length
    )
    return summary

# 使用
long_text = "这里是长文本..."
summary = summarize_text(long_text)
print(f"摘要: {summary}")
```

### 代码助手

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

# 代码生成使用低温度（更确定性）
code_sampler = make_sampler(temp=0.2, top_p=0.95)

def generate_code(description: str, language: str = "Python") -> str:
    prompt = f"用{language}写一个{description}"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    code = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        sampler=code_sampler,
        max_tokens=500
    )
    return code

# 使用
code = generate_code("快速排序算法")
print(code)
```

## 多轮对话

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")

class ChatBot:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load(model_path)
        self.history = []

    def chat(self, user_input: str) -> str:
        # 添加用户消息
        self.history.append({"role": "user", "content": user_input})

        # 生成回复
        prompt = self.tokenizer.apply_chat_template(
            self.history, add_generation_prompt=True
        )
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=500
        )

        # 添加助手回复到历史
        self.history.append({"role": "assistant", "content": response})

        return response

    def reset(self):
        self.history = []

# 使用
bot = ChatBot("mlx-community/Qwen2-7B-Instruct-4bit")
print(bot.chat("你好，我是小明"))
print(bot.chat("你还记得我叫什么名字吗？"))
```

## 错误处理

```python
from mlx_lm import load, generate
import warnings

# 加载模型时处理警告
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 生成时处理错误
try:
    response = generate(model, tokenizer, prompt="你好", max_tokens=100)
except Exception as e:
    print(f"生成失败: {e}")
```

## 性能优化

### 使用缓存

```python
from mlx_lm import load, generate
from functools import lru_cache

class CachedModel:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load(model_path)

    @lru_cache(maxsize=128)
    def generate_cached(self, prompt: str) -> str:
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=200
        )
```

### 内存管理

```python
import gc
from mlx_lm import load, generate

# 使用完模型后释放内存
def run_model():
    model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")
    response = generate(model, tokenizer, prompt="你好")
    return response

# 调用后清理
result = run_model()
gc.collect()  # 强制垃圾回收
```

## 下一步

- [模型微调教程](./05-模型微调.md) - 学习如何微调模型
- [长文本处理](./07-长文本处理.md) - 处理长上下文
