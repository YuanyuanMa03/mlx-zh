# 示例代码

本目录包含 MLX-LM 中文使用指南的示例代码。

## 文件列表

### 基础示例

- `simple_chat.py` - 简单对话示例
- `stream_chat.py` - 流式生成示例
- `interactive_chat.py` - 交互式聊天机器人

### 应用示例

- `document_summarizer.py` - 文档摘要工具
- `code_assistant.py` - 代码生成助手

### 配置文件

- `lora_config.yaml` - LoRA 微调配置示例

## 运行示例

### 前置准备

```bash
# 安装 MLX-LM
pip install mlx-lm

# 进入示例目录
cd examples
```

### 运行示例

```bash
# 简单对话
python simple_chat.py

# 流式对话
python stream_chat.py

# 交互式聊天
python interactive_chat.py

# 文档摘要
python document_summarizer.py

# 代码助手
python code_assistant.py
```

## 依赖

所有示例只需要 `mlx-lm`：

```bash
pip install mlx-lm
```

## 自定义示例

### 更换模型

修改示例中的模型路径：

```python
# 使用 Qwen2.5（推荐）
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# 使用更大的模型
model, tokenizer = load("mlx-community/Qwen2.5-14B-Instruct-4bit")

# 使用代码专用模型
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
```

### 调整参数

```python
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=500,    # 最大生成 token 数
    temp=0.7,         # 温度（0-1，越高越随机）
    top_p=0.9,        # Top-p 采样
    top_k=50,         # Top-k 采样
    verbose=False     # 是否显示详情
)
```

## 贡献示例

欢迎提交更多示例！请参考 `CONTRIBUTING.md` 了解贡献指南。
