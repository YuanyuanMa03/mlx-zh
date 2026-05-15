# MLX-LM 中文使用指南

<div align="center">

**让中国 Mac 开发者轻松运行大语言模型**

[English](./ENGLISH.md) | 简体中文

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLX](https://img.shields.io/badge/MLX-LM-blue)](https://github.com/ml-explore/mlx-lm)

</div>

---

> 本指南基于 [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) 项目编写
> 原项目采用 MIT 许可证，版权所有 © 2023 Apple Inc.

## 项目简介

**MLX-LM** 是一个由 Apple 开发的 Python 包，用于在 Apple 芯片的 Mac 设备上运行和微调大语言模型（LLM）。

### 核心特性

- **与 Hugging Face Hub 集成**：轻松使用数千个 LLM 模型
- **模型量化**：支持模型量化并上传到 Hugging Face Hub
- **模型微调**：支持 LoRA 和全模型微调，包括量化模型
- **分布式推理与训练**：使用 `mx.distributed` 进行分布式处理
- **专为 Apple 芯片优化**：充分利用 Apple Silicon 的性能

### 为什么选择 MLX-LM？

对于中国 Mac 开发者来说，MLX-LM 的优势在于：

1. **无需 GPU**：在 Mac 上即可运行大模型，无需购买昂贵的 GPU
2. **内存效率高**：支持各种量化技术，降低内存占用
3. **原生支持**：Apple 官方维护，与 macOS 深度集成
4. **中文模型支持**：支持 Qwen、Qwen2、Qwen2.5、ChatGLM 等中文模型

---

## 快速开始

### 安装

```bash
# 使用 uv（极速，推荐）⚡
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install mlx-lm

# 使用 pip 安装
pip install mlx-lm

# 或使用 conda
conda install -c conda-forge mlx-lm
```

### 五分钟上手

```bash
# 文本生成
mlx_lm.generate --prompt "你好，请介绍一下你自己"

# 交互式聊天
mlx_lm.chat

# 指定模型
mlx_lm.generate --model Qwen/Qwen3.5-9B-MLX --prompt "写一首关于春天的诗"
```

---

## 文档目录

- [系统要求](docs/01-系统要求.md)
- [安装方法](docs/02-安装方法.md)
- [命令行工具详解](docs/03-命令行工具.md)
- [Python API 使用](docs/04-Python-API.md)
- [模型微调教程](docs/05-模型微调.md)
- [模型转换与量化](docs/06-模型转换与量化.md)
- [长文本处理技巧](docs/07-长文本处理.md)
- [常见问题解答](docs/08-常见问题.md)
- [中文模型推荐](docs/09-中文模型推荐.md)

---

## 代码示例

### 基础用法

```python
from mlx_lm import load, generate

# 加载模型
model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-4bit")

# 生成文本
messages = [{"role": "user", "content": "你好"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

### 流式生成

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-4bit")

for response in stream_generate(model, tokenizer, "讲一个故事", max_tokens=512):
    print(response.text, end="", flush=True)
```

更多示例请查看 [examples](./examples) 目录。

---

## 推荐的中文模型

| 模型 | 参数量 | 量化版本 | 推荐场景 |
|------|--------|----------|----------|
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen3.5-9B-MLX) | 7B | [4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit) | 通用对话 |
| [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 14B | [4bit](https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit) | 复杂任务 |
| [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 32B | [4bit](https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-4bit) | 高质量输出 |
| [Gemma-2-9B-It](https://huggingface.co/google/gemma-2-9b-it) | 9B | [4bit](https://huggingface.co/mlx-community/gemma-2-9b-it-4bit) | 英中双语 |

---

## 常见问题

**Q: 我的 Mac 内存只有 16GB，能运行什么模型？**

A: 建议使用 7B 参数量的 4-bit 量化模型，如 `Qwen2.5-7B-Instruct-4bit`。

**Q: 如何提高生成速度？**

A: 使用较小的模型、确保使用量化版本、或升级到 macOS 15+ 以启用大模型优化。

**Q: 支持哪些中文模型？**

A: MLX-LM 支持 Qwen 系列、ChatGLM 系列、Baichuan 等多种中文模型。

更多问题请查看 [常见问题文档](docs/08-常见问题.md)。

---

## 资源链接

- [MLX-LM 官方仓库](https://github.com/ml-explore/mlx-lm)
- [MLX 官方文档](https://ml-explore.github.io/mlx/)
- [MLX Community (Hugging Face)](https://huggingface.co/mlx-community)
- [问题反馈](https://github.com/ml-explore/mlx-lm/issues)

---

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与贡献。

---

## 许可证

本项目指南采用 MIT 许可证，与原 MLX-LM 项目保持一致。

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 致谢

感谢 [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) 项目的所有贡献者。

---

<div align="center">

Made with ❤️ for Chinese Mac Developers

</div>
