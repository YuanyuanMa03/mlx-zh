# Chinese Guide for MLX-LM

<div align="center">

**Making MLX-LM accessible for Chinese Mac developers**

English | [简体中文](./README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLX](https://img.shields.io/badge/MLX-LM-blue)](https://github.com/ml-explore/mlx-lm)

</div>

---

> This guide is based on the [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) project.
> The original project is licensed under MIT, Copyright © 2023 Apple Inc.

## What is MLX-LM?

**MLX-LM** is a Python package developed by Apple for running and fine-tuning large language models (LLMs) on Apple Silicon Macs.

### Key Features

- **Hugging Face Hub Integration**: Access thousands of LLMs easily
- **Model Quantization**: Support for model quantization and uploading
- **Model Fine-tuning**: LoRA and full model fine-tuning support
- **Distributed Training**: Using `mx.distributed`
- **Apple Silicon Optimized**: Built for performance on Apple chips

### Why MLX-LM for Chinese Developers?

1. **No GPU needed**: Run LLMs on your Mac without expensive hardware
2. **Memory efficient**: Various quantization options
3. **Native support**: Officially maintained by Apple
4. **Chinese model support**: Qwen, Qwen2, ChatGLM, and more

## Quick Start

### Installation

```bash
pip install mlx-lm
```

### Five-minute quickstart

```bash
# Text generation
mlx_lm.generate --prompt "Hello, please introduce yourself"

# Interactive chat
mlx_lm.chat

# Specify a model
mlx_lm.generate --model Qwen/Qwen2.5-7B-Instruct --prompt "写一首关于春天的诗"
```

## Documentation

- [System Requirements](docs/01-系统要求.md)
- [Installation](docs/02-安装方法.md)
- [Command Line Tools](docs/03-命令行工具.md)
- [Python API](docs/04-Python-API.md)
- [Model Fine-tuning](docs/05-模型微调.md)
- [Model Conversion](docs/06-模型转换与量化.md)
- [Long Context Handling](docs/07-长文本处理.md)
- [FAQ](docs/08-常见问题.md)
- [Chinese Model Recommendations](docs/09-中文模型推荐.md)

## Code Examples

### Basic Usage

```python
from mlx_lm import load, generate

# Load model
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Generate text
messages = [{"role": "user", "content": "你好"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

### Streaming

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

for response in stream_generate(model, tokenizer, "Tell me a story", max_tokens=512):
    print(response.text, end="", flush=True)
```

More examples in the [examples](./examples) directory.

## Recommended Chinese Models

| Model | Parameters | Quantized | Best For |
|-------|------------|-----------|----------|
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7B | [4bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) | General chat |
| [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 14B | [4bit](https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit) | Complex tasks |
| [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 32B | [4bit](https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-4bit) | High quality |
| [Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 7B | [4bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit) | Code generation |

## Resources

- [MLX-LM Official Repository](https://github.com/ml-explore/mlx-lm)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX Community (Hugging Face)](https://huggingface.co/mlx-community)
- [Issue Tracker](https://github.com/ml-explore/mlx-lm/issues)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This guide is licensed under the MIT License, same as the original MLX-LM project.

## Acknowledgments

Thanks to all contributors of the [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) project.

---

<div align="center">

Made with ❤️ for Chinese Mac Developers

</div>
