#!/usr/bin/env python3
"""
流式对话示例 - MLX-LM 中文使用指南

这个示例展示了如何使用流式生成，实时显示输出。
"""

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def main():
    print("正在加载模型...")
    model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-4bit")
    print("模型加载完成！\n")

    # 创建采样器
    sampler = make_sampler(temp=0.8, top_p=0.9)

    # 流式生成示例
    prompt = "请讲一个关于科技的有趣故事"

    print(f"提示词: {prompt}\n")
    print("助手: ", end="", flush=True)

    # 流式生成
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=sampler,
        max_tokens=500
    ):
        print(response.text, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    main()
