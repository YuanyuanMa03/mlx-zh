#!/usr/bin/env python3
"""
简单的对话示例 - MLX-LM 中文使用指南

这个示例展示了如何使用 MLX-LM 进行简单的对话。
"""

from mlx_lm import load, generate


def main():
    # 加载模型（使用中文模型）
    print("正在加载模型...")
    model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")
    print("模型加载完成！\n")

    # 简单对话
    questions = [
        "你好，请介绍一下你自己",
        "什么是机器学习？",
        "用Python写一个Hello World",
    ]

    for question in questions:
        print(f"用户: {question}")

        # 使用聊天模板
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # 生成回复
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=300,
            temp=0.7,
            verbose=False
        )

        print(f"助手: {response}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
