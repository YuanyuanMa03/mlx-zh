#!/usr/bin/env python3
"""
文档摘要示例 - MLX-LM 中文使用指南

这个示例展示了如何使用 MLX-LM 进行文档摘要。
"""

from mlx_lm import load, generate


def summarize_document(model, tokenizer, text: str, max_length: int = 200) -> str:
    """对文档进行摘要"""
    prompt = f"请用简练的语言总结以下文档的主要内容：\n\n{text}"

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    summary = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_length,
        temp=0.5
    )

    return summary


def main():
    print("正在加载模型...")
    model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")
    print("模型加载完成！\n")

    # 示例文档
    document = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，
    但能像人那样思考、也可能超过人的智能。

    机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。
    深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。
    这些技术已经在图像识别、语音识别、自然语言处理等领域取得了重大突破。
    """

    print("原文档:")
    print(document)
    print("\n" + "=" * 50 + "\n")

    summary = summarize_document(model, tokenizer, document)

    print("摘要:")
    print(summary)


if __name__ == "__main__":
    main()
