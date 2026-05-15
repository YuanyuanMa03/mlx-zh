#!/usr/bin/env python3
"""
代码助手示例 - MLX-LM 中文使用指南

这个示例展示了如何使用 MLX-LM 创建一个代码生成助手。
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


class CodeAssistant:
    def __init__(self, model_path: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"):
        """初始化代码助手"""
        print("正在加载代码模型...")
        self.model, self.tokenizer = load(model_path)
        # 代码生成使用低温度（更确定性）
        self.code_sampler = make_sampler(temp=0.2, top_p=0.95)
        # 代码解释使用稍高的温度
        self.explain_sampler = make_sampler(temp=0.4, top_p=0.9)
        print("模型加载完成！\n")

    def generate_code(self, description: str, language: str = "Python") -> str:
        """根据描述生成代码"""
        prompt = f"用{language}写一个{description}，请添加必要的注释。"

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        code = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            sampler=self.code_sampler,
            max_tokens=500
        )

        return code

    def explain_code(self, code: str) -> str:
        """解释代码"""
        prompt = f"请解释以下代码的功能：\n\n```python\n{code}\n```"

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        explanation = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            sampler=self.explain_sampler,
            max_tokens=400
        )

        return explanation


def main():
    assistant = CodeAssistant()

    print("=" * 50)
    print("代码生成助手")
    print("=" * 50 + "\n")

    # 示例1: 生成代码
    print("【生成代码示例】\n")

    tasks = [
        "快速排序算法",
        "二分查找函数",
        "简单的计算器类",
    ]

    for task in tasks:
        print(f"任务: {task}")
        code = assistant.generate_code(task)
        print(f"代码:\n{code}\n")
        print("-" * 50 + "\n")

    # 示例2: 解释代码
    print("\n【代码解释示例】\n")

    sample_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
    """

    print("待解释代码:")
    print(sample_code)
    print()

    explanation = assistant.explain_code(sample_code.strip())
    print(f"解释: {explanation}")


if __name__ == "__main__":
    main()
