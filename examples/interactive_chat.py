#!/usr/bin/env python3
"""
交互式聊天示例 - MLX-LM 中文使用指南

这个示例展示了一个简单的交互式聊天机器人。
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


class ChatBot:
    def __init__(self, model_path: str):
        """初始化聊天机器人"""
        print(f"正在加载模型 {model_path}...")
        self.model, self.tokenizer = load(model_path)
        self.sampler = make_sampler(temp=0.7, top_p=0.9)
        self.history = []
        print("模型加载完成！\n")

    def chat(self, user_input: str) -> str:
        """处理用户输入并生成回复"""
        # 添加用户消息到历史
        self.history.append({"role": "user", "content": user_input})

        # 构建提示词
        prompt = self.tokenizer.apply_chat_template(
            self.history,
            add_generation_prompt=True
        )

        # 生成回复
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            sampler=self.sampler,
            max_tokens=500
        )

        # 添加助手回复到历史
        self.history.append({"role": "assistant", "content": response})

        return response

    def reset(self):
        """清空对话历史"""
        self.history = []
        print("对话历史已清空\n")


def main():
    # 创建聊天机器人
    bot = ChatBot("mlx-community/Qwen3.5-9B-MLX-4bit")

    print("=" * 50)
    print("简单聊天机器人")
    print("输入 'quit' 退出，'clear' 清空历史")
    print("=" * 50 + "\n")

    while True:
        user_input = input("你: ").strip()

        if user_input.lower() == "quit":
            print("再见！")
            break
        elif user_input.lower() == "clear":
            bot.reset()
            continue
        elif not user_input:
            continue

        response = bot.chat(user_input)
        print(f"\n助手: {response}\n")


if __name__ == "__main__":
    main()
