import json
import torch
import numpy as np
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = self.max_source_length + self.max_target_length

        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    question = json_line["question"]
                    answer = json_line["answer"]
                    self.data.append({
                        "question": question,
                        "answer": answer
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, question, answer):
        messages = [
            {"role": "system", "content": "你是一个医疗方面的专家，可以根据患者的问题进行解答。"},
            {"role": "user", "content": question}
        ]
        # 经历过一段时间对于输入和输出的思考和探索，发现这个代码里的输入和输出格式是暂且发现的最优的方式
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length, truncation=True)
        # 因为是训练，所以有输出
        response = self.tokenizer(answer, add_special_tokens=False, max_length=self.max_target_length, truncation=True)
        # 输入是 question+answer
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        # 输出是 answer，而不去计算question部分的loss，-100 是一个约定俗成的用于忽略损失计算的值。
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
        # 注意！！！这里这三个list的长度是完全一致的，否则无法训练
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]

        input_ids, attention_mask, labels = self.preprocess(**item_data)

        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data)


# tokenizer（分词器）的核心作用是将人类可理解的自然语言转换为模型可处理的数字序列（即 input_ids），这是所有基于 Transformer 的语言模型的通用预处理步骤
# 1. input_ids（输入序列）
# 内容：由 question 对应的 token 序列 + answer 对应的 token 序列 + 填充符（pad_token_id）组成，是模型的完整输入。
# 作用：模型通过读取这个序列，学习 “给定问题，如何生成正确回答” 的映射关系（因果语言模型的核心是 “根据前文预测下一个 token”）。
# 2. attention_mask（注意力掩码）
# 内容：与 input_ids 长度相同的 0/1 序列，1 表示该位置是有效 token（需被模型关注），0 表示是填充 token（pad_token，需被忽略）。
# 作用：告诉模型哪些 token 是真实输入，哪些是为了统一长度而填充的，避免填充符干扰注意力计算和模型学习。
# 3. labels（标签序列，用于计算损失）
# 内容：question 部分的 token 被标记为 -100，answer 部分的 token 保持原始 id，填充符也标记为 -100。
# 作用：在训练时，PyTorch 的 CrossEntropyLoss 会自动忽略值为 -100 的位置，只计算 answer 部分的损失（这是关键！）。
# 因为我们的目标是让模型学习 “根据问题生成回答”，而不是学习 “重复问题”，所以问题部分的损失需要被屏蔽。