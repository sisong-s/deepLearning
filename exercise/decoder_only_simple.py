# 最简单版本的 Decoder-Only 模型（GPT 风格）
# 模仿 self_attention_with_padding_mask.py 的代码风格
# 单层：因果自注意力 + FFN + 残差/LayerNorm + LM Head

import torch
import torch.nn as nn
import math


class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, max_seq_len=512, padding_idx=0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.padding_idx = padding_idx

        # 嵌入
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_embed   = nn.Embedding(max_seq_len, d_model)

        # 因果自注意力
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # LayerNorm + FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # 输出头
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享：词嵌入和输出头共用权重
        self.output_head.weight = self.token_embed.weight

    def split_heads(self, x):
        return x.reshape(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)

    def make_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def make_padding_mask(self, token_ids):
        return (token_ids == self.padding_idx).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)

    def scaled_dot_product_attention(self, q, k, v, mask):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v), attn_weights

    def forward(self, token_ids):
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)  # (1, T)

        # 词嵌入 + 位置嵌入
        x = self.token_embed(token_ids) + self.pos_embed(positions)

        # --- 因果自注意力子层 ---
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        causal_mask = self.make_causal_mask(T, x.device)
        pad_mask    = self.make_padding_mask(token_ids)
        mask = causal_mask | pad_mask

        attn_out, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        x = self.norm1(x + self.w_o(attn_out))   # 残差 + LN

        # --- FFN 子层 ---
        x = self.norm2(x + self.ffn(x))           # 残差 + LN

        return self.output_head(x), attn_weights


if __name__ == "__main__":
    VOCAB_SIZE  = 1000
    D_MODEL     = 32
    N_HEADS     = 4
    D_FF        = 128
    MAX_SEQ_LEN = 32

    model = DecoderOnly(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, MAX_SEQ_LEN, padding_idx=0)

    # 模拟含 padding 的 batch
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0],
                               [6, 7, 8, 0, 0, 0, 0]])

    logits, attn_weights = model(token_ids)
    # torch.Size([2, 7, 1000])  batch seq vocab
    print("logits shape:", logits.shape)
    # torch.Size([2, 4, 7, 7])  batch n_heads seq_len seq_len
    print("attn_weights shape:", attn_weights.shape)

    # 验证因果掩码：上三角应全为 0
    print("\nbatch 0, head 0 注意力权重（上三角≈0 说明因果掩码生效）：")
    print(attn_weights[0][0].detach().round(decimals=3))

    # 语言模型 loss：预测下一个 token
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    input_ids  = token_ids[:, :-1]   # (2, 6)
    target_ids = token_ids[:, 1:]    # (2, 6)
    out, _ = model(input_ids)
    loss = criterion(out.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))
    print("\nloss:", loss.item())

#### 2. `.detach()`

# 核心作用：**将张量从计算图中剥离，断开梯度反向传播**

# - 模型推理、打印数值时必须加，否则张量携带梯度`grad_fn`，直接打印会附带计算链路信息，杂乱冗余；
# - 剥离后变成纯数值张量，仅保留数据，不参与后续梯度更新。

# #### 3. `.round(decimals=3)`

# 对张量内所有数值**四舍五入保留 3 位小数**

# nn.Embedding(vocab_size, d_model) 内部存储的权重张量形状是 (vocab_size, d_model) —— 每一行是一个词的嵌入向量
# nn.Linear(d_model, vocab_size) 内部存储的权重张量形状是 (out_features, in_features) = (vocab_size, d_model) —— 每一行对应一个输出神经元的权重

# 1. **unsqueeze**
# 直译：增加维度、升维
# PyTorch 固定术语：**维度扩充**
# 2. **diagonal**
# 直译：对角线、斜线
# 函数参数释义：**对角线偏移量**
# 3. **decimals**
# 直译：小数
# 参数释义：**保留小数位数**