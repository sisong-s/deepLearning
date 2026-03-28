# Decoder-Only（GPT 风格）最简实现
# 基于 self_attention_with_padding_mask.py 扩展而来
#
# 与 SelfAttentionWithPaddingMask 的核心差异：
#   1. 新增 Causal Mask（因果掩码）：token 只能关注自己和之前的位置
#   2. Padding Mask 与 Causal Mask 合并（取 OR）
#   3. 增加 FFN 子层 + 残差连接 + LayerNorm → 构成一个 DecoderBlock
#   4. 堆叠 N 个 DecoderBlock + 词嵌入 + 位置编码 → DecoderOnly 模型

import torch
import torch.nn as nn
import math


# ------------------------------------------------------------------ #
#  1. 带 Padding Mask + Causal Mask 的多头自注意力                     #
#     直接复用 SelfAttentionWithPaddingMask，仅在 forward 中增加因果掩码 #
# ------------------------------------------------------------------ #

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, padding_idx=0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.padding_idx = padding_idx

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """(batch, seq, d_model) → (batch, n_heads, seq, d_k)"""
        return x.reshape(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)

    def make_causal_mask(self, seq_len, device):
        """
        生成因果掩码（上三角矩阵，主对角线以上为 True = 需屏蔽）
        (1, 1, seq_len, seq_len)，广播到 (batch, n_heads, seq_q, seq_k)
        """
        # torch.ones 上三角（不含对角线）为 True：位置 j > i 被屏蔽
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, seq_len)

    def make_padding_mask(self, token_ids):
        """
        生成 padding 掩码。
        (batch, seq_len) → (batch, 1, 1, seq_len)，True 表示 padding 位置
        """
        return (token_ids == self.padding_idx).unsqueeze(1).unsqueeze(2)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """缩放点积注意力，mask 为 True 的位置填 -inf"""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v), attn_weights

    def forward(self, x, token_ids=None):
        """
        参数：
            x         : (batch, seq_len, d_model)
            token_ids : (batch, seq_len) 原始 token id，用于生成 padding mask
        """
        B, T, _ = x.shape

        q = self.split_heads(self.w_q(x))   # (B, n_heads, T, d_k)
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        # --- 合并 Causal Mask 与 Padding Mask ---
        causal_mask = self.make_causal_mask(T, x.device)          # (1, 1, T, T)
        if token_ids is not None:
            pad_mask = self.make_padding_mask(token_ids)           # (B, 1, 1, T)
            mask = causal_mask | pad_mask                          # (B, 1, T, T)
        else:
            mask = causal_mask

        attn_out, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 合并多头 → (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, self.d_model)
        return self.w_o(attn_out), attn_weights


# ------------------------------------------------------------------ #
#  2. 单个 Decoder Block：自注意力 + FFN，均带残差 + LayerNorm          #
# ------------------------------------------------------------------ #

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, padding_idx=0):
        super().__init__()
        self.attn    = CausalSelfAttention(d_model, n_heads, padding_idx)
        self.norm1   = nn.LayerNorm(d_model)
        # FFN：两层全连接，中间维度 d_ff（通常 4 × d_model）
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, token_ids=None):
        # 子层1：自注意力 + 残差
        attn_out, attn_weights = self.attn(x, token_ids)
        x = self.norm1(x + self.dropout(attn_out))

        # 子层2：FFN + 残差
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights


# ------------------------------------------------------------------ #
#  3. Decoder-Only 模型                                               #
#     词嵌入 + 位置编码 → N × DecoderBlock → 线性分类头               #
# ------------------------------------------------------------------ #

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len=512, dropout=0.1, padding_idx=0):
        super().__init__()
        self.padding_idx  = padding_idx
        self.token_embed  = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # 可学习的位置嵌入（最简方式，GPT-2 同款）
        self.pos_embed    = nn.Embedding(max_seq_len, d_model)
        self.dropout      = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, padding_idx)
            for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_model)
        # 语言模型头：将隐状态映射回词表，预测下一个 token
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        """
        参数：
            token_ids : (batch, seq_len) —— 输入 token id 序列

        返回：
            logits : (batch, seq_len, vocab_size) —— 每个位置预测下一 token 的分数
        """
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)  # (1, T)

        # 词嵌入 + 位置嵌入
        x = self.dropout(self.token_embed(token_ids) + self.pos_embed(positions))

        # 逐层前向传播，token_ids 透传给每层用于生成 padding mask
        for layer in self.layers:
            x, _ = layer(x, token_ids)

        x = self.norm(x)
        return self.lm_head(x)   # (batch, seq_len, vocab_size)


# ------------------------------------------------------------------ #
#  测试代码                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # ---- 超参数 ----
    VOCAB_SIZE  = 1000
    D_MODEL     = 128
    N_HEADS     = 4
    N_LAYERS    = 2
    D_FF        = 512
    MAX_SEQ_LEN = 32
    PADDING_IDX = 0

    model = DecoderOnly(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN,
        padding_idx=PADDING_IDX,
    )

    # 模拟 batch：2 条句子，长度 10，末尾有 padding
    token_ids = torch.tensor([
        [5, 3, 7, 2, 9, 1, 0, 0, 0, 0],   # 后 4 位是 padding
        [1, 4, 6, 8, 3, 9, 2, 5, 7, 1],   # 无 padding
    ])  # (batch=2, seq_len=10)

    logits = model(token_ids)
    print("logits shape:", logits.shape)   # (2, 10, 1000)

    # 语言模型训练：用 token_ids 的偏移版本作为目标（预测下一个 token）
    # input : token_ids[:, :-1]   (2, 9)
    # target: token_ids[:, 1:]    (2, 9)
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)
    input_ids  = token_ids[:, :-1]   # (2, 9)
    target_ids = token_ids[:, 1:]    # (2, 9)

    out = model(input_ids)           # (2, 9, vocab_size)
    # CrossEntropyLoss 需要 (N, C) 和 (N,)
    loss = criterion(out.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))
    print("loss:", loss.item())

    # 验证 causal mask 效果：第0条句子、第0个 DecoderBlock、第0个头的注意力矩阵
    # 下三角应有值，上三角应≈0
    x_embed = model.dropout(
        model.token_embed(token_ids) + model.pos_embed(
            torch.arange(token_ids.size(1)).unsqueeze(0)
        )
    )
    _, attn_w = model.layers[0].attn(x_embed, token_ids)
    print("\n第0条句子，第0个头，注意力权重矩阵（因果掩码后上三角应≈0）：")
    print(attn_w[0, 0].detach().round(decimals=3))
