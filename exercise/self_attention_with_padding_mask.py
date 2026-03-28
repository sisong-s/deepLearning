# 带 Padding Mask + Causal Mask 的多头自注意力
# 自注意力：Q、K、V 均来自同一序列（x）
# Padding Mask：自动根据 padding_idx（默认 0）生成掩码，屏蔽 padding token 的注意力
# Causal Mask ：上三角掩码，确保每个 token 只能关注自身及之前的位置（防止看到未来）

import torch
import torch.nn as nn
import math


class SelfAttentionWithPaddingMask(nn.Module):
    def __init__(self, d_model, n_heads, padding_idx=0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model      # 模型维度（如 512）
        self.n_heads = n_heads      # 注意力头数（如 8）
        self.d_k = d_model // n_heads  # 单头维度（如 64）
        self.padding_idx = padding_idx  # padding token 的 id（通常为 0）

        # Q、K、V 线性映射层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出线性层
        self.w_o = nn.Linear(d_model, d_model)

    # ---- 辅助函数 ----

    def split_heads(self, x):
        """
        (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        先 reshape 拆分最后一维为 (n_heads, d_k)，再转置让头维度提前
        """
        return x.reshape(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)

    def make_padding_mask(self, token_ids):
        """
        根据输入 token id 序列自动生成 padding mask。

        参数：
            token_ids: (batch, seq_len) —— 原始 token id（整数张量）

        返回：
            mask: (batch, 1, 1, seq_len) —— True 表示该位置是 padding，需要被屏蔽
                  广播后可直接作用于 (batch, n_heads, seq_q, seq_k) 的注意力分数矩阵

        原理：
            padding 位置的 token_id == padding_idx（通常为 0）
            将这些位置标记为 True，后续在 softmax 前填充为 -inf，使其权重趋近于 0
        """
        # (batch, seq_len) → True 表示是 padding 位置
        mask = (token_ids == self.padding_idx)  # (batch, seq_len)
        # 增加两个维度，方便与注意力分数矩阵广播
        return mask.unsqueeze(1).unsqueeze(2)   # (batch, 1, 1, seq_len)

    def make_causal_mask(self, seq_len, device):
        """
        生成因果掩码（Causal Mask / Look-Ahead Mask）。

        参数：
            seq_len: 序列长度
            device : 张量所在设备

        返回：
            mask: (1, 1, seq_len, seq_len) —— 上三角（主对角线以上）为 True，表示需要被屏蔽
                  广播后确保位置 i 只能关注 j <= i 的 key

        原理：
            torch.triu(..., diagonal=1) 取主对角线以上的上三角部分
            位置 (i, j) 为 True 表示 query i 不应关注 key j（因为 j > i，是未来位置）
        """
        # 上三角（不含对角线）为 True：j > i 的位置需要屏蔽
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, seq_len)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        缩放点积注意力（支持 padding mask + causal mask）

        步骤：
            1. Q·K^T → 相似度矩阵
            2. 缩放（/ sqrt(d_k)）
            3. 应用 mask：被屏蔽位置填 -inf，softmax 后权重≈0
            4. Softmax → 注意力权重
            5. 加权 V → 输出

        参数：
            q, k, v : (batch, n_heads, seq_len, d_k)
            mask    : (batch, 1, seq_len, seq_len) 或 None
                      padding mask 与 causal mask 合并后传入
        """
        # Step 1 & 2：计算缩放后的注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1))   # (batch, n_heads, seq_len, seq_len)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # Step 3：应用 padding mask，将 padding 位置的分数设为 -inf
        if mask is not None:
            # mask 为 True 的位置（padding）填充 -1e9，使 softmax 后权重趋近于 0
            # masked_fill_(condition, value)：condition 为 True 的位置填充 value
            # pytorch 对 tensor 的 方法
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        # Step 4：softmax 得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)   # (batch, n_heads, seq_len, seq_len)

        # Step 5：加权 V
        output = torch.matmul(attn_weights, v)              # (batch, n_heads, seq_len, d_k)
        return output, attn_weights

    def forward(self, x, token_ids=None):
        """
        自注意力前向传播：Q、K、V 均来自同一输入 x

        参数：
            x         : (batch, seq_len, d_model) —— 输入嵌入向量
            token_ids : (batch, seq_len)           —— 原始 token id（用于生成 padding mask）
                        若为 None，则不应用 mask

        返回：
            final_output : (batch, seq_len, d_model)
            attn_weights : (batch, n_heads, seq_len, seq_len)
        """
        # Step 1：自注意力中 Q、K、V 都来自同一输入 x
        q_proj = self.w_q(x)   # (batch, seq_len, d_model)
        k_proj = self.w_k(x)
        v_proj = self.w_v(x)

        # Step 2：拆分多头 → (batch, n_heads, seq_len, d_k)
        q_split = self.split_heads(q_proj)
        k_split = self.split_heads(k_proj)
        v_split = self.split_heads(v_proj)

        # Step 3：自动生成并合并 causal mask 与 padding mask
        causal_mask = self.make_causal_mask(x.size(1), x.device)       # (1, 1, seq_len, seq_len)
        if token_ids is not None:
            pad_mask = self.make_padding_mask(token_ids)                # (batch, 1, 1, seq_len)
            mask = causal_mask | pad_mask                               # (batch, 1, seq_len, seq_len)
        else:
            mask = causal_mask
        # mask: True 的位置会在 softmax 前被填充为 -inf

        # Step 4：计算多头注意力（含 mask）
        attn_output, attn_weights = self.scaled_dot_product_attention(
            q_split, k_split, v_split, mask=mask
        )
        # attn_output: (batch, n_heads, seq_len, d_k)

        # Step 5：合并多头 → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()          # (batch, seq_len, n_heads, d_k)
        attn_output = attn_output.reshape(attn_output.size(0), -1, self.d_model)  # (batch, seq_len, d_model)

        # Step 6：输出线性层
        final_output = self.w_o(attn_output)   # (batch, seq_len, d_model)
        return final_output, attn_weights


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 模拟一个 batch：2 条句子，最大长度为 6
    # 句子1：真实长度 4，末尾 2 个 token 是 padding（id=0）
    # 句子2：真实长度 6，无 padding
    token_ids = torch.tensor([
        [5, 3, 7, 2, 0, 0],   # 末尾两个位置是 padding
        [1, 4, 6, 8, 3, 9],   # 无 padding
    ])  # (batch=2, seq_len=6)

    # 模拟 embedding 后的输入（实际使用时来自 nn.Embedding）
    d_model = 512
    x = torch.randn(2, 6, d_model)   # (batch=2, seq_len=6, d_model=512)

    # 初始化自注意力模块（padding_idx=0）
    self_attn = SelfAttentionWithPaddingMask(d_model=512, n_heads=8, padding_idx=0)

    # 前向传播（传入 token_ids 以自动生成 mask）
    output, weights = self_attn(x, token_ids=token_ids)

    print("Output shape     :", output.shape)    # (2, 6, 512)
    print("Attn weights shape:", weights.shape)  # (2, 8, 6, 6)

    # 验证 mask 效果：
    # 1. padding 列（第0条句子第4、5列）权重≈0
    # 2. 上三角（未来位置）权重≈0，即 causal mask 生效
    print("\n第0条句子，第0个头，注意力权重矩阵（行=query，列=key）：")
    print(weights[0, 0].detach().round(decimals=4))
    print("→ 上三角（未来位置）权重应≈0（causal mask）")
    print("→ 第4、5列（padding key 位置）权重应≈0（padding mask）")


# ------------------- 核心逻辑说明 -------------------

# 1. 自注意力：Q/K/V 均来自同一输入 x，forward 只接收一个 x 而非分开的 q/k/v。

# 2. Padding Mask 生成（make_padding_mask）：
#    - 输入 token_ids (batch, seq_len)，对 padding_idx（=0）的位置标记 True
#    - reshape 为 (batch, 1, 1, seq_len)，利用广播自动扩展到 (batch, n_heads, seq_q, seq_k)
#    - 只屏蔽 key 维度（最后一维），意义：任何 query 都不应该"关注" padding 位置的 key

# 3. Causal Mask 生成（make_causal_mask）：
#    - torch.triu(..., diagonal=1) 生成上三角矩阵（主对角线以上为 True）
#    - 位置 (i, j) 为 True 表示 query i 不能关注 key j（j > i，未来位置）
#    - reshape 为 (1, 1, seq_len, seq_len)，广播到所有 batch 和 head

# 4. 两个 Mask 合并：
#    - 在 forward 中：mask = causal_mask | pad_mask（取 OR）
#    - 任一为 True 的位置都会被屏蔽，综合了"不看未来"和"不看 padding"两种约束

# 5. Mask 应用（masked_fill）：
#    - 在 softmax 前，将 mask=True 的位置填充为 -1e9（近似 -inf）
#    - softmax 后这些位置的权重 ≈ 0，等效于忽略这些 token 的信息
