# 以下是基于 PyTorch 的多头注意力简化版代码，移除了复杂优化（如 dropout、mask），仅保留核心逻辑，便于理解：
 
import torch
import torch.nn as nn
import math

class SimplifiedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # 1. 确保模型维度能被头数整除（核心前提）
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # 关键参数
        self.d_model = d_model  # 输入/输出维度（如 512）
        self.n_heads = n_heads  # 注意力头数量（如 8）
        # 地板除法 执行除法后，对结果进行向下取整（返回不大于商的最大整数）
        # / 返回小数，//返回整数
        self.d_k = d_model // n_heads  # 单个头的维度（如 512/8=64）# 键（Key）的维度
        
        # 2. 线性层：将输入映射为 Q, K, V（共享权重，后拆分多头）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 3. 输出线性层：合并多头后恢复维度
        self.w_o = nn.Linear(d_model, d_model)

    # 辅助函数：拆分多头（batch_size, seq_len, d_model）→（batch_size, n_heads, seq_len, d_k）
    def split_heads(self, x):
        # 先reshape，再转置调整维度顺序（让“头”作为第二维度）
        return x.reshape(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)

    # 核心：计算单头注意力分数
    def scaled_dot_product_attention(self, q, k, v):
        # 1. 计算 Q·K^T（相似度矩阵）
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, n_heads, seq_q, seq_k)
        
        # 2. 缩放：除以 sqrt(d_k)（避免分数过大导致softmax梯度消失）
        attn_scores = attn_scores / math.sqrt(self.d_k)
        
        # 3. Softmax：获取注意力权重（每行和为1）
        # 注意力权重（attn_weights）是 查询序列与键序列之间的相似度分数
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 4. 权重加权 V：得到单头注意力输出
        output = torch.matmul(attn_weights, v)
        return output, attn_weights 

    def forward(self, q, k, v):
        # 1. 线性映射：输入→Q/K/V（维度均为 (batch, seq_len, d_model)）
        q_proj = self.w_q(q) #  Projection
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)
        
        # 2. 拆分多头：维度变为 (batch, n_heads, seq_len, d_k)
        q_split = self.split_heads(q_proj)
        k_split = self.split_heads(k_proj)
        v_split = self.split_heads(v_proj)
        
        # 3. 计算多头注意力（所有头并行计算）
        attn_output, attn_weights = self.scaled_dot_product_attention(q_split, k_split, v_split)
        # attn_output: (batch, n_heads, seq_q, d_k)；attn_weights: (batch, n_heads, seq_q, seq_k)
        
        # 4. 合并多头：转置后reshape，恢复为 (batch, seq_q, d_model)]
        # .contiguous() 是一个用于调整张量内存布局的方法，其核心作用是确保张量在内存中是连续存储的。
        attn_output = attn_output.transpose(1, 2).contiguous()  # 先转置：(batch, seq_q, n_heads, d_k)
        attn_output = attn_output.reshape(attn_output.size(0), -1, self.d_model)  # 再合并：(batch, seq_q, d_model)
        
        # 5. 输出线性层：最终输出（维度与输入一致）
        final_output = self.w_o(attn_output)
        return final_output, attn_weights


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 模拟输入：(batch_size=2, seq_len=10, d_model=512)
    q = torch.randn(2, 10, 512)
    k = torch.randn(2, 10, 512)  # 自注意力中 Q/K/V 输入相同，此处也可设为 q
    v = torch.randn(2, 10, 512)
    
    # 初始化多头注意力（d_model=512，n_heads=8）
    multi_head_attn = SimplifiedMultiHeadAttention(d_model=512, n_heads=8)
    
    # 前向传播
    output, weights = multi_head_attn(q, k, v)
    
    # 验证输出维度（应与输入一致：(2,10,512)）
    print("Output shape:", output.shape)  # torch.Size([2, 10, 512])
    # 验证注意力权重维度（(batch, n_heads, seq_q, seq_k)）
    print("Attention weights shape:", weights.shape)  # torch.Size([2, 8, 10, 10])
 
# 核心逻辑说明（简化版关键步骤）：
 
# 1. 参数初始化：确保  d_model  能被  n_heads  整除，定义 Q/K/V 映射和输出线性层。
# 2. 拆分多头：将 Q/K/V 从  (batch, seq_len, d_model)  拆分为  (batch, n_heads, seq_len, d_k) ，实现“多头并行计算”。
# 3. 缩放点积注意力：计算 Q与K的相似度→缩放→Softmax得权重→加权V，是注意力的核心。
# 4. 合并多头：将多头输出重新拼接为  (batch, seq_len, d_model) ，通过线性层输出最终结果。
 
# 该版本保留了多头注意力的“并行关注不同特征”核心能力，去掉了工程化中的复杂模块，适合理解原理。

# 在注意力机制中，Q、K、V 可能来自不同的序列（例如机器翻译中，Q 是目标语言序列，K 和 V 是源语言序列），此时三者的长度可能不同：
# seq_q：查询序列 Q 的长度；
# seq_k：键序列 K 的长度；
# seq_v：值序列 V 的长度（通常与 seq_k 相同，因为 K 和 V 常来自同一序列）。
# 总结：关系与区别
# 当 Q、K、V 来自同一序列（如自注意力）时，seq_q = seq_k = seq_v = seq_len（此时 seq_q 就是 seq_len 的具体体现）。
# 当 Q、K、V 来自不同序列时，seq_q 仅指代 Q 的长度

# multi_head_attn = multi_head_attn(d_model=512, n_heads=8)  # 实例名与类名相同
# 当你创建实例 multi_head_attn 后，变量 multi_head_attn 指向的是实例对象，而非原来的类。
# 若后续需要再次使用类（如创建新实例 multi_head_attn2 = multi_head_attn(...)），会报错（因为此时 multi_head_attn 是实例，不是类）。

# Python 社区的命名规范（PEP8）中规定：类名使用 “驼峰式命名法”（首字母大写，如 MultiHeadAttn），
# 而函数 / 变量名使用 “蛇形命名法”（全小写，下划线分隔，如 multi_head_attn）。