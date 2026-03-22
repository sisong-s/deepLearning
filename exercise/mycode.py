import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        return x.reshape(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(self, q, k ,v):
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attm_scores = attn_scores / math.sqrt(self.d_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
    
    def forward(self, q, k ,v):
        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)
        
        q_split = self.split_heads(q_proj)
        k_split = self.split_heads(k_proj)
        v_split = self.split_heads(v_proj)
        
        output, attn_weights = self.scaled_dot_product_attention(q_split, k_split, v_split)
        output = output.transpose(1,2).contiguous()
        output = output.reshape(output.size(0), -1, self.d_model)
        final_output = self.w_o(output)
        return final_output, attn_weights
        
if __name__ == "__main__":
    print(torch.__version__)
    q = torch.randn(2, 10, 512)
    k = torch.randn(2, 10, 512)  # 自注意力中 Q/K/V 输入相同，此处也可设为 q
    v = torch.randn(2, 10, 512)

    multi_head_attn = MultiHeadAttention(d_model = 512, n_heads = 8)

    output, weights = multi_head_attn(q, k, v)
    print(output.shape)
    print(weights.shape)