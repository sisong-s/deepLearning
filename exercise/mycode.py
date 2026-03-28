import torch
import torch.nn as nn
import math
class MHA(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_k = dim // n_heads
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

    def split_heads(self,x):
        return x.reshape(x.size(0), -1, self.n_heads, self.d_k).transpose(1,2)

    def scaled_dot_product_attention(self, q, k, v):
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.d_k), dim = -1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output,attn_weights

    def forward(self, q, k, v):
        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)
        q_split = self.split_heads(q_proj)
        k_split = self.split_heads(k_proj)
        v_split = self.split_heads(v_proj)
        output, weights = self.scaled_dot_product_attention(q_split,k_split,v_split)
        
        final_output = output.transpose(1,2).contiguous()
        final_output = self.w_o(final_output.reshape(final_output.size(0), -1, self.dim))
        return final_output, weights

if __name__ == '__main__':
    q = torch.randn(8,16,512)
    k = torch.randn(8,16,512)
    v = torch.randn(8,16,512)
    mha = MHA(512, 8)
    output, weights = mha.forward(q, k ,v)
    print(output.shape, weights.shape)