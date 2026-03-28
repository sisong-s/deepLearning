import torch
import torch.nn as nn
import math
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.padding_idx = padding_idx
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        return x.reshape(x.size(0), -1, self.n_heads, self.d_k).transpose(1,2)
    
    def make_padding_mask(self, token_ids):
        return (token_ids == self.padding_idx).unsqueeze(1).unsqueeze(2)
    
    def make_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len,seq_len, device=device), diagonal = 1).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def scaled_dot_product_attention(self, q, k, v, mask):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v), attn_weights

    def forward(self, x, token_ids):
        q_proj = self.w_q(x)
        k_proj = self.w_k(x)
        v_proj = self.w_v(x)
        q_split = self.split_heads(q_proj)
        k_split = self.split_heads(k_proj)
        v_split = self.split_heads(v_proj)
        padding_mask = self.make_padding_mask(token_ids)
        causal_mask = self.make_causal_mask(x.size(1), x.device)
        mask = padding_mask | causal_mask
        output, attn_weights = self.scaled_dot_product_attention(q_split,k_split,v_split,mask)
        output = output.transpose(1,2).reshape(x.size(0),-1,self.d_model)
        return self.w_o(output), attn_weights

if __name__ == "__main__":
    token_ids = torch.tensor([[1,2,3,4,5,0],[6,7,8,0,0,0]])
    x = torch.randn(2,6,32)
    self_attn = SelfAttention(d_model=32,n_heads=4,padding_idx=0)
    output, attn_weights = self_attn(x, token_ids)
    # torch.Size([2, 6, 32])
    print(output.shape)
    # torch.Size([2, 4, 6, 6]) b num_heads seq_len seq_len:n*d * d*n = n*n //// 6*8 * 8*6 = 6*6
    print(attn_weights.shape)