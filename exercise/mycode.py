import torch
import torch.nn as nn
import math

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, seq_len, padding_idx):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_ff = d_ff
        self.padding_idx = padding_idx
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx = padding_idx)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embed.weight

    def split_heads(self, x):
        return x.reshape(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)

    def make_causal_mask(self):
        return torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal = 1).bool().unsqueeze(0).unsqueeze(0)
    
    def make_padding_mask(self, token_ids):
        return (token_ids == self.padding_idx).unsqueeze(1).unsqueeze(2)
    
    def scaled_dot_product_attn(self, q, k ,v ,mask):
        attn_score = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.d_k)
        attn_score = attn_score.masked_fill( mask, 1e-9)
        attn_weights = torch.softmax(attn_score, dim=-1)
        return torch.matmul(attn_weights, v), attn_weights

    def forward(self, token_idx):
        B, T = token_idx.shape
        
        pos = torch.arange(T).unsqueeze(0)
        x = self.token_embed(token_idx) + self.pos_embed(pos)

        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        causal_mask = self.make_causal_mask()
        padding_mask = self.make_padding_mask(token_ids)
        mask = causal_mask | padding_mask

        attn_output, attn_weights = self.scaled_dot_product_attn( q, k, v, mask)
        attn_output = attn_output.transpose(1,2).reshape(B, T, self.d_model)

        x = self.norm1(x + self.w_o(attn_output))
        x = self.norm2(x + self.ffn(x))
        logits = self.output_head(x)

        return logits, attn_weights

if __name__ == "__main__":
    model = DecoderOnly(vocab_size=1000, d_model=32, n_heads=4, d_ff= 128, seq_len =5, padding_idx =0)
    token_ids = torch.tensor([[1,2,0,0,0],[4,5,0,0,0]])
    logits , attn_weights =model(token_ids)
    print(logits.shape)
    print(attn_weights.shape)