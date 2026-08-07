import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class Attention_GQA(nn.Module):
    def __init__(self,
                 d_model=200,
                 num_heads=4,
                 num_kv_heads=2,
                 #dropout=0.1,
                 max_seq_len=4096):
        super(Attention_GQA, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.max_seq_len=max_seq_len
        self.num_kv_heads=num_kv_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_model // num_heads % 2 == 0, "d_model/num_heads must be even for RoPE"
        assert d_model % num_kv_heads == 0, "d_model must be divisible by num_kv_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.d_k=d_model//num_heads
        self.kv_dim = num_kv_heads * self.d_k
        #self.dropout=nn.Dropout(dropout)

        self.q=nn.Linear(d_model, d_model,bias=False)
        self.k=nn.Linear(d_model, self.kv_dim,bias=False)
        self.v=nn.Linear(d_model, self.kv_dim,bias=False)
        self.out=nn.Linear(d_model, d_model,bias=False)

        self._RoPE_prepare()
    
    def _RoPE_prepare(self):
        theta = 500000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        positions = torch.arange(self.max_seq_len).float().unsqueeze(1)  # (max_seq_len, 1)
        freqs = positions * inv_freq.unsqueeze(0)                  # (max_seq_len, d_k/2)
        emb = torch.cat((freqs, freqs), dim=-1)                    # (max_seq_len, d_k)
        self.register_buffer('cos_cached', emb.cos())              # (max_seq_len, d_k)
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, *,attn_mask,key_padding_mask):
        """
        x: (B, L, d_model)
        attn_mask: (B, 1, L, L) (float)
        key_padding_mask: (B, L) (True for padding)
        We guarantee args is like above.
        """
        B, L, E = x.size()

        Q=self.q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K=self.k(x).view(B, L, self.num_kv_heads, self.d_k).transpose(1, 2)
        V=self.v(x).view(B, L, self.num_kv_heads, self.d_k).transpose(1, 2)
        # Q/K/V: (B, num_heads, L, d_k)

        cos = self.cos_cached[:L, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:L, :].unsqueeze(0).unsqueeze(0)
        Q = Q * cos + rotate_half(Q) * sin
        K = K * cos + rotate_half(K) * sin
        if self.num_kv_heads != self.num_heads:
            repeat_times = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(repeat_times, dim=1)
            V = V.repeat_interleave(repeat_times, dim=1)

        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(x.dtype) * float('-inf')
        # key_padding_mask: (B, L) -> (B, 1, 1, L)
        attn_mask=attn_mask.to(x.dtype)
        mask = key_padding_mask + attn_mask

        context = F.scaled_dot_product_attention(Q, K, V, 
                                                attn_mask=mask,
                                                is_causal=False, 
                                                #dropout_p=self.dropout.p if self.training else 0.0,
                                                dropout_p=0.0
                                                )
        
        context=context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        # context: (B, L, d_model)
        out=self.out(context)
        # out: (B, L, d_model)
        return out

if __name__ == "__main__":
    model = Attention_GQA(d_model=200, num_heads=4, num_kv_heads=2, 
                        #dropout=0.1,
                        max_seq_len=4096)
    model.eval()
    B, L, D = 4, 10, 200
    x = torch.randn(B, L, D)

    key_padding_mask = torch.zeros(B, L, dtype=torch.bool)
    key_padding_mask[1, 7:] = True
    key_padding_mask[2, 8:] = True
    key_padding_mask[3, 6:] = True

    causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    attn_base = torch.zeros(L, L, dtype=torch.float32)
    attn_base = attn_base.masked_fill(causal_mask, float('-inf'))
    attn_mask = attn_base.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

    output = model(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    print(output.shape)