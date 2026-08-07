# export CUDA_VISIBLE_DEVICES=2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        theta = 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        positions = torch.arange(self.max_seq_len).float().unsqueeze(1)  # (max_seq_len, 1)
        freqs = positions * inv_freq.unsqueeze(0)                  # (max_seq_len, d_k/2)
        emb = torch.cat((freqs, freqs), dim=-1)                    # (max_seq_len, d_k)
        self.register_buffer('cos_cached', emb.cos())              # (max_seq_len, d_k)
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x):
        """
        x: (B, L, d_model)
        attn_mask: (B, 1, L, L) (True for pad)
        key_padding_mask: (B, L) (True for pad)
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

        context = F.scaled_dot_product_attention(Q, K, V, 
                                                is_causal=True,
                                                #dropout_p=self.dropout.p if self.training else 0.0,
                                                dropout_p=0.0
                                                )
        
        context=context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        # context: (B, L, d_model)
        out=self.out(context)
        # out: (B, L, d_model)
        return out

class FNN_SwiGLU(nn.Module):
    def __init__(self,d_model=200, d_ff=267*2):
        super(FNN_SwiGLU, self).__init__()
        self.linear_gate = nn.Linear(d_model, d_ff,bias=False)
        self.linear_main=nn.Linear(d_model, d_ff,bias=False)
        self.linear_out=nn.Linear(d_ff, d_model,bias=False)
        #self.drop_out=nn.Dropout(dropout)
        self.act=nn.SiLU()
        
    def forward(self, x):
        x_gate=self.linear_gate(x)
        x_main=self.linear_main(x)
        x_out=self.act(x_gate)*x_main
        #x_out=self.drop_out(x_out)
        x_out=self.linear_out(x_out)
        #x_out=self.drop_out(x_out)
        return x_out

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model=200,
                 num_heads=4,
                 num_kv_heads=2,
                 dropout=0.1,
                 max_seq_len=4096,
                 residual_scale=1):
        super(TransformerBlock, self).__init__()
        self.attention = Attention_GQA(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads, max_seq_len=max_seq_len)
        self.fnn=FNN_SwiGLU(d_model=d_model, d_ff=int(d_model*2.67))
        self.norm1=nn.RMSNorm(d_model)
        self.norm2=nn.RMSNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.residual_scale=residual_scale

    def forward(self, x):
        """
        x: (B, L, d_model)
        attn_mask: (B, 1, L, L) (float)
        key_padding_mask: (B, L) (True for padding)
        We guarantee args is like above.
        """
        # Attention
        residual=x
        x=self.norm1(x)
        attn_out=self.attention(x)
        x=residual+self.dropout1(attn_out)*self.residual_scale
        # FNN
        residual=x
        x=self.norm2(x)
        fnn_out=self.fnn(x)
        x=residual+self.dropout2(fnn_out)*self.residual_scale
        return x

class MyModel(nn.Module):
    def __init__(self,
                 vocab_size=32000,
                 num_layers=24,
                 d_model=200,
                 num_heads=4,
                 num_kv_heads=2,
                 dropout=0.1,
                 max_seq_len=4096):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers=nn.ModuleList()
        self.max_seq_len=max_seq_len
        self.d_model=d_model
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model=d_model,
                                                num_heads=num_heads,
                                                num_kv_heads=num_kv_heads,
                                                dropout=dropout,
                                                max_seq_len=max_seq_len))
        self.final_norm=nn.RMSNorm(d_model)
        self.lm_head=nn.Linear(d_model, vocab_size, bias=False)

    def gradient_checkpointing_enable(self,**args):
        print("[model.py] model always has gradient checkpoint")

    def forward(self, input_ids,**kwargs):
        """
        x: (B, L, d_model)
        attn_mask: (B, 1, L, L) (float)
        key_padding_mask: (B, L) (True for padding)
        We guarantee args is like above.
        """
        x=self.embedding(input_ids)* (self.d_model ** 0.5)
        for layer in self.layers:
            if self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x=self.final_norm(x)
        logits=self.lm_head(x)
        return logits

if __name__ == "__main__":
    model = MyModel(vocab_size=32000,
                    num_layers=24,
                    d_model=2048,
                    num_heads=16,
                    num_kv_heads=8,
                    dropout=0.1,
                    max_seq_len=8192).to("cuda")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total/1e9:.2f}B")

    model.eval()
    B, L = 4, 10
    input_ids = torch.randint(0, 32000, (B, L)).to("cuda")

    output = model(input_ids)
    print(output.shape)