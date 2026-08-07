import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model=100, num_heads=4 ,dropout=0.1):
        super(Attention, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k=d_model//num_heads
        self.dropout=nn.Dropout(dropout)

        self.q=nn.Linear(d_model, d_model,bias=False)
        self.k=nn.Linear(d_model, d_model,bias=False)
        self.v=nn.Linear(d_model, d_model,bias=False)
        self.out=nn.Linear(d_model, d_model,bias=False)
    
    def forward(self, x, *,attn_mask,key_padding_mask):
        """
        x: (B, L, d_model)
        attn_mask: (B, 1, L, L) (float)
        key_padding_mask: (B, L) (True for padding)
        We guarantee args is like above.
        """
        B, L, E = x.size()

        Q=self.q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K=self.k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V=self.v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        # Q/K/V: (B, num_heads, L, d_k)

        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(x.dtype) * float('-inf')
        # key_padding_mask: (B, L) -> (B, 1, 1, L)
        attn_mask=attn_mask.to(x.dtype)
        mask = key_padding_mask + attn_mask

        context = F.scaled_dot_product_attention(Q, K, V, 
                                                attn_mask=mask,
                                                is_causal=False, 
                                                dropout_p=self.dropout.p if self.training else 0.0)
        
        context=context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        # context: (B, L, d_model)
        out=self.out(context)
        # out: (B, L, d_model)
        return out

if __name__ == "__main__":
    model = Attention(d_model=100, num_heads=4, dropout=0.1)
    model.eval()
    B, L, D = 4, 10, 100
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