import torch
import torch.nn as nn
from FNN import FNN_SwiGLU
from Attention import Attention_GQA

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model=200,
                 num_heads=4,
                 num_kv_heads=2,
                 dropout=0.1,
                 max_seq_len=4096):
        super(TransformerBlock, self).__init__()
        self.attention = Attention_GQA(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads, max_seq_len=max_seq_len)
        self.fnn=FNN_SwiGLU(d_model=d_model, d_ff=int(d_model*2.67))
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

    def forward(self, x, *, attn_mask, key_padding_mask):
        """
        x: (B, L, d_model)
        attn_mask: (B, 1, L, L) (float)
        key_padding_mask: (B, L) (True for padding)
        We guarantee args is like above.
        """
        # Attention
        x=self.norm1(x)
        attn_out=self.attention(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x=x+self.dropout1(attn_out)
        # FNN
        x=self.norm2(x)
        fnn_out=self.fnn(x)
        x=x+self.dropout2(fnn_out)
        return x
        
if __name__ == "__main__":
    model = TransformerBlock(d_model=200, num_heads=4, dropout=0.1, num_kv_heads=2, max_seq_len=4096)
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
        
        
        