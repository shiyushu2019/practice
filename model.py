import torch
import torch.nn as nn
from Transformer import TransformerBlock

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
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model=d_model,
                                                num_heads=num_heads,
                                                num_kv_heads=num_kv_heads,
                                                dropout=dropout,
                                                max_seq_len=max_seq_len))
        self.final_norm=nn.LayerNorm(d_model)
        self.lm_head=nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, *, attn_mask, key_padding_mask):
        """
        x: (B, L, d_model)
        attn_mask: (B, 1, L, L) (float)
        key_padding_mask: (B, L) (True for padding)
        We guarantee args is like above.
        """
        x=self.embedding(input_ids)
        for layer in self.layers:
            x=layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x=self.final_norm(x)
        logits=self.lm_head(x)
        return logits

if __name__ == "__main__":
    model = MyModel(vocab_size=32000, num_layers=24, d_model=200, num_heads=4, num_kv_heads=2, dropout=0.1, max_seq_len=4096)
    model.eval()
    B, L = 4, 10
    input_ids = torch.randint(0, 32000, (B, L))

    key_padding_mask = torch.zeros(B, L, dtype=torch.bool)
    key_padding_mask[1, 7:] = True
    key_padding_mask[2, 8:] = True
    key_padding_mask[3, 6:] = True

    causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    attn_base = torch.zeros(L, L, dtype=torch.float32)
    attn_base = attn_base.masked_fill(causal_mask, float('-inf'))
    attn_mask = attn_base.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

    output = model(input_ids, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    print(output.shape)