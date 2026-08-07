 """
scores=torch.matmul(Q, K.transpose(-2, -1))/self.d_k**0.5
# score: (B, num_heads, L, L)
if mask is not None:
    scores = scores + mask
scores = torch.softmax(scores, dim=-1)
scores = self.dropout(scores)
context=torch.matmul(scores, V)
# context: (B, num_heads, L, d_k)
"""