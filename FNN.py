import torch
import torch.nn as nn

class FNN_SwiGLU(nn.Module):
    def __init__(self,d_model=100, d_ff=267):
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

if __name__ == "__main__":
    model = FNN_SwiGLU(d_model=100, d_ff=267)
    x = torch.randn(32, 100)  # Example input with batch size 32 and feature size 100
    output = model(x)
    print(output.shape)  # Should print torch.Size([32, 100])
