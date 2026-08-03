import numpy as np
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers):
        assert num_layers >= 2 and num_layers %2==0
        super(MyModel, self).__init__()
        layers=nn.ModuleList()
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range((num_layers-2)//2):
            layers.append(ResidualBlock(hidden_size))
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers=layers
    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return x



if __name__ == "__main__":
    # Example usage of torch
    a=torch.rand((1,8))  # batch size 1, input size 8
    b=torch.rand((8,16)) # hidden size 16
    c=torch.rand((16,16)) # linnear
    d=torch.rand((16,8)) # output size 8
    out=a@b
    for _ in range(10):
        out=out@c
    out=out@d
    print(out.size())