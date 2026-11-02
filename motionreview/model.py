import torch
from torch import nn
class ForecastNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(15*51,192),nn.ReLU(),nn.Linear(192,192),nn.ReLU(),nn.Linear(192,15*51))
    def forward(self,x):return x[:,-1:]+self.net(x.flatten(1)).reshape(-1,15,51)
