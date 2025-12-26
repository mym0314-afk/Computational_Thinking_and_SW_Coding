import torch
import torch.nn as nn

class EDST(nn.Module):
  def __init__(self, scale=4, num_blocks=16):
    super().__init()
    
    self.head = nn.Conv2d(3, 64, 3, padding=1)
    
    body = []
    for _ in range(num_blocks):
      body.append(nn.Conv2d(64, 64, 3, padding=1))
      body.append(nn.ReLU(inplace=True))
    self.body = nn.Sequential(*body)
    
    self.tail = nn.Sequential(
      nn.Conv2d(64, 3*scale*scale, 3, padding=1),
      nn.PixelShuffle(scale)
    )
    
  def forward(self, x):
    x = self.head(x)
    res = self.body(x)
    x = x + res
    return self.tail(x)