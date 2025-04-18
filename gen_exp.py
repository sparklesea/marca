import numpy as np
import struct
import math
from struct import pack, unpack
import torch
from torch.nn.functional import silu

def fast_silu(x: torch.Tensor):
    # x = ((1.4426950409*x+126.94201519)*(1<<23)).to(torch.int)
    # return x.view(torch.float32)
    temp = x.clone()
    return temp/(1+(-x).mul_(1.4426950409).add_(126.94201519).mul_(1 << 23).to(torch.int).view(torch.float32))

input = torch.tensor([i for i in range(-7, 3)], dtype=torch.float32)
y_ref = silu(input)
y = fast_silu(input)

print("y: ", y)
print("y_ref: ", y_ref)
# assert torch.allclose(y, y_ref)

# input.mul_(1.4426950409).add_(126.94201519).mul_(1 << 23).to(torch.int).view(torch.float32)
# # print(input.view(torch.float32))
# print(input)