import time

import torch
from torch import nn

from functional import WindowAttentionFunction, WindowValueFunction


device = torch.device("cuda")
dtype = torch.float32
torch.manual_seed(1)
embed_dim = 1024
length = 1024
head = 16
window = 256
ntest = 100
kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}

def run(query, key, value, window, kernel=True):
    if kernel:
        attn_weights = WindowAttentionFunction.apply(query, nn.functional.pad(key, (0, 0, window - 1, 0)), window)
        attn_output = WindowValueFunction.apply(attn_weights, nn.functional.pad(value, (0, 0, window - 1, 0)))
    else:
        attn_weights = query.unsqueeze(3).mul(nn.functional.pad(key, (0, 0, window - 1, 0)).unfold(-2, window, 1).transpose(-1, -2)).sum(dim=-1)
        attn_output = attn_weights.unsqueeze(-1).mul(nn.functional.pad(value, (0, 0, window - 1, 0)).unfold(-2, window, 1).transpose(-1, -2)).sum(dim=-2)

    return attn_output

def correct():
    query = torch.rand(4, head, length, embed_dim // head, **kwargs)
    query_c = query.detach().clone()
    query_c.requires_grad = True

    output = run(query, query, query, window, kernel=True)
    output_c = run(query_c, query_c, query_c, window, kernel=False)
    output.sum().backward()
    output_c.sum().backward()
    forward_correct = ((output - output_c).abs() < 1e-2).all()
    backward_correct = ((query.grad - query_c.grad).abs() < 1e-2).all()
    return forward_correct.item(), backward_correct.item()

def show_time(query, key, value, window, kernel=True):
    times = []
    # GPU warm up
    for _ in range(10):
        res = run(query, key, value, window, kernel)
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        run(query, key, value, window, kernel)
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times)

query = torch.rand(4, head, length, embed_dim // head, **kwargs)
key = torch.rand(4, head, length, embed_dim // head, **kwargs)
value = torch.rand(4, head, length, embed_dim // head, **kwargs)
print("Kernel implementation:", correct())
print("Kernel time:", show_time(query, key, value, window, True))
print("Baseline time:", show_time(query, key, value, window, False))


