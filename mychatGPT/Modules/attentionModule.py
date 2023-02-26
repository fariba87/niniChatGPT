# version 4: self-attention!
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
# تا اینجا هنوز با هم حرف نمیزنن ارتباط با دات پروداکن میشه
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T) وزن ها صفر نیستن دیگخ

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))  # نمیخایم یونیفورم باشه میخوایم وابسته به داده باشه این کار را با self attention انجام می دیم بالا wei جدید تعریف شده
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v # aggregation ro directly anjam nemidi aval az laye value rad mikoni
#out = wei @ x

out.shape