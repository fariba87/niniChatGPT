# read it in to inspect it
from utils import encoder_decoder_char
import torch
from ConFig.config import ConfigReader
cfg = ConfigReader()


def dataLoader():
    with open('../Data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(type(text))
    print("length of dataset in characters: ", len(text))

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print('vocab_size:',vocab_size)
    # example
    encode, decode = encoder_decoder_char(chars)
    data = torch.tensor(encode(text), dtype=torch.long)
    # data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    # print(data[:1000])  # the 1000 characters we looked at earier will to the GPT look like this
# print(encode("hii there"))
# print(decode(encode("hii there")))
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, chars
# let's now encode the entire text dataset and store it into a torch.Tensor


# Let's now split up the data into train and validation sets


# block_size = 8
# train_data[:block_size+1]

# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")
# torch.manual_seed(1337)
# batch_size = 4 # how many independent sequences will we process in parallel?
# block_size = 8 # what is the maximum context length for predictions?
train_data, val_data, vocab_size, chars =dataLoader()
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i:i+cfg.block_size] for i in ix])
    y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])
    return x, y


# xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
#
# print('----')

# for b in range(cfg.batch_size): # batch dimension
#     for t in range(cfg.block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")