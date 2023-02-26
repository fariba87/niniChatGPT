import torch
from train import BigramLanguageModel
from utils import encoder_decoder_char
from read_dataset import chars,vocab_size
from ConFig.config import ConfigReader


cfg = ConfigReader()
model = BigramLanguageModel(vocab_size,cfg)
encode, decode = encoder_decoder_char(chars)
model = model.to('cpu')
PATH = './niniGPT.pt'
model.load_state_dict(torch.load(PATH))
model.eval()
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device='cuda')
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
