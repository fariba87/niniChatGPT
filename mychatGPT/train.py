import torch
from ConFig.config import ConfigReader
from read_dataset import get_batch
from utils import encoder_decoder_char
#from Modules.loss import estimate_loss

from read_dataset import dataLoader

#from read_dataset import vocab_size,chars
cfg = ConfigReader()
train_data, val_data, vocab_size, chars = dataLoader()
encode, decode = encoder_decoder_char(chars)
torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from model import BigramLanguageModel
def load_model():
    model = BigramLanguageModel(vocab_size, cfg)
    m = model.to(cfg.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    return model , optimizer
model , optimizer = load_model()
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            X = X.to(device)
            Y  = Y.to(device)
            model.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
do_train = False
if do_train:
    for iter in range(cfg.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')
        xb = xb.to(device)
        yb = yb.to(device)
        model = model.to(device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    PATH = './niniGPT.pt'
    torch.save(model.state_dict(), PATH)


# generate from the model
# inference:
context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))  # or m???






