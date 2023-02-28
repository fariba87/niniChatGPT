import torch
import torch
import torch.nn as nn
from torch.nn import functional as F
# super simple bigram model
from Modules.mylayers import Block
from ConFig.config import ConfigReader
cfg1 = ConfigReader()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)  # change to embedding dim 
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, n_head=cfg.n_head) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)  # final layer norm
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)
        self.cfg = cfg

    def forward(self, idx, targets=None):
        idx.to(device)
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C) ) [assume nembed =C]
        tok_emb.to(device)
        '''
        As we changed token_embedding_table, tok_emb is not direct logits now (it will give us the token->we need a dense layer to change it to logits     
        '''

        pos_emb = self.position_embedding_table(torch.arange(T, device=self.cfg.device))  # (T,C) az 0 ta T-1
        x = tok_emb + pos_emb  # (B,T,C) [b,t,c +t , c broadcat emal mishe]
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.cfg.block_size:]  # we never pass in more than block size (the size of pos embedding should be this too)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
