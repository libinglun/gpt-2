import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPTConfig, GPT
from dataloader import GPT2DataLoader


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

train_loader = GPT2DataLoader(B=4, T=32)

model = GPT(GPTConfig())
model.eval().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    x, y = train_loader.next_batch()
    logits, loss = model(x.to(device), y.to(device))
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss = {loss.item()}")

import sys; sys.exit(0)

tokens = encoder.encode("Hello, I'm a language model,")
print(tokens)
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_seq, 1).to(device)

torch.manual_seed(42)
while tokens.size(1) < max_len:
    with torch.no_grad():
        logits = model(tokens)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        next_tokens = torch.gather(topk_indices, -1, ix)                 # (num_seq, 1)
        tokens = torch.cat((tokens, next_tokens), dim=-1)

for i in range(num_seq):
    print(">", encoder.decode(tokens[i].tolist()))

