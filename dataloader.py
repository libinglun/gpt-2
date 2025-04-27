import tiktoken
import torch


class GPT2DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("data/input.txt") as f:
            text = f.read()
        encoder = tiktoken.get_encoding('gpt2')
        self.tokens = encoder.encode(text)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[self.current_pos : self.current_pos + B * T + 1])
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B * T

        if self.current_pos + B * T + 1 >= len(self.tokens):
            self.current_pos = 0

        return x, y
