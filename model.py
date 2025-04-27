from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    window_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768  # n_embed = d_model = n_head * head_size = 12 * 64 = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer('mask', torch.tril(torch.ones(config.window_size, config.window_size))
                             .view(1, 1, config.window_size, config.window_size))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        q, k, v = self.c_attn(x).split(self.n_embed, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(1,
                                                                                            2)  # (batch_size, n_head, seq_len, head_size)
        k = k.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(1,
                                                                                            2)  # (batch_size, n_head, seq_len, head_size)
        v = v.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(1,
                                                                                            2)  # (batch_size, n_head, seq_len, head_size)

        W_attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (batch_size, n_head, seq_len, seq_len)
        W_attn = W_attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == False,
                                    float('-inf'))  # (batch_size, n_head, seq_len, seq_len)
        W_attn = F.softmax(W_attn, dim=-1)  # (batch_size, n_head, seq_len, seq_len)

        V_attn = W_attn @ v  # (batch_size, n_head, seq_len, head_size)
        V_attn = V_attn.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                          embed_dim)  # (batch_size, seq_len, embed_dim)
        O_attn = self.c_proj(V_attn)  # (batch_size, seq_len, embed_dim)

        return O_attn


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate='tanh')  # don't need to approximate anymore
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        # ---------------------Changes from Transformer Architecture-------------------------#
        # apply layernorm before attention and MLP -- this will create a clean residue stream
        # so that residue wouldn't pass through the layer norm
        # attn -> reduce, mlp -> map: Transformer -> repeated application of map-reduce
        # -----------------------------------------------------------------------------------#
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embed),
            wpe=nn.Embedding(config.window_size, config.n_embed),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing scheme introduced in paper: https://arxiv.org/pdf/1608.05859
        # ~30% parameters saved
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        B, T = x.size()
        assert T <= self.config.window_size, f"Sequence length is too long: {T} > {self.config.window_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embed)
        tok_emb = self.transformer.wte(x)  # (B, T, n_embed)
        x = tok_emb + pos_emb  # (B, T, n_embed)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls):
        """Loads pretrained GPT-2 model weights from huggingface"""
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt2 (124M)")

        config = GPTConfig()
        model = GPT(config)
        state_dict = model.state_dict()
        keys = state_dict.keys()
        keys = [k for k in keys if not k.endswith('.attn.mask')]  # discard this mask / buffer, not a param

        model_pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
        state_dict_pretrained = model_pretrained.state_dict()

        keys_pretrained = state_dict_pretrained.keys()
        keys_pretrained = [k for k in keys_pretrained if not k.endswith('.attn.masked_bias')]
        keys_pretrained = [k for k in keys_pretrained if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        assert len(keys_pretrained) == len(keys_pretrained), f"mismatched keys: {len(keys_pretrained)} != {len(keys)}"

        for k in keys_pretrained:
            if any(k.endswith(w) for w in transposed):
                assert state_dict_pretrained[k].shape[::-1] == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_pretrained[k].t())
            else:
                assert state_dict_pretrained[k].shape == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_pretrained[k])

        return model
