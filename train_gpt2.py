from dataclasses import dataclass
import math
import torch 
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 6
    n_embed: int = 768    # n_embed = d_model = n_head * head_size = 12 * 64 = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.W_qkv = nn.Linear(config.n_embed, config.n_embed * 3)
        self.W_o = nn.Linear(config.n_embed, config.n_embed) 

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        q, k, v = self.W_qkv(x).split(self.n_embed, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(1, 2)   # (batch_size, n_head, seq_len, head_size)
        k = k.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(1, 2)   # (batch_size, n_head, seq_len, head_size)
        v = v.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(1, 2)   # (batch_size, n_head, seq_len, head_size)

        W_attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                          # (batch_size, n_head, seq_len, seq_len)
        W_attn = W_attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == False, float('-inf'))    # (batch_size, n_head, seq_len, seq_len)
        W_attn = F.softmax(W_attn, dim=-1)                                                          # (batch_size, n_head, seq_len, seq_len)

        V_attn = W_attn @ v                                                                         # (batch_size, n_head, seq_len, head_size)
        V_attn = V_attn.transpose(1, 2).contiguous.view(batch_size, seq_len, embed_dim)             # (batch_size, seq_len, embed_dim)
        O_attn = self.W_o(V_attn)                                                                   # (batch_size, seq_len, embed_dim)

        return O_attn                                                            


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate='tanh')    # don't need apprxoximate anymore
        self.proj = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        #---------------------Changes from Transformer Architecture-------------------------#
        # apply layernorm before attention and MLP -- this will create a clean residue stream
        # residue wouldn't pass through the layer norm
        # attn -> reduce, mlp -> map: Transformer -> repeated application of map-reduce
        #-----------------------------------------------------------------------------------#
        x = x + self.attn(self.layer_norm1(x))  
        x = x + self.mlp(self.layer_norm2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleList(dict(
            token_embedding_weights = nn.Embedding(config.vocab_size, config.n_embed),
            position_embedding_weights = nn.Embedding(config.block_size, config.n_embed),
            hidden_layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layer_norm = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)