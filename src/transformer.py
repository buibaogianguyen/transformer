import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import json
import os

# HYPERPARAMS
class TransformerConfig:
    def __init__(self):
        self.d_model: int = 512
        self.n_layers: int = 4
        self.n_heads: int = 8
        self.dropout: float = 0.1
        self.d_ff: int = 2048
        self.src_vocab_size: int = 10000
        self.trg_vocab_size: int = 10000
        self.seq_len: int = 256
        self.eps: float = 1e-6

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def tokenizer(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer.tokenize(text)

class LayerNormalization(nn.Module):
    def __init__(self, features, eps):
        super().__init__()

        self.features = features
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) 
        std = x.std(dim=-1, keepdim=True)
        normalized = (x-mean)/(std+self.eps)
        scaledshifted = self.alpha * normalized + self.bias
        return scaledshifted

class Embedder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        scaled_emb = self.embedding(x) * math.sqrt(self.d_model)
        return scaled_emb

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].detach()
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_model, dropout):
        super().__init__()
        assert d_model % num_head == 0

        self.d_model = d_model
        self.num_head = num_head
        self.d_per_h = d_model//num_head
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        for linear in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(linear.weight)

        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(q, k, v, mask, dropout):
        d_per_h = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_per_h)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill_(mask == 0, -1e9)

        scores = scores.softmax(dim = -1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output, scores
    
    def forward(self, q, k, v, mask):
        query = self.Wq(q)
        key = self.Wk(k)
        value = self.Wv(v)

        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_per_h).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_per_h).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_per_h).transpose(1, 2)

        x, attn_scores = self.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_model)

        x = self.Wo(x)

        return x, attn_scores
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self,d_model,heads,dropout, eps, d_ff):
        super().__init__()
        self.norm1 = LayerNormalization(d_model, eps)
        self.norm2 = LayerNormalization(d_model, eps)

        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        norm_x = self.norm1(x)
        attn_output, attn_scores = self.attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout1(attn_output)

        norm_x = self.norm2(x)
        ff_output = self.ff(norm_x)
        x = x + self.dropout2(ff_output)
        
        return x, attn_scores
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, dropout, eps, d_ff):
        super().__init__()
        self.norm1 = LayerNormalization(d_model, eps)
        self.norm2 = LayerNormalization(d_model, eps)
        self.norm3 = LayerNormalization(d_model, eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout)
            
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_output, src_mask, trgt_mask):
        norm_x = self.norm1(x)
        attn_output1, attn_scores1 = self.attn_1(norm_x,norm_x,norm_x,trgt_mask)
        x = x + self.dropout1(attn_output1)

        norm_x = self.norm2(x)
        attn_output2, attn_scores2 = self.attn_2(norm_x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output2)

        norm_x = self.norm3(x)
        ff_output = self.ff(norm_x)
        x = x + self.dropout3(ff_output)

        return x, attn_scores1, attn_scores2
        
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = Embedder(config.d_model, config.src_vocab_size)
        self.pos_encoder = PositionalEncoder(config.d_model, config.seq_len, config.dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(config.d_model, config.n_heads, config.dropout, config.eps, config.d_ff)
            for _ in range(config.n_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src, src_mask):
        x = self.embedder(src)
        x = self.pos_encoder(x)
        attn_scores = []
        for layer in self.layers:
            x, scores = layer(x, src_mask)
            attn_scores.append(scores)
        return x, attn_scores
    
    def create_src_mask(self, src, pad_idx=0):
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = Embedder(config.d_model, config.trg_vocab_size)
        self.pos_encoder = PositionalEncoder(config.d_model, config.seq_len, config.dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(config.d_model, config.n_heads, config.dropout, config.eps, config.d_ff)
            for _ in range(config.n_layers)
        ])
        self.fc_out = nn.Linear(config.d_model, config.trg_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, trg, enc_output, src_mask, trg_mask):
        x = self.embedder(trg)
        x = self.pos_encoder(x)
        attn_scores1, attn_scores2 = [], []
        for layer in self.layers:
            x, scores1, scores2 = layer(x, enc_output, src_mask, trg_mask)
            attn_scores1.append(scores1)
            attn_scores2.append(scores2)
        output = self.fc_out(x)
        return output, attn_scores1, attn_scores2

    def create_trg_mask(self, trg, pad_idx=0):
        trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
        seq_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=trg.device)).bool()
        trg_sub_mask = trg_sub_mask.unsqueeze(0).unsqueeze(0)

        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        enc_output, enc_attn_scores = self.encoder(src, src_mask)
        output, dec_attn_scores1, dec_attn_scores2 = self.decoder(trg, enc_output, src_mask, trg_mask)
        return output, enc_attn_scores, dec_attn_scores1, dec_attn_scores2