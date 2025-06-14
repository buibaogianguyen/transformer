# Transformer Question Answering Model
This project implements a PyTorch based Transformer model that processes input questions and generates answers based on a custom dataset.



# Navigation
- [Attention is all you need](#attention-is-all-you-need)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Model](#model)
  - [Layer Normalization](#layer-normalization)
  - [Multi-Head Attention Mechanism](#multi-head-attention-mechanism)
  - [Encoder](#encoder)
    - [Encoder Block](#encoder-block)
    - [Encoder](#encoder-1)
  - [Decoder](#decoder)
    - [Decoder Block](#decoder-block)
    - [Decoder](#decoder-1)
  - [Transformer](#transformer)


# Attention Is All You Need
This model is based entirely on the deep-learning architecture introduced in the 2017 research paper [Attention is All You Need](https://arxiv.org/abs/1706.03762), authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.

<p align="center">
<img src="https://i.postimg.cc/Bnh9Yj0d/Screenshot-2025-06-14-181649.png" width="500">
</p>

# Project Structure

``` bash
transformer/
├── src/
│   └── transformer.py  # Core model implementation
├── data.json          # Dataset file (input-output pairs)
├── requirements.txt   # Dependencies
├── train.py          # Training script
├── inference.py      # Inference script
└── LICENSE           # MIT License
```

# Setup

Clone the repository:
git clone https://github.com/buibaogianguyen/transformer.git
cd transformer


Install dependencies:
pip install -r requirements.txt


Prepare the dataset:

Ensure a data.json file is present in the root directory with the format:

[
    {"input": "Example question 1", "output": "Example answer 1"},
    {"input": "Example question 2", "output": "Example answer 2"},
    ...
]


# Model

## Layer Normalization

```python
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
```

## Embedder
``` python
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
```

## Positional Encoder
``` python
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
```

## Multi-Head Attention Mechanism

``` python
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
```

## Feedforward

``` python
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
```

## Encoder
### Encoder Block

``` python
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
```

### Encoder

``` python
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
```

## Decoder
### Decoder Block
``` python
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
```
### Decoder
``` python
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
```

## Transformer
``` python
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
```

# Usage
Training
To train the model, run:
python train.py --data_path data.json --model_path model.pth --vocab_path vocabulary.json --epochs 10 --batch_size 32


--data_path: Path to the dataset JSON file.
--model_path: Path to save/load model weights.
--vocab_path: Path to save/load vocabulary.
--epochs: Number of training epochs.
--batch_size: Batch size for training.

The model and vocabulary will be saved to model.pth and vocabulary.json.
Inference
To run inference with a trained model, use:
python inference.py --prompt "What is the tuition fee for grade 12?" --model_path model.pth --vocab_path vocabulary.json --max_len 256


--prompt: Input question for the model.
--model_path: Path to the trained model weights.
--vocab_path: Path to the vocabulary file.
--max_len: Maximum sequence length for inference.

Example
python inference.py --prompt "What is the tuition fee for grade 12?"

Output:
Prompt: What is the tuition fee for grade 12?
Response: [Generated answer]

Model Architecture
The model is a standard Transformer with:

4 encoder and decoder layers
8 attention heads
512-dimensional embeddings
2048-dimensional feed-forward layers
Dropout of 0.1
Maximum sequence length of 256 tokens

It uses a custom vocabulary built from the dataset and the BERT tokenizer for preprocessing.
Requirements

Python 3.8+
PyTorch 2.0.0+
Transformers 4.30.0+

See requirements.txt for the full list.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.
