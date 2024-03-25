from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader


@dataclass
class ModelArgs:
    d_model: int = 512 #size of the embedding vectors
    d_ff: int = 2048 #size of the hidden dimensions of the feed forward networks
    N: int=6 #the number of encoder and decoder layers
    dropout: float = 0.1 
    num_heads: int = 8 #the number of heads in each attention block
    vocab_size: int = 50000 #the size of the vocabulary
    pad_idx : int = 0 #index of the padding token 
    max_seq_len: int = 128



class PositionalEncoding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.max_length = args.max_seq_len
        self.encoding = torch.zeros(self.max_length, self.d_model)


        self.pos = torch.arange(0, self.max_length).unsqueeze(1)

        self.denominator = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))

        self.quotient = self.pos * self.denominator

        self.encoding[:, 0::2] = torch.sin(self.quotient)
        self.encoding[:, 1::2] = torch.cos(self.quotient)

        #add dimension for batches
        self.encoding = self.encoding.unsqueeze(0)


    def forward(self, x):
        """Design Choice: Positional encodings are meant to be added to embeddings. We do that here.
        However, this can also be done in the forward method of the Transformer/EncoderDecoder class
        (depending on what you call it).

        Args:
            x (torch.Tensor): 
        Returns:
            torch.Tensor: Sum of positional encodings and embeddings
        """
        #dynamically adjust the seq len by slicing
        return x + self.encoding[ : , :x.size(1)].detach()


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs, attn_mask=False, pad_mask=False):
        super().__init__()
        self.d_out_kq = args.d_model // args.num_heads
        d_out_v = self.d_out_kq
        self.W_q = nn.Linear(args.d_model, self.d_out_kq, bias=False)
        self.W_k = nn.Linear(args.d_model, self.d_out_kq, bias=False)
        self.W_v = nn.Linear(args.d_model, d_out_v, bias=False)
        self.attn_mask = attn_mask
        self.pad_mask=pad_mask
        self.pad_idx = args.pad_idx

    #x_2 for cross attention
    def forward(self, x, encoder_output=None):
        if encoder_output is not None:
            queries = self.W_q(x)
            keys =  self.W_k(encoder_output)
            values =  self.W_v(encoder_output)
        else:
            queries = self.W_q(x)
            keys = self.W_k(x)
            values = self.W_v(x)

        attn_scores = queries @ keys.transpose(-2,-1)
        #for masked attention
        if self.attn_mask:
            
            mask = torch.triu(torch.ones(attn_scores.shape), diagonal=1)
            #print('mask', mask.shape)
            #print('scores', attn_scores.shape)
            attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
            #print(attn_scores)
        

        attn_weights = torch.softmax(attn_scores/self.d_out_kq**0.5, dim=-1)
        context_vector = attn_weights @ values

        return context_vector

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, attn_mask=False):
        super().__init__()
        self.attn_mask=attn_mask
        self.heads = nn.ModuleList([SelfAttention(args, self.attn_mask) for _ in range(args.num_heads)])
    
    #x_2 for cross attention
    def forward(self, x, encoder_output=None):
        if encoder_output is not None:
            return torch.cat([head(x, encoder_output) for head in self.heads], dim=-1)
        else:
            return torch.cat([head(x) for head in self.heads], dim=-1)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in_out, d_hidden, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_in_out, d_hidden)
        self.w2 = nn.Linear(d_hidden, d_in_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.w1(x))
        return self.w2(self.dropout(x))  

class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ff = PositionWiseFeedForward(args.d_model, args.d_ff, args.dropout)
        self.mha = MultiHeadAttention(args)
        self.norm1 = nn.LayerNorm(args.d_model)
        self.norm2 = nn.LayerNorm(args.d_model)

    def forward(self, x):
        x = self.norm1(x + self.mha(x))
        return self.norm2(x+self.ff(x))
    
class Encoder(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.N)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mmha = MultiHeadAttention(args, attn_mask=True)
        self.mhca = MultiHeadAttention(args)
        self.ff = PositionWiseFeedForward(args.d_model, args.d_ff, args.dropout)
        self.norm1 = nn.LayerNorm(args.d_model)
        self.norm2 = nn.LayerNorm(args.d_model)
        self.norm3 = nn.LayerNorm(args.d_model)
    
    def forward(self, x, encoder_output):
        x = self.norm1(x+self.mmha(x))
        x = self.norm2(x + self.mhca(x, encoder_output))
        return self.norm3(x + self.ff(x))

class Decoder(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.N)])
    
    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

class Transformer(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.pe = PositionalEncoding(args)
        self.input_emb = Embeddings(args)
        self.output_emb = Embeddings(args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.linear = nn.Linear(args.d_model, args.vocab_size)

        #for weight sharing
        #self.input_emb.weight = self.linear.weight
        #self.output_emb.weight = self.input_emb.embedding.weight
    
    def forward(self, src, tgt):
        src = self.pe(self.input_emb(src))
        enc_output = self.encoder(src)
        dec_output = self.decoder(self.output_emb(tgt), enc_output)

        return self.linear(dec_output)



