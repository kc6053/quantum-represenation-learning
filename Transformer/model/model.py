
import torch
import torch.nn as nn
import math
from Transformer.config import *

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that the two can be summed.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # register_buffer is important for moving the tensor to the correct device
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    A standard Transformer model using PyTorch's nn.Transformer module.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        # --- Components ---
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # The core Transformer module from PyTorch
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: ensures input/output tensors are (batch, seq, feature)
        )

        # Final linear layer to map the decoder output to the target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for the Transformer model.
        """
        # 1. Embed and add positional encoding to source and target sequences
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # 2. Pass the sequences through the Transformer
        # The masks are crucial for training:
        # - tgt_mask: Prevents the decoder from "cheating" by looking at future tokens.
        # - *_padding_mask: Prevents the model from paying attention to padding tokens.
        output = self.transformer(src, tgt, 
                                  src_mask=src_mask, 
                                  tgt_mask=tgt_mask, 
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # 3. Pass the output through the final linear layer
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        This is used by the decoder to prevent attending to future tokens.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
