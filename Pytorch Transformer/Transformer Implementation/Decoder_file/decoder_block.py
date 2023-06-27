from self_attention import SelfAttention
from Encoder_file.transformer_block import TransformerBlock
import torch.nn as nn


# DECODER BLOCK
class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_size,
                 heads,
                 forward_expansion,
                 dropout,
                 device):

        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # masked self attention
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))

        # passing this query, value, key to a whole transformer block
        # forward(self, value, key, query, mask):
        out = self.transformer_block(value, key, query, src_mask)
        return out
