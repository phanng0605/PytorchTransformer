from self_attention import SelfAttention
import torch.nn as nn


# TRANSFORMER BLOCK (USE AS A WHOLE ENCODER BLOCK)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        # normalization layer
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # feed forward layer include resizing the input size & apply reLU
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # call the forward method (since the SelfAttention class has only 1 method)
        attention = self.attention(value, key, query, mask)

        # Add & Normalize (including skip connection)
        x = self.dropout(self.norm1(attention + query))

        forward = self.feed_forward(x)

        out = self.dropout(self.norm2(x + forward))

        return out
