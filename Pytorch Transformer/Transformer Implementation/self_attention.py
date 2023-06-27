import torch
import torch.nn as nn


# SELF ATTENTION IMPLEMENTATION
class SelfAttention(nn.Module):
    # this function is to set the parameters that are about to be passed into the forward function
    def __init__(self, embed_size, heads):
       # create an instance of this class
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads ==
                embed_size), "Embedding size must be divisibled by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        # get number of training examples
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # matrix multiplication
        energy = torch.einsum("nqhd, nkhd -> nhqk",  [queries, keys])

        # queries shape = (N, query_len, heads, head_dim)
        # keys shape = (N, key_len, heads, head_dim)
        # energy shape = (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float(-1e20))

        # compute attention
        attention = torch.softmax(energy / self.embed_size ** (1/2), dim=3)

        # output
        out = torch.einsum(
            "nhql, nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape = (N, heads, query_len, key_len)
        # values shape = (N, value_len, heads, head_dim)
        # after einsum, output shape = (N, query_len, heads, head_dim), then we flatten its last two dimension (heads*self.head_dim)

        # pass through the fully connected layer
        out = self.fc_out(out)
        return out
