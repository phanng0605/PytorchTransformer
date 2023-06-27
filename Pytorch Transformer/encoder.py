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


# ENCODER IMPLEMENTATIOn
class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(
            x) + self.positional_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
