# Pytorch Transformer
This is my implementation of Transformer using Pytorch from scratch, going through the Encoder-Decoder structure.


Here is the structure of it when we go through the details: 

<img src="https://lenngro.github.io/assets/images/2020-11-07-Attention-Is-All-You-Need/transformer-model-architecture.png" alt="Alternative Text" width="500">

We can obserse that this includes:  
1. Transformer Block - which is the main component of the Encoder.
2. Decoder Block - which includes the Masked Multi-Head Attention and the Transformer Block.
3. Positional Embedding - this takes into account the position of each token, making this structure surpass normal LSTMs or RNNs.
