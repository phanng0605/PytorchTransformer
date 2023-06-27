# Pytorch Transformer
This is my implementation of Transformer using Pytorch from scratch, going through the Encoder-Decoder structure.




Here is the structure of it when we go through the details: 

<img src="https://lenngro.github.io/assets/images/2020-11-07-Attention-Is-All-You-Need/transformer-model-architecture.png" alt="Alternative Text" width="500">


<p>

<b>A. Some emergent things to take note</b>

We can obserse that this structure includes:  
1. Transformer Block - which is the main component of the Encoder.
2. Decoder Block - which includes the Masked Multi-Head Attention and the Transformer Block.
3. Multi-Head Attention and Masked Multi-Head Attention - this takes into account the importance of each token to others, making this structure surpasses normal LSTMs or RNNs
4. Positional Embedding - this takes into account the position of each token, making this structure surpasses normal LSTMs or RNNs.

</p>



<p>
<b>B. In case you want to try to implement this on your local computer</b>


To clone this repository, run the following command:

```
git clone https://github.com/phanng0605/PytorchTransformer.git
```

Install required packages:

```
pip install -r requirements.txt
```
</p>
