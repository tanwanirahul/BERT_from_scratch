
from dataclasses import dataclass


@dataclass
class BertConfig:
    '''
        Defines configuration for BERT base model. The base configurations 
        from the paper is as follows:

        # BERTBASE (L=12, H=768, A=12, Total Parameters=110M)
        
        # Paper -> https://arxiv.org/pdf/1810.04805
    '''
    n_embed: int  = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 30522
    max_length: int = 512
    batch_size: int = 256
    inter_dims: int = 3072 ##(4 * n_embed)
    layer_norm_eps: float = 1e-12
    dropout_prob: float = 0.1
    n_segments: int = 2 # No. of segments for Segment Embeddings.
    pad_token_indx: int = 0
