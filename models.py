'''
Contains the implementation of the BERT model architecture.
'''
import torch
import torch.nn as nn

from config import BertConfig
from dataclasses import dataclass
from transformers import BertModel

@dataclass
class BertModelOutput:
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None

class BertEmbedding(nn.Module):
    '''
        Embedding Layer of the BERT model. Implements the logic
        for Token Embeddings, Segment Embedding and Positional Embeddings.
    '''
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed, padding_idx=config.pad_token_indx)
        self.position_embeddings = nn.Embedding(config.max_length, config.n_embed)
        self.token_type_embeddings = nn.Embedding(config.n_segments, config.n_embed)

        self.LayerNorm = nn.LayerNorm(config.n_embed, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_length).expand((1,-1)), persistent=False)
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(self, input_ids, token_type_ids):
        '''
            On forward pass, we will run the input_ids through word_embeddings, and sum it
            up with positional embeddings and token_type_embeddings/segment embeddings. 
        '''
        assert input_ids is not None, f"Expect input_ids to be not None"
        
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:,:seq_length].expand(input_shape[0], seq_length)
        
        position_ids = self.position_ids[:, :seq_length]

        token_embeddings = self.word_embeddings(input_ids)
        segment_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)

class SelfAttention(nn.Module):
    '''
        Implements self attention module of the transformer.
    '''
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        assert config.n_embed % config.n_heads == 0, f"Embedding dimension of {config.n_embed} cannot be evenly divided among {config.n_heads} heads!"
        self.n_heads = config.n_heads
        self.head_size = config.n_embed // self.n_heads

        # Attention in HF is broken up into 2 classes:
        # 1. Self Attention and Self Output. Instead of creating
        # two separate classes, we are putting them into a ModuleDict instead.
        
        # Define the parameters for QKV of the attention module.
        self.self = nn.ModuleDict(dict(
            query = nn.Linear(config.n_embed, config.n_embed),
            key = nn.Linear(config.n_embed, config.n_embed),
            value = nn.Linear(config.n_embed, config.n_embed),
            dropout = nn.Dropout(p=config.dropout_prob)
        ))

        # Self attention output linear layer.
        self.output = nn.ModuleDict(dict(
            dense = nn.Linear(config.n_embed, config.n_embed),
            LayerNorm = nn.LayerNorm(config.n_embed, eps=config.layer_norm_eps),
            dropout = nn.Dropout(config.dropout_prob)
        ))

    def forward(self, hidden_state, attention_mask):
        '''
            Forward pass for self atttention layer.
        '''

        batch_size, seq_length, n_embd = hidden_state.shape

        q = self.self.query(hidden_state)
        k = self.self.key(hidden_state)
        v = self.self.value(hidden_state)

        # Convert from B, Seq, Embd -> B, Seq, n_heads, n_embd/n_heads.
        q = q.view(batch_size, seq_length, self.n_heads, n_embd // self.n_heads).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_heads, n_embd // self.n_heads).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_heads, n_embd // self.n_heads).transpose(1, 2)

        # convert attention mask from B, Seq -> B, 1, Seq, Seq
        expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length).to(hidden_state.dtype)
        expanded_mask = expanded_mask.masked_fill(expanded_mask==0, torch.finfo(hidden_state.dtype).min)

        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, expanded_mask, is_causal=False)
        attn_output = attn_output.transpose(1,2).view(batch_size, seq_length, n_embd)
        
        attn_layer_output = self.output.dropout(self.output.dense(attn_output))
        attn_layer_output = self.output.LayerNorm(attn_layer_output + hidden_state)

        return attn_layer_output
        
class BertIntermediate(nn.Module):
    '''
        First fully connected layer of the Feed Forward step of encoder.
    '''
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.n_embed, config.inter_dims)
        self.activation = nn.GELU()

    def forward(self, hidden_state):
        '''
            Forward pass for the first fully connected in FeedForward block.
        '''
        return self.activation(self.dense(hidden_state))

class BertOutput(nn.Module):
    '''
        Final fully connected layer of Feed Forward step of encoder.
    '''
    def __init__(self, config:BertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.inter_dims, config.n_embed)
        self.LayerNorm = nn.LayerNorm(config.n_embed, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.dropout_prob)
        
    def forward(self, hidden_state, input_state):
        '''
            Second fully connected layer of feed forward.
            hidden_state: output of the first fully connected.
            input_state: input to the feed forward block to implement residual
            connection.
        '''
        output = self.dropout(self.dense(hidden_state))
        output = self.LayerNorm(output + input_state)
        return output

class BertLayer(nn.Module):
    '''
        Implments the single layer of the BERT encoder block.
    '''
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.attention = SelfAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_state, attention_mask):
        '''
            Forward pass for the Bert Layer.
        '''
        attn_output = self.attention(hidden_state, attention_mask)
        inter_output = self.intermediate(attn_output)
        layer_output = self.output(inter_output, attn_output)

        return layer_output

class BertEncoder(nn.Module):
    '''
        Implements the BERT Encoder block.
    '''
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.n_layers)])

    def forward(self, hidden_state, attention_mask):
        '''
            Runs the input tensor with embedding through the Encoder layers.
        '''
        for i, bert_layer in enumerate(self.layer):
            layer_outputs = bert_layer(hidden_state, attention_mask)
            hidden_state = layer_outputs

        return hidden_state

class BertPooler(nn.Module):
    '''
        Implements the BERT's final Dense layer to pool the output
        from the cls token.
        The output of the BERT model is the hidden state captured in the 
        cls Token.
    '''
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.n_embed, config.n_embed)
        self.activation = nn.Tanh()

    def forward(self, x):
        '''
            Forward for the final linear layer.
        '''
        return self.activation(self.dense(x[:, 0]))
        
class Bert(nn.Module):
    '''
        End to End implementation for BERT model. We only implement the base BERT
        model without any LM heads on top. To pre-train the model with LM heads for specific
        task, we will have to extend the BertModel with additional head layers.
        
        See HF documentation for BertModel and BertForPreTraining
    '''
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        '''
            Forward pass for the bert model. 
            input_ids: Tokenized input ids upto max length.
            token_type_ids: For segment information.
            attention_mask: Attention Mask is 1 for tokens that need to be processes and 0 for padding tokens.
        '''
        batch_size, seq_length = input_ids.shape

        embeddings = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embeddings, attention_mask)
        pooler_output = self.pooler(encoder_outputs)
        return BertModelOutput(last_hidden_state=encoder_outputs, pooler_output=pooler_output)
    
    @classmethod
    def from_pretrained(cls, dtype):
        '''
            Loads the weights from HF's BertModel onto our model.
        '''
        # Instantiate our implementation of the BERT model.
        model = Bert(BertConfig())

        model.to(dtype=dtype)

        # Instantiate the Bert model from HF.
        hf_model = BertModel.from_pretrained("google-bert/bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")

        # Assert both models contain the exact same state dict representation.
        sd = model.state_dict()
        hf_sd = hf_model.state_dict()
        assert len(sd.keys()) == len(hf_sd.keys()), f"Keys Mismatch!"
        
        for k in sd.keys():
            assert sd[k].shape == hf_sd[k].shape, f"Shape mismatch for key: {k}"
            assert sd[k].dtype == hf_sd[k].dtype, f"Type for the keys didn't match. Expected: {sd[k].dtype}; Found:{hf_sd[k].dtype}"
            with torch.no_grad():
                sd[k].copy_(hf_sd[k])

        return model