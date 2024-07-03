# BERT 
Implementation of BERT (base) Model 

`Paper: https://arxiv.org/pdf/1810.04805`

The implementation contains BERT model that is only intended to be used as an Encoder. Using BERT as the Decoder isn't supported. 

The implementation uses torch's [SDPA  attention mechanism](https://arxiv.org/pdf/1810.04805) which is an optimized implementation that leverages optimized cuda kernels if the CUDA backend is available.

The implementation is based on the parameter configuration of the BERT base model. Below are the key parameter configuration details (defined in `config.py`)

- Embedding dimension (n_embed): **768**
- No. of Encoder blocks (n_layers): **12**
- No. of Heads in each Encoder block (n_heads): **12**
- Max Sequence Length: **512**

Furthermore, though the entire architecture is constructed end to end, the implementation does not contain the training loop. Instead, the weights are loaded/transferred from the HF's Bert Model. The training loop might be added in future.

To confirm the implementation is accurate, the implementation contains `validate.py` that compares the output from BERT model that is implemented in the code with the output from HF's Bert Model.

