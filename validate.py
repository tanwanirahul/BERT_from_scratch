'''
Validates BERT implementation by comparing the output of
our BERT model with HuggingFace's BERT implementation.

1. Download the pre-trained weights from the Huggingface.
2. Load the weights in our BERT implementation.
3. Generate the output for a sample seed data from our BERT implementation.
4. Run the same seed data through Huggingface's BERT model.
5. Compare the results from step3 and step4 to validate the implementation.
'''

##BERTBASE (L=12, H=768, A=12, Total Parameters=110M)
import torch
from transformers import BertModel
from models import Bert
from transformers import AutoTokenizer, BertForPreTraining
import torch.nn as nn
import torch

if __name__ == "__main__":
    # Set manual seed to be able to compare results.
    torch.manual_seed(101)
    torch.cuda.manual_seed(101)

    print(f"Loading BERT model from HF.")
    # Load the HF model.
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")

    print(f"Creating our BERT model with weights from HF's model.")
    # Create BERT model from our implementation with the weights initialized from HF's model.
    bert_model = Bert.from_pretrained(dtype=torch.float16)

    model.eval()
    bert_model.eval()

    # Evaluation text.
    text_inputs = ["Hello, my dog is cute", 
                   "You're cute too", 
                   "Delhi is the capital of India", 
                   "what are your plans for this evening?"]

    # Tokenize input texts.
    inputs = tokenizer(text_inputs, padding="max_length", truncation=True, max_length=64, return_tensors="pt")

    print(f"Running evaluations on HF model.")
    hf_outputs = model(**inputs)
    print(f"Running evaluations on our BERT model.")
    outputs = bert_model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    print(f"last hidden state from our model: {last_hidden_states[0][0][:10]}")
    
    hf_last_hidden_states = hf_outputs.last_hidden_state
    print(f"last hidden state from HF model: {hf_last_hidden_states[0][0][:10]}")

    # compare the results.
    is_equal = torch.equal(last_hidden_states, hf_last_hidden_states)
    print(f"\nDo outputs match: {is_equal}")