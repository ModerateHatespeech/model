import torch
import torch.optim as optim

from fastai import *
from fastai.text import *
from fastai.callbacks import *

import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

import pandas as pd
import numpy as np

from unidecode import unidecode
import re

transformer_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Adapted from: https://www.kaggle.com/code/maroberti/fastai-with-transformers-bert-roberta/notebook
# Define model wrapper for FAI learner
class CustomTransformerModel(nn.Module):
  
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel, self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids != transformer_tokenizer.pad_token_id).type(input_ids.type()) 
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits

# Define adaption of FAI Vocab class based on transformers tokenizer
class TransformersVocab(Vocab):
  
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

# Define Tokenizer based on FAI BaseTokenizer class for transformers
class TransformersBaseTokenizer(BaseTokenizer):
  
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'roberta', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = 128 # stop sequence analysis at 128 tokens
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
        tokens = [CLS] + tokens + [SEP]
        return tokens

# Initialize predictor from loaded model
class MHSPredictor:
  
  def __init__(self, model_path: str = "production.pkl"):
    transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
    fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])
    self.model = load_learner("", model_path)
    
  def predict(self, text: str):
    return self.model.predict(self.__preprocess(text))
                              
  def __preprocess(self, text: str):
    # preprocessing routines
    text = re.sub(r'@[^ ]+', '@USER', text)
    text = re.sub(r'https?:\/\/[^ ]+', '<HTTP>', text)
    text = re.sub(r'(?:\n|\r)', ' ', text)
    text = re.sub(r" +", ' ', text)
    text = re.sub('(?: $|^ )', '', text)
    return unidecode(text)

