import numpy as np

# fastai
import fastai
from fastai.text import BaseTokenizer, Tokenizer, nn, List, Vocab, Collection
from fastai.text import TokenizeProcessor, NumericalizeProcessor
from fastai.text import TextList, Learner, DatasetType

# transformers
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers import AdamW
from functools import partial

def get_preds_and_labels(learner, dfin, TEXT_COLUMN, Y_COLUMN, transformer_processor, learner_mapping):
    
    learner.data.add_test(TextList.from_df(dfin, cols=TEXT_COLUMN, processor=transformer_processor))
    test_preds, y = learner.get_preds(DatasetType.Test)
    test_preds = test_preds.detach().cpu().numpy()

    #y = y.detach().cpu().numpy()
    y = dfin[Y_COLUMN].tolist()
    best_preds = np.argmax(test_preds, axis=1)
    all_preds_ranked = np.argsort(-test_preds, axis=1)
    #save_pickle(all_preds_ranked, "mapping_10Mall_preds_ranked.pkl")

    learner_mapping = {v:k for k,v in learner_mapping.items()}
    y = np.array([ learner_mapping[v] for v in y])

    return best_preds, all_preds_ranked, y



def compute_accuracy(real_labels, best_preds, all_preds):
    acc_1 = np.mean([1 if r in a[:1] else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
    acc_3 = np.mean([1 if r in a[:3] else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
    acc_5 = np.mean([1 if r in a[:5] else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
    acc_10 = np.mean([1 if r in a[:10] else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
    acc_at_total = np.mean([1 if r in a else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
    return np.round(acc_1,6), np.round(acc_3,6), np.round(acc_5,6), np.round(acc_10,6), np.round(acc_at_total,6)

class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        #if self.model_type in ['bert']:
        #    return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens

class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel, pad_idx):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        self.pad_idx = pad_idx
        
    def forward(self, input_ids, attention_mask=None):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=self.pad_idx).type(input_ids.type()) 
        
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits