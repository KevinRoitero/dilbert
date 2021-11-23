#!/usr/bin/env python
# coding: utf-8

# from constants import *

# =====================================================
# =====================================================

TEST_PATH = "/mnt/sda/disease-language-bert/data/CLEF-output-textified.txt" #"../data/death/text100K.txt"
PRED_FILE_NAME = "predictions_CLEF"
model_type = 'other'  # "other"
TOKENIZER_ROOT = "UNCASED_TRAINING/model_output_files/ICD11"
pretrained_model_name = TOKENIZER_ROOT
learner_name_save = "./by-code/ICD11-checkpoint-1299750"

LOWER_CASE = True  # keep it like this unless you really want it CASED
SAVE_PREDS = True
# =====================================================
# =====================================================

DRAFT = False

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 
import os

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--path", default=None, type=str)
args = parser.parse_args()
if args.path is not None:
    learner_name_save = "./"+args.path
    if os.path.exists("models/"+args.path+"/predictions_CLEF_metrics.txt"):
        print("already calculated")
        exit()


from utils import *


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

from finetuning_utils import TransformersBaseTokenizer, TransformersVocab, CustomTransformerModel
from finetuning_utils import compute_accuracy, get_preds_and_labelsBEA_rev, get_preds_and_labelsBEA

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification

MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig),
    'biobert' : (AutoModelForSequenceClassification,AutoTokenizer,AutoConfig),
    'other' : (AutoModelForSequenceClassification,AutoTokenizer,AutoConfig)
}

print('fastai version :', fastai.__version__)
print('transformers version :', transformers.__version__)

seed = 42
use_fp16 = False

if DRAFT:
    bs = 1
else:
    bs = 16

if DRAFT:
    if "CLEF" in TEST_PATH:
        train =  pd.read_csv(TEST_PATH, sep="\t", encoding='latin-1')[:2]
        train.drop("certificate", axis=1, inplace=True)
        train.rename(columns={"certificate2":"certificate"}, inplace=True)
        
        test =  pd.read_csv(TEST_PATH, sep="\t", encoding='latin-1')[:100]
        test.drop("certificate", axis=1, inplace=True)
        test.rename(columns={"certificate2":"certificate"}, inplace=True)
    else:
        train = pd.read_csv(TEST_PATH, sep="\t")[:2]
        test =  pd.read_csv(TEST_PATH, sep="\t")[:100]
else:
    if "CLEF" in TEST_PATH:
        train =  pd.read_csv(TEST_PATH, sep="\t", encoding='latin-1')[:2]
        train.drop("certificate", axis=1, inplace=True)
        train.rename(columns={"certificate2":"certificate"}, inplace=True)
        
        test =  pd.read_csv(TEST_PATH, sep="\t", encoding='latin-1')
        test.drop("certificate", axis=1, inplace=True)
        test.rename(columns={"certificate2":"certificate"}, inplace=True)
    else:
        train = pd.read_csv(TEST_PATH, sep="\t")[:2]
        test =  pd.read_csv(TEST_PATH, sep="\t")
    
TEXT_COLUMN = 'certificate'
Y_COLUMN = 'UCOD'

learner_mapping = load_json(f"models/{learner_name_save[2:]}/learner_mapping.json")
learner_mapping =  {int(k):v for k,v in learner_mapping.items()}

print(train.shape,test.shape)
categories = list(learner_mapping.values())
test = test[test[Y_COLUMN].isin(categories)]
print(train.shape,test.shape)

print(train.head())
print(test.head())
print()


model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
seed_all(seed)

transformer_tokenizer = tokenizer_class.from_pretrained(TOKENIZER_ROOT, do_lower_case=LOWER_CASE)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])


transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)
tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)
transformer_processor = [tokenize_processor, numericalize_processor]


pad_first = bool(model_type in ['xlnet'])
pad_idx = transformer_tokenizer.pad_token_id


tokens = transformer_tokenizer.tokenize('EXAMPLE of sentence')
print(tokens)
ids = transformer_tokenizer.convert_tokens_to_ids(tokens)
print(ids)
transformer_tokenizer.convert_ids_to_tokens(ids)

databunch = (TextList.from_df(train, cols=TEXT_COLUMN, processor=transformer_processor)
             .split_none()
             .label_from_df(cols= Y_COLUMN)
             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))


print("@@@@@@@@@\n\n\tFinetuning outputs will be saved in", learner_name_save, "\n\n@@@@@@@@@")
config = config_class.from_pretrained(TOKENIZER_ROOT)
config.num_labels = len(categories)
config.use_bfloat16 = use_fp16

config.do_lower_case = True
config.max_seq_length = 256 
config.train_batch_size = 16
config.eval_batch_size = 16
config.gradient_accumulation_steps = 1
config.small = False
config.max_steps = -1
config.num_train_epochs = 4.0
config.device = None
    
seed_all(seed)
transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)
custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model, pad_idx=pad_idx)
    
CustomAdamW = partial(AdamW, correct_bias=False)
learner = Learner(databunch, 
                  custom_transformer_model, 
                  opt_func = CustomAdamW, 
                  metrics=[fastai.text.accuracy])
learner.load(f"{learner_name_save}/learner")

if use_fp16: learner = learner.to_fp16()
    
best_preds, all_preds_ranked, y = get_preds_and_labelsBEA_rev(
    learner, test, TEXT_COLUMN, Y_COLUMN, transformer_processor, learner_mapping)
        
if SAVE_PREDS:
    preds = best_preds.tolist()
    if type(preds[0]) != str:
        preds = [learner_mapping[p] for p in preds]
    test["prediction"] = preds
    test.to_csv(f"models/{learner_name_save[2:]}/{PRED_FILE_NAME}.csv")
    
acc_1, acc_3, acc_5, acc_10, acc_at_total = compute_accuracy(y, best_preds, all_preds_ranked)
print("="*80)
print("========== ACCURACY ON TEST ==========")
print("Accuracy@1: ", acc_1)
print("Accuracy@3: ", acc_3)
print("Accuracy@5: ", acc_5)
print("Accuracy@10: ", acc_10)
print("Accuracy@total: ", acc_at_total)
print()
print("len of test is", len(y))
print("="*80)

with open(f"models/{learner_name_save[2:]}/{PRED_FILE_NAME}_metrics.txt", "w") as f:
    f.write("="*80+"\n")
    f.write("========== ACCURACY ON TEST =========="+"\n")
    f.write("Accuracy@1: "+str( acc_1)+"\n")
    f.write("Accuracy@3: "+str( acc_3)+"\n")
    f.write("Accuracy@5: "+str( acc_5)+"\n")
    f.write("Accuracy@10: "+str( acc_10)+"\n")
    f.write("Accuracy@total: "+str( acc_at_total)+"\n")
    f.write("\n")
    f.write("len of test is "+ str(len(y))+"\n")
    f.write("="*80+"\n")
