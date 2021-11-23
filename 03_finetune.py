#!/usr/bin/env python
# coding: utf-8

# from constants import *

# =====================================================
# =====================================================

TRAIN_PATH = "/mnt/sda/disease-language-bert/data/X-train-CLEF-text.csv"
#"/mnt/sda/disease-language-bert/data/X-train-CLEF-text.csv"
#"../data/death/text400K-by-year.txt"
#"../data/death/text400K-by-code.txt"
TEST_PATH = "/mnt/sda/disease-language-bert/data/X-test-CLEF-text.csv"
#"/mnt/sda/disease-language-bert/data/X-test-CLEF-text.csv"
#"../data/death/text100K-by-year.txt"
#"../data/death/text100K-by-code.txt"
model_type = 'other'  # "other"

"""
--model_type='roberta'    --pre="roberta-base"
--model_type='biobert'    --pre="monologg/biobert_v1.0_pubmed_pmc"
--model_type='biobert'    --pre="emilyalsentzer/Bio_ClinicalBERT"
--model_type='distilbert' --pre="distilbert-base-uncased"
--model_type='xlm'        --pre="xlm-clm-enfr-1024"
--model_type='xlnet'      --pre="xlnet-base-cased"
--model_type='bert'       --pre="bert-base-uncased"
"""

all_checkpoints_root= "UNCASED_TRAINING/model_output_files/ICD11"

SAVE_LEARNER = False
SAVE_PREDS = True
LOWER_CASE = True  # keep it like this unless you really want it CASED
# =====================================================
# =====================================================

DRAFT = False
ARGS = [] # used when running code in jupyter, to pass "cline args"

TEST_NAME = TEST_PATH.split("/")[-1].split(".")[0]

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 
import os

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
from finetuning_utils import compute_accuracy, get_preds_and_labels

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

if ARGS == []:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=None, type=int)
    parser.add_argument("--pre", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    args = parser.parse_args()
else:
    args = obj()
    args.checkpoints = ARGS
    args.pre = None
    args.model_type = None

print('fastai version :', fastai.__version__)
print('transformers version :', transformers.__version__)

if args.model_type is not None:
    model_type = args.model_type
print("model_type", model_type)

seed = 42
use_fp16 = False

if DRAFT:
    bs = 1
else:
    bs = 16

if args.pre is not None:
    checkpoints = [args.pre]
    TOKENIZER_ROOT = args.pre
else:
    print(all_checkpoints_root)
    checkpoints = [str(x) for x in Path(all_checkpoints_root).glob("**/checkpoint-*")]
    def sort_checkpoints(x):
        return int(x.split("-")[-1])
    checkpoints.sort(key= sort_checkpoints)
    if args.checkpoints is not None:
        new_checkpoints = [checkpoints[x-1] for x in args.checkpoints]
        checkpoints = new_checkpoints
    TOKENIZER_ROOT = all_checkpoints_root
print(checkpoints)

results_num = 1
if not os.path.exists("out_"+TOKENIZER_ROOT):
    print("@@@@@@@@@\n\n\tCreating output dir for metrics", "out_"+TOKENIZER_ROOT, "\n\n@@@@@@@@@")
    os.makedirs("out_"+TOKENIZER_ROOT)
out_file_name = f"out_{TOKENIZER_ROOT}/results_{TEST_NAME}_v{results_num}.csv"
while os.path.exists(out_file_name):
    results_num += 1
    out_file_name = f"out_{TOKENIZER_ROOT}/results_{TEST_NAME}_v{results_num}.csv"
print("@@@@@@@@@\n\n\tMetrics will be saved to", out_file_name, "\n\n@@@@@@@@@")

if DRAFT:
    if "CLEF" in TEST_PATH:
        train = pd.read_csv(TRAIN_PATH, sep="\t")[:100]
        
        test =  pd.read_csv(TEST_PATH, sep="\t", encoding='latin-1')[:100]
        test.drop("certificate", axis=1, inplace=True)
        test.rename(columns={"certificate2":"certificate"}, inplace=True)
    else:
        train = pd.read_csv(TRAIN_PATH, sep="\t")[:100]
        test =  pd.read_csv(TEST_PATH, sep="\t")[:100]
else:
    if "CLEF" in TEST_PATH:
        sep = "," if "X-" in TEST_PATH else "\t"
        if "CLEF" in TRAIN_PATH:
            train = pd.read_csv(TRAIN_PATH, sep=sep, encoding='latin-1')
            train.drop("certificate", axis=1, inplace=True)
            train.rename(columns={"certificate2":"certificate"}, inplace=True)
            
        
        test =  pd.read_csv(TEST_PATH, sep=sep, encoding='latin-1')
        test.drop("certificate", axis=1, inplace=True)
        test.rename(columns={"certificate2":"certificate"}, inplace=True)
    else:
        train = pd.read_csv(TRAIN_PATH, sep="\t")
        test =  pd.read_csv(TEST_PATH, sep="\t")
    
TEXT_COLUMN = 'certificate'
Y_COLUMN = 'UCOD'

print(train.shape,test.shape)
categories = np.unique(train[Y_COLUMN])
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

results_df = pd.DataFrame(
    columns=["Checkpoint", "Accuracy@1", "Accuracy@3", "Accuracy@5", "Accuracy@10"]
)

for checkpoint in checkpoints:
    pretrained_model_name = checkpoint
    if "checkpoint" in pretrained_model_name:
        model_name_base = checkpoint.split("/")[-2]
        model_checkpoint_name = checkpoint.split("-")[-1]
    else:
        model_name_base = pretrained_model_name.replace("/", "_")
        model_checkpoint_name = "pretrained"
    leaner_name_save = f"./{model_name_base}-checkpoint-{model_checkpoint_name}"
    print("@@@@@@@@@\n\n\tFinetuning outputs will be saved in", leaner_name_save, "\n\n@@@@@@@@@")
    config = config_class.from_pretrained(pretrained_model_name)
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

    if use_fp16: learner = learner.to_fp16()
    
    if DRAFT:
        pass
    else:
        learner.fit_one_cycle(4, max_lr=3e-5)
    
    learner_mapping = {k:v for k,v in enumerate(learner.data.classes)}
    if not os.path.exists(f"models/{leaner_name_save[2:]}"):
        os.makedirs(f"models/{leaner_name_save[2:]}")
    save_json(learner_mapping, f"models/{leaner_name_save[2:]}/learner_mapping.json")
    if SAVE_LEARNER:
        learner.save(leaner_name_save+"/learner", return_path=True)
    
    best_preds, all_preds_ranked, y = get_preds_and_labels(
        learner, test, TEXT_COLUMN, Y_COLUMN, transformer_processor, learner_mapping)
        
    if SAVE_PREDS:
        preds = best_preds.tolist()
        if type(preds[0]) != str:
            preds = [learner_mapping[p] for p in preds]
        test["prediction"] = preds
        test.to_csv(f"models/{leaner_name_save[2:]}/predictions_on_{TEST_NAME}.csv")
    
    acc_1, acc_3, acc_5, acc_10, acc_at_total = compute_accuracy(y, best_preds, all_preds_ranked)
    
    results_df = results_df.append({
        "Checkpoint": model_checkpoint_name,
        "Accuracy@1": acc_1,
        "Accuracy@3": acc_3,
        "Accuracy@5": acc_5,
        "Accuracy@10": acc_10},
        ignore_index=True)
    
    if DRAFT:
        pass
    else:
        results_df.to_csv(out_file_name, index=None)
    
print(results_df)
results_df.plot()
