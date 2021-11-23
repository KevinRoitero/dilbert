#!/usr/bin/env python
# coding: utf-8

from constants import *
from utils import *

from tqdm import tqdm
import os
from io import open
from pathlib import Path

MODEL_NAME = MODEL.name
seed_all(42)


acc_batch_size = MODEL.pretraining.batch_size*MODEL.pretraining.gradient_accumulation_steps

def find_maxlen(num):
    
    done = False
    while not done:
        train = int(num*0.9)
        real_batches = train//MODEL.pretraining.batch_size+1
        virt_batches = train//acc_batch_size+1

        if real_batches % virt_batches != 0:
            num-=1
        else:
            done = True
    print(num, real_batches, virt_batches)
    return num


if not os.path.exists(MODEL.corpus):
    os.makedirs(MODEL.corpus)
    
paths = [str(x) for x in Path(TOK.corpus).glob("**/*.txt")]
data = paths

print(len(data))
maxlen = find_maxlen(len(data))
data = data[:maxlen]

data_sets = dict()
for path in data:
    k = path.split("/")[-2]
    if k not in data_sets.keys():
        data_sets[k] = []
    data_sets[k].append(path)
print()
print({k:len(v) for k,v in data_sets.items()})


splits = dict()

for k,v in data_sets.items():
    if k not in splits:
        splits[k] = dict()
    splits[k]["train"] = int(len(v)*0.9)
    splits[k]["eval"] = len(v) - splits[k]["train"]

print(splits)


print("all data:", len(data))
print("train data:", sum([v["train"] for v in splits.values()]), [v["train"] for v in splits.values()])
print("eval data:", sum([v["eval"] for v in splits.values()]), [v["eval"] for v in splits.values()])


train_data = []
eval_data = []
for corpus, paths in data_sets.items():
    train_data += paths[:splits[corpus]["train"]]
    eval_data += paths[splits[corpus]["train"]:]
print(len(train_data))
print(len(eval_data))


if not os.path.exists(f"{MODEL.corpus}/train.txt"):
    with open(os.path.join(MODEL.corpus,'train.txt'), 'w', encoding="utf-8") as f:
        cnt = 0
        txt = ""
        for item in tqdm(train_data, desc="creating training dataset"):
            with open(item, "r", encoding="utf-8") as i:
                txt += i.read()+"\n"
            cnt += 1
            if cnt > 100:
                f.write("%s" % txt)
                cnt = 0
                txt = ""


if not os.path.exists(f"{MODEL.corpus}/eval.txt"):
    with open(os.path.join(MODEL.corpus,'eval.txt'), 'w', encoding="utf-8") as f:
        for item in tqdm(eval_data, desc= "creating test dataset"):
            with open(item, "r", encoding="utf-8") as i:
                f.write("%s\n" % i.read())


# Check that PyTorch sees the GPU
import torch
print("cuda available", torch.cuda.is_available())


from transformers import BertTokenizerFast

fast_tokenizer_args = load_json(TOK.fast_tokenizer_args)
tokenizer = BertTokenizerFast.from_pretrained(TOK.out, **fast_tokenizer_args)
print(tokenizer.vocab_size)


from transformers import BertConfig

config = BertConfig(
    vocab_size = TOK.vocab_size,
    hidden_size = MODEL.hidden_size,
    max_position_embeddings = MODEL.max_pos_emb,
    num_attention_heads = MODEL.num_attention_heads,
    num_hidden_layers = MODEL.num_hidden_layers,
    pad_token_id = tokenizer.pad_token_id,
    type_vocab_size = 1,
)
print(config)


from transformers import BertForMaskedLM

model = BertForMaskedLM(config=config)
print(model.num_parameters())  # ~110_104_122


from transformers import LineByLineTextDataset

acc_batch_size = MODEL.pretraining.batch_size*MODEL.pretraining.gradient_accumulation_steps

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=f"{MODEL.corpus}/train.txt",
    block_size=MODEL.pretraining.block_size,
)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=f"{MODEL.corpus}/eval.txt",
    block_size=MODEL.pretraining.block_size,
)




num_tokens_train_dataset = 0
for item in train_dataset:
    num_tokens_train_dataset += len(item)-2
print("Tokens in training dataset:", num_tokens_train_dataset)  # bert: 3.3 * 10^9


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability= MODEL.pretraining.mlm_prob
)



from transformers import Trainer, TrainingArguments

model_output_dir = f"{MODEL.out}/{MODEL.name}"
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)
    
save_steps = MODEL.pretraining.save_steps
if type(save_steps)==str and "epoch" in save_steps:
    mul = save_steps.split("-")[0]
    save_steps = int((len(train_dataset)//acc_batch_size+1) * float(mul))

training_args = TrainingArguments(
    output_dir = model_output_dir,
    overwrite_output_dir = True,
    
    num_train_epochs = MODEL.pretraining.epochs,
    
    learning_rate = 1e-4,
    weight_decay = 0.01,
    warmup_steps = 10_000,
    
    per_device_train_batch_size = MODEL.pretraining.batch_size,
    gradient_accumulation_steps = MODEL.pretraining.gradient_accumulation_steps,
    
    save_total_limit = MODEL.pretraining.save_limit,
    
    evaluate_during_training = True,
    save_steps = save_steps,
    logging_steps = save_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    prediction_loss_only=True,
)
save_const(f"{model_output_dir}/constants.json")

print("eval/save every  ", save_steps, "steps")
print("one epoch has    ", len(train_dataset)//acc_batch_size+1, "steps")
print("saving once every", save_steps/(len(train_dataset)//acc_batch_size+1), "epochs")



trainer.train()


trainer.save_model(model_output_dir)


from shutil import copyfile
copyfile(f"{TOK.out}/vocab.txt", f"{model_output_dir}/vocab.txt")


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=model_output_dir,
    tokenizer=model_output_dir
)


print(fill_mask("I had trouble [MASK] the pain kept me awake all night"))

