#!/usr/bin/env python
# coding: utf-8

from constants import *
from utils import *

import os
from io import open
from tqdm import tqdm
from pathlib import Path


paths = [str(x) for x in Path(CORPUS.all_clean_files).glob("**/*.txt")]


part = [x.split("/")[-2] for x in paths]
from collections import Counter
Counter(part)


tokenizer_save_dir = TOK.out
if not os.path.exists(tokenizer_save_dir):
    os.makedirs(tokenizer_save_dir)
print(tokenizer_save_dir)


tokenizer_args = {
    "lowercase": TOK.lower,
    "cls_token": "[CLS]",
    "pad_token": "[PAD]",
    "sep_token": "[SEP]",
    "unk_token": "[UNK]",
    "mask_token": "[MASK]",
}
fast_tokenizer_args = tokenizer_args.copy()
fast_tokenizer_args["do_lower_case"] = TOK.lower

tokenizer = TOK.tok_class(**tokenizer_args)

tokenizer_args["vocab_file"] = f"{tokenizer_save_dir}/vocab.txt"

save_json(tokenizer_args, TOK.tokenizer_args)
save_json(fast_tokenizer_args, TOK.fast_tokenizer_args)

tokenizer.train(
    files=paths, vocab_size=TOK.vocab_size, min_frequency=TOK.min_frequency,
    special_tokens=["[CLS]","[PAD]","[SEP]","[UNK]","[MASK]",],
)
print(tokenizer.get_vocab_size())


tokenizer.save(tokenizer_save_dir)


print("ORIGINAL")
print(tokenizer)
print(tokenizer.get_vocab_size())
print(tokenizer.encode("This is a sample text").tokens)

tokenizer_args = load_json(TOK.tokenizer_args)
tok2 = TOK.tok_class(**tokenizer_args)
print()
print("RELOADED")
print(tok2)
print(tok2.get_vocab_size())
print(tok2.encode("This is a sample text").tokens)


from transformers import BertTokenizerFast
fast_tokenizer_args = load_json(TOK.fast_tokenizer_args)
tok = BertTokenizerFast.from_pretrained(TOK.out, **fast_tokenizer_args)
print()
print("FAST TOKENIZER (transformers)")
print(tok, tok.unk_token, tok.sep_token, tok.cls_token, tok.pad_token, tok.mask_token)
print(tok.vocab_size)
print(tok.tokenize("This is a sample text", add_special_tokens=True))

