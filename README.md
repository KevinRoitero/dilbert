# DiLBERT
Repo for the paper "DiLBERT: Cheap Embeddings for Disease Related Medical NLP"

# Pretrained Model

The pretrained model presented in the paper is available on the [huggingface model hub](https://huggingface.co/beatrice-portelli/DiLBERT):

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("beatrice-portelli/DiLBERT")
model = AutoModelForMaskedLM.from_pretrained("beatrice-portelli/DiLBERT")
```

# Contents

- `00_clean_corpus.py` document preprocessing and cleaning
- `01_build_tokenizer.py` build a tokenizer from scratch based on the current corpus
- `02_pretraine_model.py` pretraining script (see `constants.py` for architecture and pretraining parameters)
- `03_finetune.py` finetuning script (classification task)
- `04_test.py` test script (classification task)