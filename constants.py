from tokenizers import BertWordPieceTokenizer
from utils import *

class obj():
    def __init__(self):
        pass
    def __repr__(self):
        txt = "{\n"
        for k,v in self.__dict__.items():
            txt += f"  {k}: {v},\n"
        txt += "}"
        return txt
    def to_json(self):
        out = dict()
        for k,v in self.__dict__.items():
            if type(v) == obj:
                new_v = v.to_json()
            elif type(v) == type:
                new_v = "class "+v.__name__
            else:
                new_v = v
            out[k] = new_v
        return out
    def from_json(self, data):
        for k,v in data.items():
            if type(v) == str and v.split()[0] == "class":
                v = eval(v.split()[-1])
            elif type(v) == dict:
                new_obj = obj()
                new_obj.from_json(v)
                v = new_obj
            setattr(self, k, v)

def load_const(path, g):
    data = load_json(path)
    CORPUS = obj(); TOK = obj(); MODEL = obj()
    CORPUS.from_json(data["CORPUS"])
    TOK.from_json(data["TOK"])
    MODEL.from_json(data["MODEL"])
    ROOT = data["ROOT"]
    g["CORPUS"] = CORPUS
    g["TOK"] = TOK
    g["MODEL"] = MODEL
    g["ROOT"] = ROOT
    
def save_const(path):
    data = {
        "ROOT": ROOT,
        "CORPUS": CORPUS.to_json(),
        "TOK": TOK.to_json(),
        "MODEL": MODEL.to_json(),
    }
    save_json(data, path)

CORPUS = obj();
TOK = obj()
MODEL = obj(); MODEL.pretraining = obj()

MODEL.name = "ICD11"
ROOT = "UNCASED_TRAINING/"

# ====================================================

CORPUS.icd = "../data/corpus.csv"
CORPUS.wiki = "../data/corpusWikiComplete.csv"
CORPUS._100k = "../data/death/text100K.txt"
CORPUS._400k = "../data/death/text400K.txt"
CORPUS.all_clean_files = "../data/clean_files_with_indexTerms"
CORPUS.CLEF = '/mnt/sda/disease-language-bert/data/CLEF-output-textified.txt'

# ====================================================

TOK.tok_class = BertWordPieceTokenizer
TOK.corpus = CORPUS.all_clean_files
TOK.out = f"{ROOT}tokenizer_output_files/{TOK.tok_class.__name__}"
TOK.tokenizer_args = f"{TOK.out}/tokenizer_args.json"
TOK.fast_tokenizer_args = f"{TOK.out}/fast_tokenizer_args.json"
TOK.vocab_size = 30522  # 50257 for GPT-2 BBPE, 30522 for bert-base
TOK.min_frequency = 2
TOK.lower = True

# ====================================================

MODEL.corpus = f"{ROOT}model_training_files"
MODEL.out =  f"{ROOT}model_output_files"
MODEL.hidden_size = 768  # bert-base: 768
MODEL.max_pos_emb = 512  # bert-base: 512
MODEL.num_attention_heads = 12  # bert-base: 12
MODEL.num_hidden_layers = 12  # bert-base: 12

# ====================================================

MODEL.pretraining.block_size = 128
MODEL.pretraining.mlm_prob = 0.15  # bert: 15%

MODEL.pretraining.batch_size = 32 # 16  # bert: 256
MODEL.pretraining.gradient_accumulation_steps = 1 #256 // MODEL.pretraining.batch_size

MODEL.pretraining.epochs = 50
MODEL.pretraining.save_limit = None  # None for unlimited checkpoints
MODEL.pretraining.save_steps = "1-epoch"  # -1 to never eval/save during training,
                                          # "X-epoch" to eval/save every X epochs,
                                          # any number to eval/save every X batches

# ====================================================
