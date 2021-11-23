import json
import pickle

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

import torch
import random
import numpy as np
from fastai.text import defaults

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    defaults.device = torch.device('cuda') # makes sure the gpu is used

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False