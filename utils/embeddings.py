
from utils.paths import *
import pandas as pd
import numpy as np
def convert_to_array(s):
    # Remove brackets, replace '\n' with space, split by space, and convert to float
    return np.array([float(x) for x in s.strip('[]').replace('\n', ' ').split()])

def load_embeddings(embed_type:str):
    """"
    optinos for embed_type = ['GloVe',"BERT"]
    """
    # load embeddings
    path = annot_path /'EN'/f'lppEN_word_embeddings_{embed_type}.csv'
    # path = f'F:\\dataset\\annotation\\EN\\lppEN_word_embeddings_{embed_type}.csv'
    df = pd.read_csv(path)
    df[embed_type] = df[embed_type].apply(convert_to_array)
    return df
