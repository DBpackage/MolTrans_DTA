import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs
from math import sqrt
from scipy import stats
import torch
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, roc_curve, precision_score, recall_score

vocab_path = './ESPF/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 205
max_p = 545
 

def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def drug2emb_encoder(x):
    max_d = 50
    #max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, pka, df_dti):
        'Initialization'
        self.labels = labels
        self.pka    = pka  
        self.list_IDs = list_IDs
        self.df = df_dti
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        #d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']
        
        d_v, input_mask_d = drug2emb_encoder(d)
        p_v, input_mask_p = protein2emb_encoder(p)
        
        y = self.labels[index]
        pka = self.pka[index] 
        return d_v, p_v, input_mask_d, input_mask_p, y, pka ####


def create_fold(data, frac=(0.7, 0.1, 0.2), random_state=None, size=None):
    """
    Splits the data into train, validation, and test sets based on the given fractions and size.

    Parameters:
        data (pandas DataFrame): the input data to split.
        frac (tuple of floats): the fractions for the train, validation, and test sets, respectively.
        random_state (int or numpy.random.RandomState): the random state to use for the split.
        size (float): the size to reduce the original data by, where 1.0 represents the original size.

    Returns:
        tuple of pandas DataFrames: the train, validation, and test sets.
    """
    assert sum(frac) == 1, "Fractions must sum up to 1."
    assert size is None or size > 0, "Size must be None or greater than 0."

    # Reduce the data size if size argument is provided
    if size is not None:
        data = data.sample(frac=size, replace=False, random_state=random_state)

    train_frac, valid_frac, test_frac = frac
    shuffled_df = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # split the data into test and train+valid sets
    test = shuffled_df.sample(frac=test_frac, replace=False, random_state=random_state)
    train_valid = shuffled_df[~shuffled_df.index.isin(test.index)]

    # split the train+valid set into train and valid sets
    train = train_valid.sample(frac=(train_frac/(train_frac+valid_frac)), replace=False, random_state=random_state)
    valid = train_valid[~train_valid.index.isin(train.index)]

    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def ci(y, f):
    #start_time = time.time()
    #print("NEW CI START!")
    y = np.asarray(y)
    f = np.asarray(f)
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    n = len(y)
    c, d = 0, 0
    z = 0.0
    S = 0.0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[i] != y[j]:
                z += 1
                if f[i] < f[j]:
                    S += 1
                elif f[i] == f[j]:
                    S += 0.5
    if z > 0:
        ci = S / z
    else:
        ci = 0.0
    return ci
