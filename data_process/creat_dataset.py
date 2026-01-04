import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call

import torch
import torch.nn as nn



def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']


# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK


def get_length(x):
    return x.shape[1] - (np.sum(x, axis=-1) == 0).sum(1)


class MOSI:
    def __init__(self, config):

        DATA_PATH = str(config.data_path)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        self.train = load_pickle(DATA_PATH + '/train.pkl')
        self.dev = load_pickle(DATA_PATH + '/dev.pkl')
        self.test = load_pickle(DATA_PATH + '/test.pkl')
        self.pretrained_emb, self.word2id = None, None

    def get_data(self, mode):
        if mode == "train":
            return self.train, self.test, None
        elif mode == "valid":
            #return self.dev, self.word2id, None
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class MOSEI:
    def __init__(self, config):

        DATA_PATH = str(config.data_path)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        self.train = load_pickle(DATA_PATH + '/train.pkl')
        self.dev = load_pickle(DATA_PATH + '/dev.pkl')
        self.test = load_pickle(DATA_PATH + '/test.pkl')
        self.pretrained_emb, self.word2id = None, None

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()



