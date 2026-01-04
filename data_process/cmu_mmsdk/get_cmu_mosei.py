from collections import defaultdict
import os, re

import numpy
import numpy as np
import torch
from mmsdk import mmdatasdk as md, mmdatasdk
import os

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
from transformers import BertTokenizer
from torch.utils.data import Dataset


'''
This code is based on https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
'''

index = 0
current_dir = os.path.dirname(os.path.abspath(__file__))
bert_path = os.path.join(current_dir, 'pretrain_data', 'bert_en')
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
DATASET = md.cmu_mosei
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']
# def return_unk():
# return UNK
# word2id.default_factory = return_unk
def splittraindevtest(train_split, dev_split, test_split, dataset, features):
    global index
    EPS = 1e-6
    text_field, visual_field, acoustic_field, label_field = features
    # construct a word2id mapping that automatically takes increment when new words are encountered
    # place holders for the final train/dev/test dataset
    train = []
    dev = []
    test = []
    # define a regular expression to extract the video ID out of the keys
    pattern = re.compile('(.*)\[.*\]')
    num_drop = 0 # a counter to count how many data points went into some processing issues
    for segment in dataset[label_field].keys():
        # get the video ID and the features out of the aligned dataset
        vid = re.search(pattern, segment).group(1)
        label = dataset[label_field][segment]['features']
        _words = dataset[text_field][segment]['features']
        _visual = dataset[visual_field][segment]['features']
        _acoustic = dataset[acoustic_field][segment]['features']
        id = index
        index = index + 1
        # if the sequences are not same length after alignment, there must be some problem with some modalities
        # we should drop it or inspect the data again
        if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
            print(
            f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
            num_drop += 1
            continue

        # remove nan values
        label = np.nan_to_num(label)
        _visual = np.nan_to_num(_visual)
        _acoustic = np.nan_to_num(_acoustic)

        # remove speech pause tokens - this is in general helpful
        # we should remove speech pauses and corresponding visual/acoustic features together
        # otherwise modalities would no longer be aligned
        actual_words = []
        words = []
        visual = []
        acoustic = []
        for i, word in enumerate(_words):
            if word[0] != b'sp':
                actual_words.append(word[0].decode('utf-8'))
                words.append(word2id[word[0].decode('utf-8')])  # SDK stores strings as bytes, decode into strings here
                visual.append(_visual[i, :])
                acoustic.append(_acoustic[i, :])

        words = np.asarray(words)
        visual = np.asarray(visual)
        acoustic = np.asarray(acoustic)

        # z-normalization per instance and remove nan/infs
        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
        acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))


        if vid in train_split:
            train.append(((words, visual, acoustic, actual_words), label, segment, id))
        elif vid in dev_split:
            dev.append(((words, visual, acoustic, actual_words), label, segment, id))
        elif vid in test_split:
            test.append(((words, visual, acoustic, actual_words), label, segment, id))
        else:
            print(f"Found video that doesn't belong to any splits: {vid}")

    def return_unk():
        return UNK

    word2id.default_factory = return_unk
    print(f"Total number of {num_drop} datapoints have been dropped.")
    return train, dev, test


def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)

    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    # print('visual:', visual.shape)
    # print('acoustic:', acoustic.shape)
    # print('sentences:',sentences.shape)
    SENT_LEN = sentences.size(0)
    #print(type(sample[1] for sample in batch))
    # index = torch.LongTensor([sample[2] for sample in batch])
    bert_details = []
    for sample in batch:
        text = " ".join(sample[0][3])  # len(text)= 179 len(sample[0][3])=39 ['and', 'um', 'i', 'have', 'to', 'admit', 'i', 'was', 'watching', 'this', 'i', 'put', 'it', 'on', ...]
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, max_length=SENT_LEN + 2, add_special_tokens=True, pad_to_max_length=True)
        bert_details.append(encoded_bert_sent)

    # Bert things are batch_first
    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])  # bert_sentences.shape=([64, 41]) 堆叠(64): len(41): [101, 1045, 16755, 2009, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
    bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])  # bert_sentence_types.shape=([64, 41]) 堆叠(64): len(41): [0, 0 ...]
    bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])  # bert_sentence_att_mask.shape=([64, 41]) 堆叠(64): len(41): [1, 0 ...]
    # except:
    #     print('cmu_mosi batch in except')
    #     sum_sentences = torch.sum(sentences,dim=0).float()  # ([64, 32, 300]) -> [32, 300] 消除第一个特征维度 64 保留对齐的300和32 bathsize 维度
    #     bert_sentences = (sum_sentences - torch.min(sum_sentences)) / (torch.max(sum_sentences) - torch.min(sum_sentences))  # 将矩阵归一化到 0-1 之间 [32, 300]
    #     # 生成全一矩阵
    #     shape = (bert_sentences.size(0), bert_sentences.size(1))
    #     bert_sentence_types = torch.ones(shape)
    #     bert_sentence_att_mask = torch.ones(shape)

    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    id = torch.tensor([sample[3] for sample in batch]).unsqueeze(1)
    return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, id


def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
    found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()

def align_cmu():
    # choose your path
    DATA_PATH = '/home/zzj/zjm/cmu-mosi/CMUMosei'

    visual_field = 'CMU_MOSEI_VisualOpenFace2'
    acoustic_field = 'CMU_MOSEI_COVAREP'
    text_field = 'CMU_MOSEI_TimestampedWords'

    features = [
        text_field,
        visual_field,
        acoustic_field
    ]

    recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
    dataset = md.mmdataset(recipe)

    def avg(intervals: np.array, features: np.array) -> np.array:
        try:
            return np.average(features, axis=0)
        except:
            return features

    # first we align to words with averaging, collapse_function receives a list of functions
    dataset.align(text_field, collapse_functions=[avg])

    label_field = 'CMU_MOSEI_Labels'
    label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
    dataset.add_computational_sequences(label_recipe, destination=None)
    dataset.align(label_field, replace=True)
    print('done!')

    ### save
    deploy_files = {x: x for x in dataset.computational_sequences.keys()}
    dataset.deploy("./deployed", deploy_files)
    print('file save.')
    #aligned_cmumosi_highlevel = md.mmdataset('./deployed')


def get_mosei_dataloader(batch_size):
    visual_field1 = 'CMU_MOSEI_VisualOpenFace2'
    acoustic_field1 = 'CMU_MOSEI_COVAREP'
    text_field1 = 'CMU_MOSEI_TimestampedWords'
    label_field1 = 'CMU_MOSEI_Labels'
    features1 = [
        text_field1,
        visual_field1,
        acoustic_field1,
        label_field1
    ]
    DATA_PATH1 = './deployed'
    if os.path.exists(os.path.join(DATA_PATH1, label_field1) + '.csd'):
        print('load:')
    else:
        print('align:')
        align_cmu()

    recipe1 = {feat: os.path.join(DATA_PATH1, feat) + '.csd' for feat in features1}
    dataset1 = mmdatasdk.mmdataset(recipe1)
    tensors = dataset1.get_tensors(seq_len=25, non_sequences=["Opinion Segment Labels"], direction=False,
                                   folds=[mmdatasdk.cmu_mosi.standard_folds.standard_train_fold, mmdatasdk.cmu_mosi.
                                   standard_folds.standard_valid_fold,mmdatasdk.cmu_mosi.standard_folds.standard_test_fold])

    fold_names = ["train", "valid", "test"]
    train, dev, test = splittraindevtest(mmdatasdk.cmu_mosi.standard_folds.standard_train_fold, mmdatasdk.cmu_mosi.
                                         standard_folds.standard_valid_fold,
                                         mmdatasdk.cmu_mosi.standard_folds.standard_test_fold,
                                         dataset1, features1)

    # (text,audio,visio), label, segment = train[10]
    # print('text', text.shape)
    # print('audio', audio.shape)
    # print('visio', visio.shape)
    # print('label',label.shape)

    print('lenT:',len(train))
    print('lenE:', len(dev))
    print('lenS:', len(test))
    # train_data_set = mosi_Dataset(train)
    # dev_data_set = mosi_Dataset(dev)
    # text_data_set = mosi_Dataset(test)

    batch_sz = batch_size
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
    dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_sz, collate_fn=multi_collate)
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz, collate_fn=multi_collate)

    return train_loader,dev_loader,test_loader
