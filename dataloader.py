import os
import pickle
import random
import numpy as np
from tqdm import tqdm_notebook, tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *

from config import parse_opts
from data_process.creat_dataset import MOSI, MOSEI

word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

class MSADataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        if config.dataset == 'mosi':
            dataset = MOSI(config)
        elif config.dataset == 'mosei':
            dataset = MOSEI(config)
        else:
            print("Dataset not defined correctly")
            exit()

        #self.data, self.word2id, _ = dataset.get_data(mode)
        self.data, self.data_2, _ = dataset.get_data(mode)
        self.len = len(self.data)
        self.rand_prob = 0.6
        # config.pretrained_emb = self.pretrained_emb



    @property
    def tva_dim(self):
        t_dim = 768
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        # if self.data_2 is not None and random.random() < self.rand_prob and index < 686:
        #     return self.data_2[index],index
        # else:
        return self.data[index], index

    def __len__(self):
        return self.len


class IEMOCAP_Datasets(Dataset):
    def __init__(self, config, mode):
        super(IEMOCAP_Datasets, self).__init__()

        DATA_PATH = str(config.data_path)
        aligned = config.aligned
        dataset_path = os.path.join(DATA_PATH, 'iemocap_data.pkl' if aligned else 'iemocap' + '_data_noalign.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[mode]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[mode]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[mode]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[mode]['labels'].astype(np.float32)).cpu().detach()

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        Y = torch.argmax(Y, dim=-1)
        return X,Y


def get_data_iemocap(config,mode):
    aligned = config.aligned
    dataset = 'iemocap'
    alignment = 'a' if aligned else 'na'
    data_path = str(config.data_path)
    data_path_split = os.path.join(data_path, dataset) + f'_{mode}_{alignment}.dt'
    if not os.path.exists(data_path_split):
        print(f" - Creating new {mode} data")
        data = IEMOCAP_Datasets(config, mode)
        torch.save(data, data_path_split)
    else:
        print(f" - Found cached {mode} data")
        data = torch.load(data_path_split)
    return data

def get_loader(config, mode, shuffle=True):
    """Load DataLoader of given DialogDataset"""
    if config.dataset == 'iemocap':
        dataset = get_data_iemocap(config, mode)
    else:
        dataset = MSADataset(config, mode)
    print('len:',dataset.__len__())

    config.data_len = len(dataset)
    #config.tva_dim = dataset.tva_dim
    bert_path = config.bert_path_en

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: len(x[0][0][3]), reverse=True)
        v_lens = []
        a_lens = []
        labels = []
        ids = []
        bert_tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
        for sample, index in batch:
            if len(sample[0]) > 4:  # unaligned case
                v_lens.append(torch.IntTensor([sample[0][4]]))
                a_lens.append(torch.IntTensor([sample[0][5]]))
            else:  # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            labels.append(torch.from_numpy(sample[1]))
            ids.append(index)

        id = torch.tensor(ids)
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)
        labels = torch.cat(labels, dim=0)

        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:, 0][:, None]

        # Rewrite this
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        v_masks = pad_sequence([torch.zeros(torch.FloatTensor(sample[0][1]).size(0)) for sample, _ in batch],
                               target_len=vlens.max().item(), padding_value=1)
        a_masks = pad_sequence([torch.zeros(torch.FloatTensor(sample[0][2]).size(0)) for sample, _ in batch],
                               target_len=alens.max().item(), padding_value=1)

        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample, _ in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample, _ in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample, _ in batch], target_len=alens.max().item())

        ## BERT-based features input prep

        # SENT_LEN = min(sentences.size(0),50)
        SENT_LEN = 50
        # Create bert indices using tokenizer

        bert_details = []
        for sample, _ in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)

        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample, _ in batch])
        if (vlens <= 0).sum() > 0:
            vlens[np.where(vlens == 0)] = 1

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, id, v_masks, a_masks

    if config.dataset == 'iemocap':
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            drop_last=True)
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn)

    return data_loader


class MMDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        DATA_MAP = {
            'sims3l': self.__init_sims,
        }
        DATA_MAP['sims3l']()

    def __init_sims(self):
        with open('data_set/ch_simsv2' +'/unaligned.pkl', 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                print('key :',list(data.keys()))
        # Control Number of Supvised Data
        if self.config.supvised_nums != 2722:
            if self.mode == 'train':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    temp_data[self.mode][key] = data[self.mode][key][-self.config.supvised_nums:]
                data[self.mode] = temp_data[self.mode]

            if self.mode == 'valid':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    p = int(self.config.supvised_nums / 2)
                    temp_data[self.mode][key] = data[self.mode][key][-p:]
                data[self.mode] = temp_data[self.mode]

            if self.mode == 'train_mix':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    data_sup = data[self.mode][key][2722 - self.config.supvised_nums:2722]
                    data_unsup = data[self.mode][key][2723:]
                    temp_data[self.mode][key] = np.concatenate((data_sup, data_unsup), axis=0)
                data[self.mode] = temp_data[self.mode]
        # Complete Data
        # if not self.mode == 'train_mix':
        #     self.rawText = data[self.mode]['raw_text']
        if self.config.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.ids = data[self.mode]['id']
        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        # Labels
        self.labels = {
            'M': data[self.mode][self.config.train_mode + '_labels'].astype(np.float32)
        }
        if self.config.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.config.train_mode + '_labels_' + m]

        #logger.info(f"{self.mode} samples: {self.labels['M'].shape}")
        if self.mode == 'train_mix':
            self.mask = data[self.mode]['mask']
        # Clear dirty data
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0
        # Mean feture
        if self.config.need_normalized:
            self.__normalize()

    def __normalize(self):
        self.vision_temp = []
        self.audio_temp = []
        for vi in range(len(self.vision_lengths)):
            self.vision_temp.append(np.mean(self.vision[vi][:self.vision_lengths[vi]], axis=0))
        for ai in range(len(self.audio_lengths)):
            self.audio_temp.append(np.mean(self.audio[ai][:self.audio_lengths[ai]], axis=0))
        self.vision = np.array(self.vision_temp)
        self.audio = np.array(self.audio_temp)

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.config.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def __getitem__(self, index):
        sample = {
            'index': index,
            # 'raw_text': self.rawText[index] if self.mode != 'train_mix' else [],
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'audio_lengths': self.audio_lengths[index],
            'vision_lengths': self.vision_lengths[index],
            'mask': self.mask[index] if self.mode == 'train_mix' else [],
        }
        return sample

def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test'),
    }

    args.seq_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=1,
                       shuffle=True)
        for ds in datasets.keys()
    }
    return dataLoader


if __name__ == '__main__':
    pass

