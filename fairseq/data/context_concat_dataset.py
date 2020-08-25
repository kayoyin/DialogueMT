import json
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader

from .import FairseqDataset
from .indexed_dataset import IndexedRawTextDataset

tag_dict = {"customer": "<ot> ", "agent": "<en> "}

class TwoToOneDataset(IndexedRawTextDataset):
    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.src_tokens = []
        self.src_sizes = []
        self.tgt_tokens = []
        self.tgt_sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.speakers = []
        self.ids = []
        self.read_data(path, dictionary)
        self.size = len(self.ids)


    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            chat_dict = json.load(f)

        for chat in chat_dict.values():
            for turn in chat:
                src = turn['source']
                src_tokens = dictionary.encode_line(
                    src, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.src_tokens.append(src_tokens)
                self.src_sizes.append(len(src_tokens))

                tgt = turn['target']
                tgt_tokens = dictionary.encode_line(
                    tgt, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tgt_tokens.append(tgt_tokens)
                self.tgt_sizes.append(len(tgt_tokens))

                self.speakers.append(turn['speaker'])
                self.ids.append(turn['utteranceID'])
        self.src_sizes = np.array(self.src_sizes)
        self.tgt_sizes = np.array(self.tgt_sizes)

    def __getitem__(self, idx):
        if self.ids[idx] > 0:
            cxt = tag_dict[self.speakers[idx - 1]] + self.src[idx - 1] + " <brk> "
            source, target =  cxt + tag_dict[self.speakers[idx]] + self.src[idx], self.tgt[idx]
        else:
            source, target =  tag_dict[self.speakers[idx]] + self.src[idx], self.tgt[idx]
        return source, target
