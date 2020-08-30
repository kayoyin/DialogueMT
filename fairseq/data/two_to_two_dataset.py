import json
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from .import FairseqDataset
from .fairseq_dataset import TAG_DICT
from .indexed_dataset import IndexedRawTextDataset
from .collaters import Seq2SeqCollater


class TwoToTwoDataset(IndexedRawTextDataset):
    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.src_tokens = []
        self.src_sizes = []
        self.tgt_tokens = []
        self.tgt_sizes = []
        self.src = []
        self.tgt = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.speakers = []
        self.ids = []
        self.dictionary = dictionary
        self.read_data(path)
        self.size = len(self.ids)
        
        


    def read_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            chat_dict = json.load(f)

        for chat in chat_dict.values():
            for turn in chat:
                src = turn['source']
                self.src.append(src)
                src_tokens = torch.Tensor([self.dictionary.bos()] + self.dictionary.encode(src) + [self.dictionary.eos()]).long() 
                self.src_tokens.append(src_tokens)
                self.src_sizes.append(len(src_tokens))

                tgt = turn['target']
                self.tgt.append(tgt)
                tgt_tokens = torch.Tensor([self.dictionary.bos()] + self.dictionary.encode(tgt) + [self.dictionary.eos()]).long()
                self.tgt_tokens.append(tgt_tokens)
                self.tgt_sizes.append(len(tgt_tokens))

                self.speakers.append(turn['speaker'])
                self.ids.append(turn['utteranceID'])
        self.src_sizes = np.array(self.src_sizes)
        self.tgt_sizes = np.array(self.tgt_sizes)
        self.sizes = self.src_sizes

    def __getitem__(self, idx):
        if self.ids[idx] > 0:
            cxt_speaker = torch.Tensor([self.dictionary.model.piece_to_id(TAG_DICT[self.speakers[idx - 1]])])
            src_speaker = torch.Tensor([self.dictionary.model.piece_to_id(TAG_DICT[self.speakers[idx]])])
            source = torch.cat((cxt_speaker.long(), self.src_tokens[idx - 1].long(), torch.Tensor([self.dictionary.brk()]).long(), src_speaker.long(), self.src_tokens[idx].long()))
            target = torch.cat((self.tgt_tokens[idx-1].long(), torch.Tensor([self.dictionary.brk()]).long(), self.tgt_tokens[idx].long()))
        else:
            src_speaker = torch.Tensor([self.dictionary.model.piece_to_id(TAG_DICT[self.speakers[idx]])])
            source, target =  torch.cat((src_speaker.long(), self.src_tokens[idx].long())) , self.tgt_tokens[idx]
        return {"id": idx, "source": source, "target": target}

    def collater(self, samples):
        collate_fn = Seq2SeqCollater(pad_index=self.dictionary.model.piece_to_id("<pad>"), eos_index=self.dictionary.model.piece_to_id("</s>"))
        return collate_fn.collate(samples)

