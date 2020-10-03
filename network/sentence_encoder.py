import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from . import network

class SentenceEncoder(nn.Module):

    def __init__(self, char2id, max_length, char_embedding_dim=64, hidden_size=64, encoder='cnn'):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(char2id, max_length, char_embedding_dim)
        if encoder=='cnn':
            self.encoder1 = network.encoder.CNNEncoder(max_length, char_embedding_dim, hidden_size)
            self.encoder2 = network.encoder.CNNEncoder(max_length, char_embedding_dim, hidden_size)
        elif encoder=='lstm':
            self.encoder1 = network.encoder.LSTMEncoder(max_length, char_embedding_dim, hidden_size)
            self.encoder2 = network.encoder.LSTMEncoder(max_length, char_embedding_dim, hidden_size)
        self.char2id = char2id

    def forward(self, input1, input2, mask1, mask2):
        x1 = self.embedding(input1)
        x2 = self.embedding(input2)
        x1 = self.encoder1(x1, mask1)
        x2 = self.encoder2(x2, mask2)
        return x1, x2

    def tokenize(self, abbs, target):
        
        abbs_char = []
        for abb in abbs:
            abbs_char += list(abb.lower())
            abbs_char.append('[UNK]')
        abbs_char = abbs_char[:-1]

        indexed_abbs = []
        for char in abbs_char:
            if char in self.char2id:
                indexed_abbs.append(self.char2id[char])
            else:
                indexed_abbs.append(self.char2id['[UNK]'])
        
        # mask
        mask_abb = np.zeros((self.max_length), dtype=np.int32)
        mask_abb[:len(indexed_abbs)] = 1
        
        # padding
        while len(indexed_abbs) < self.max_length:
            indexed_abbs.append(self.char2id['[PAD]'])
        indexed_abbs = indexed_abbs[:self.max_length]


        target_chars = list(target.lower())

        indexed_target = []
        for char in target_chars:
            if char in self.char2id:
                indexed_target.append(self.char2id[char])
            else:
                indexed_target.append(self.char2id['[UNK]'])

        mask_target = np.zeros((self.max_length), dtype=np.int32)
        mask_target[:len(indexed_target)] = 1

        while len(indexed_target) < self.max_length:
            indexed_target.append(self.char2id['[PAD]'])
        indexed_target = indexed_target[:self.max_length]
        

        return indexed_abbs, mask_abb, indexed_target, mask_target

    def tokenize_test(self, abbs, target):

        abbs_char = list(abbs.lower())
        indexed_abbs = []
        for char in abbs_char:
            if char in self.char2id:
                indexed_abbs.append(self.char2id[char])
            else:
                indexed_abbs.append(self.char2id['[UNK]'])
        
        # mask
        mask_abb = np.zeros((self.max_length), dtype=np.int32)
        mask_abb[:len(indexed_abbs)] = 1
        
        # padding
        while len(indexed_abbs) < self.max_length:
            indexed_abbs.append(self.char2id['[PAD]'])
        indexed_abbs = indexed_abbs[:self.max_length]


        target_chars = list(target.lower())

        indexed_target = []
        for char in target_chars:
            if char in self.char2id:
                indexed_target.append(self.char2id[char])
            else:
                indexed_target.append(self.char2id['[UNK]'])

        mask_target = np.zeros((self.max_length), dtype=np.int32)
        mask_target[:len(indexed_target)] = 1

        while len(indexed_target) < self.max_length:
            indexed_target.append(self.char2id['[PAD]'])
        indexed_target = indexed_target[:self.max_length]

        return indexed_abbs, mask_abb, indexed_target, mask_target