import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Embedding(nn.Module):

    def __init__(self, char2id, max_length, char_embedding_dim=64):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.char_embedding_dim = char_embedding_dim
        
        self.char_embedding = nn.Embedding(len(char2id), self.char_embedding_dim, padding_idx=char2id['[PAD]'])

    def forward(self, inputs):        
        
        return self.char_embedding(inputs)


