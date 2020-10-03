import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNEncoder(nn.Module):
    def __init__(self, max_length, char_embedding_dim=64, hidden_size=64):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.conv = nn.Conv1d(char_embedding_dim, self.hidden_size, 3, padding=1)

    def forward(self, inputs, mask):
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
    
        return x.transpose(1, 2)

class LSTMEncoder(nn.Module):
    def __init__(self, max_length, char_embedding_dim=64, hidden_size=64):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.lstm_enhance = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size//2, bidirectional=True, batch_first=True)
        #self.apply(self.weights_init)

    def forward(self, inputs, mask):
        return self.lstm_encoder(inputs, mask, self.lstm_enhance)

    def init_lstm(self, input_lstm):
        """
        Initialize lstm
        """
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)

        if input_lstm.bias:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('LSTM') != -1:
            self.init_lstm(m)

    def lstm_encoder(self, input, mask, lstm):

        sequence_lengths = mask.long().sum(1)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = self.sort_batch_by_length(input, sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths,
                                                     batch_first=True)
        lstmout, _ = lstm(packed_sequence_input)
        unpacked_sequence_tensor, _ = pad_packed_sequence(lstmout, batch_first=True)
        unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)

        return unpacked_sequence_tensor

    def sort_batch_by_length(self, tensor: torch.Tensor, sequence_lengths: torch.Tensor):
   
        sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
        sorted_tensor = tensor.index_select(0, permutation_index)
        index_range = Variable(torch.arange(0, len(sequence_lengths)).long()).cuda()

        _, reverse_mapping = permutation_index.sort(0, descending=False)
        restoration_indices = index_range.index_select(0, reverse_mapping)
        return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index
