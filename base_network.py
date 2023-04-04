# Code for making a base nework that would take tokenized input and pass it through an embedding layer and then through a LSTM layer to get the output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


class base_network(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, bidirectional, device):
        # super(base_network, self).__init__()
        # self.embedding = nn.Embedding(input_size, embedding_size)
        super(base_network, self).__init__()
        self.embedding = bert_model

        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        # self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding(x)
        x = x.last_hidden_state
        x, (h_n, c_n) = self.lstm(x)
        out = torch.permute(h_n[-2:, :, :], (1, 0, 2)).reshape(x.size(0), -1)
        return out
