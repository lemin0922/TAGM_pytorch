import torch
import torch.nn as nn
from torch.autograd import Variable

from model.basic_module import LSTM_my, RNNSoftPlus, GRU_my

class WeightNet(nn.Module):
    def __init__(self, attention_size, attention_sig_w):
        super(WeightNet, self).__init__()
        self.attention_size = attention_size
        self.attention_sig_w = attention_sig_w

        self.attention_weight = nn.Linear(attention_size, 1)
        self.non_learnable_factor = nn.Parameter(torch.ones(1) * self.attention_sig_w, requires_grad=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.attention_weight(x)
        attention = self.sigmoid(logits * self.non_learnable_factor.expand_as(logits))
        return attention.squeeze(2)


class AttentionNet(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            mode,
            batch_first=True,
            dropout=0.0,
            attention_sig_w=3,
            bidirectional=True
    ):
        super(AttentionNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.mode = mode
        self.batch_first = batch_first
        self.dropout = dropout
        self.attention_sig_w = attention_sig_w
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.attention_size = 2 * hidden_size if bidirectional else hidden_size

        self.dropout0 = nn.Dropout(dropout)

        if mode == 'lstm':
            self.cell = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout,
                                bidirectional=bidirectional)
        elif mode == 'gru':
            self.cell = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout,
                               bidirectional=bidirectional)
        elif mode == 'gru_my':
            self.cell = GRU_my(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout,
                                    bidirectional=bidirectional)
        elif mode == 'rnn_softplus':
            self.cell = RNNSoftPlus(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout,
                               bidirectional=bidirectional)
        elif mode == 'rnn':
            self.cell = nn.RNN(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout,
                               bidirectional=bidirectional)
        self.weight_net = WeightNet(self.attention_size, attention_sig_w)

    # def get_state(self):
    #     h0_encoder = Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size),
    #                           requires_grad=False).cuda()
    #     c0_encoder = Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size),
    #                           requires_grad=False).cuda()
    #     return h0_encoder, c0_encoder

    def forward(self, x):
        self.h0 = Variable(torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size),
                           requires_grad=False).cuda()
        self.c0 = Variable(torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size),
                           requires_grad=False).cuda()

        if self.mode == 'lstm':
            x, _ = self.cell(x, (self.h0, self.c0))
            x = self.dropout0(x)
            attention = self.weight_net(x)
        elif self.mode == 'rnn_softplus' or self.mode == 'gru_my':
            x = self.cell(x, self.h0)
            x = self.dropout0(x)
            attention = self.weight_net(x)
        else:
            x, h_t = self.cell(x, self.h0)
            x = self.dropout0(x)
            attention = self.weight_net(x)

        return attention
