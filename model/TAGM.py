import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from model.attention_module import AttentionNet
from six.moves import xrange

class TAGM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            fc_hidden_size,
            num_layers,
            num_classes,
            batch_size,
            mode,
            batch_first=True,
            att_dropout=0.0,
            att_bidirectional=True,
            att_sig_w=3,
            tagm_dropout=0.0,
            tagm_bidirectional=False,
            more_fc=False
    ):
        super(TAGM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.mode = mode
        self.batch_first = batch_first
        self.att_dropout = att_dropout
        self.att_bidirectional = att_bidirectional
        self.att_sig_w = att_sig_w
        self.tagm_dropout = tagm_dropout
        self.tagm_bidirectional = tagm_bidirectional
        self.tagm_num_direction = 2 if tagm_bidirectional else 1
        self.more_fc = more_fc

        self.one = Variable(torch.ones(1), requires_grad=False).cuda()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU(True)
        self.attention_net = AttentionNet(
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            mode,
            batch_first=batch_first,
            dropout=att_dropout,
            attention_sig_w=att_sig_w,
            bidirectional=att_bidirectional
        )
        self.dropout = nn.Dropout(tagm_dropout)

        if more_fc:
            self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
            self.fc0 = nn.Linear(fc_hidden_size, num_classes)
            self.dropout1 = nn.Dropout(tagm_dropout)
        else:
            self.fc0 = nn.Linear(hidden_size, num_classes)

        self.init_weigths()

    def init_weigths(self):
        # TODO : initialize parameters
        pass

    def forward(self, x):
        def recurrence(input_, hidden, att):
            prev_h = hidden

            in_gate = att
            forget_gate = self.one.expand_as(in_gate) - in_gate
            i2h = self.i2h(input_)
            h2h = self.h2h(prev_h)
            in_transform = self.relu(i2h + h2h)

            next_h = in_gate.unsqueeze(1).expand_as(in_transform) * in_transform \
                     + forget_gate.unsqueeze(1).expand_as(prev_h) * prev_h

            return next_h

        # apply TAGM
        hidden = Variable(torch.zeros(self.num_layers * self.tagm_num_direction, x.size(0), self.hidden_size),
                          requires_grad=False).cuda()
        attention = self.attention_net(x)

        if self.batch_first:
            x = x.transpose(0, 1)

        x_cell = []
        for i in xrange(x.size(0)):
            hidden = recurrence(x[i], hidden, attention[:, i])
            x_cell.append(hidden)

        # for i in xrange(x.size(0)):
        #     if self.mode == 'lstm':
        #         hidden = recurrence(x[i], hidden, attention[:, i])
        #         x_cell.append(hidden[0])
        #     else:
        #         hidden = recurrence(x[i], hidden, attention[:, i])
        #         x_cell.append(hidden)

        x_cell = torch.cat(x_cell, 0)
        if self.batch_first:
            x_cell = x_cell.transpose(0, 1)

        # Top classifier
        if self.more_fc:
            x_cell = self.dropout(x_cell)
            x_ = self.fc1(x_cell[:, -1, :])
            x_ = self.dropout1(x_)
            out = self.fc0(x_)
        else:
            x_cell = self.dropout(x_cell)
            out = self.fc0(x_cell[:, -1, :])

        return out, attention
        # return F.log_softmax(out, 1), attention


class TAGM_Bi(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            batch_size,
            mode,
            batch_first=True,
            att_dropout=0.0,
            att_bidirectional=True,
            att_sig_w=3,
            tagm_dropout=0.0,
            tagm_bidirectional=False,
    ):
        super(TAGM_Bi, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.mode = mode
        self.batch_first = batch_first
        self.att_dropout = att_dropout
        self.att_bidirectional = att_bidirectional
        self.att_sig_w = att_sig_w
        self.tagm_dropout = tagm_dropout
        self.tagm_bidirectional = tagm_bidirectional
        self.tagm_num_direction = 2 if tagm_bidirectional else 1

        self.one = Variable(torch.ones(1), requires_grad=False).cuda()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU(True)
        self.attention_net = AttentionNet(
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            mode,
            batch_first=batch_first,
            dropout=att_dropout,
            attention_sig_w=att_sig_w,
            bidirectional=att_bidirectional
        )
        self.dropout = nn.Dropout(tagm_dropout)
        self.fc0 = nn.Linear(hidden_size, num_classes)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(tagm_dropout)

        self.init_weigths()

    def init_weigths(self):
        # TODO : initialize parameters
        pass

    def forward(self, x):
        def recurrence(input_, hidden, att):
            prev_h = hidden

            in_gate = att
            forget_gate = self.one.expand_as(in_gate) - in_gate
            i2h = self.i2h(input_)
            h2h = self.h2h(prev_h)
            in_transform = self.relu(i2h + h2h)

            next_h = in_gate.unsqueeze(1).expand_as(in_transform) * in_transform \
                     + forget_gate.unsqueeze(1).expand_as(prev_h) * prev_h

            return next_h

        # apply TAGM
        hidden = Variable(torch.zeros(self.num_layers * self.tagm_num_direction, x.size(0), self.hidden_size),
                          requires_grad=False).cuda()
        attention = self.attention_net(x)

        if self.batch_first:
            x = x.transpose(0, 1)

        x_cell = []
        for i in xrange(x.size(0)):
            hidden = recurrence(x[i], hidden, attention[:, i])
            x_cell.append(hidden)

        x_cell = torch.cat(x_cell, 0)
        if self.batch_first:
            x_cell = x_cell.transpose(0, 1)

        # Top classifier
        x_cell = self.dropout(x_cell)
        out = self.fc0(x_cell[:, -1, :])
        out = self.dropout1(out)
        out = self.fc1(out)

        return out, attention
        # return F.log_softmax(out, 1), attention

class TAGM_GRU(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            batch_size,
            mode,
            batch_first=True,
            att_dropout=0.0,
            att_bidirectional=True,
            att_sig_w=3,
            tagm_dropout=0.0,
            tagm_bidirectional=False,
    ):
        super(TAGM_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.mode = mode
        self.batch_first = batch_first
        self.att_dropout = att_dropout
        self.att_bidirectional = att_bidirectional
        self.att_sig_w = att_sig_w
        self.tagm_dropout = tagm_dropout
        self.tagm_bidirectional = tagm_bidirectional
        self.tagm_num_direction = 2 if tagm_bidirectional else 1

        # parameter
        self.one = Variable(torch.ones(1), requires_grad=False).cuda()
        self.attention_net = AttentionNet(
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            mode,
            batch_first=batch_first,
            dropout=att_dropout,
            attention_sig_w=att_sig_w,
            bidirectional=att_bidirectional
        )

        self.i2h_zr = nn.Linear(input_size, 2 * hidden_size)
        self.h2h_zr = nn.Linear(hidden_size, 2 * hidden_size)
        self.i2h_tilda = nn.Linear(input_size, hidden_size)
        self.h2h_tilda = nn.Linear(hidden_size, hidden_size)
        self.one_ = Variable(torch.ones(1), requires_grad=False).cuda()

        # top classifier parameter
        self.dropout = nn.Dropout(tagm_dropout)
        self.fc0 = nn.Linear(hidden_size, num_classes)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)

        self.init_weigths()

    def init_weigths(self):
        # TODO : initialize parameters
        pass

    def forward(self, x):
        def recurrence(input_, hidden, att):
            prev_h = hidden

            in_gate = att
            forget_gate = self.one.expand_as(in_gate) - in_gate

            i2h_tilda = self.i2h_tilda(input_)
            h2h_tilda = self.h2h_tilda(hidden)
            zr_gate = self.i2h_zr(input_) + self.h2h_zr(hidden)
            r_gate, z_gate = zr_gate.chunk(2, 1)

            r_gate = F.sigmoid(r_gate)
            z_gate = F.sigmoid(z_gate)
            h_tilda = F.tanh(i2h_tilda + r_gate * h2h_tilda)

            next_h = z_gate * h_tilda + (self.one_.expand_as(prev_h) - z_gate) * prev_h
            next_h = att.unsqueeze(1).expand_as(next_h) * next_h# 48.05

            return next_h

        # def recurrence(input_, hidden, att): # 48.03
        #     prev_h = hidden
        #
        #     in_gate = att
        #     forget_gate = self.one.expand_as(in_gate) - in_gate
        #
        #     i2h_tilda = self.i2h_tilda(input_)
        #     h2h_tilda = self.h2h_tilda(hidden)
        #     zr_gate = self.i2h_zr(input_) + self.h2h_zr(hidden)
        #     r_gate, z_gate = zr_gate.chunk(2, 1)
        #
        #     r_gate = F.sigmoid(r_gate)
        #     z_gate = F.sigmoid(z_gate)
        #     h_tilda = F.tanh(i2h_tilda + r_gate * h2h_tilda)
        #
        #     next_h = z_gate * h_tilda + (self.one_.expand_as(prev_h) - z_gate) * prev_h
        #     next_h = att.unsqueeze(1).expand_as(next_h) * next_h
        #
        #     return next_h

        # apply TAGM
        # TODO : bidirectional
        # hidden = Variable(torch.zeros(self.num_layers * self.tagm_num_direction, x.size(0), self.hidden_size),
        #                   requires_grad=False).cuda() # 48

        hidden = Variable(torch.zeros(x.size(0), self.hidden_size), requires_grad=False).cuda()
        attention = self.attention_net(x)

        if self.batch_first:
            x = x.transpose(0, 1)

        x_cell = []
        for i in xrange(x.size(0)):
            hidden = recurrence(x[i], hidden, attention[:, i])
            x_cell.append(hidden.unsqueeze(0))

        # for i in xrange(x.size(0)):
        #     if self.mode == 'lstm':
        #         hidden = recurrence(x[i], hidden, attention[:, i])
        #         x_cell.append(hidden[0])
        #     else:
        #         hidden = recurrence(x[i], hidden, attention[:, i])
        #         x_cell.append(hidden)

        x_cell = torch.cat(x_cell, 0)
        if self.batch_first:
            x_cell = x_cell.transpose(0, 1)

        # Top classifier
        x_cell = self.dropout(x_cell)
        out = self.fc0(x_cell[:, -1, :])

        return out, attention
        # return F.log_softmax(out, 1), attention

class TAGM_LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            batch_size,
            mode,
            batch_first=True,
            att_dropout=0.0,
            att_bidirectional=True,
            att_sig_w=3,
            tagm_dropout=0.0,
            tagm_bidirectional=False,
    ):
        super(TAGM_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.mode = mode
        self.batch_first = batch_first
        self.att_dropout = att_dropout
        self.att_bidirectional = att_bidirectional
        self.att_sig_w = att_sig_w
        self.tagm_dropout = tagm_dropout
        self.tagm_bidirectional = tagm_bidirectional
        self.tagm_num_direction = 2 if tagm_bidirectional else 1

        # parameter
        self.one = Variable(torch.ones(1), requires_grad=False).cuda()
        self.attention_net = AttentionNet(
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            mode,
            batch_first=batch_first,
            dropout=att_dropout,
            attention_sig_w=att_sig_w,
            bidirectional=att_bidirectional
        )

        self.i2h_zr = nn.Linear(input_size, 2 * hidden_size)
        self.h2h_zr = nn.Linear(hidden_size, 2 * hidden_size)
        self.i2h_tilda = nn.Linear(input_size, hidden_size)
        self.h2h_tilda = nn.Linear(hidden_size, hidden_size)
        self.one_ = Variable(torch.ones(1), requires_grad=False).cuda()

        # top classifier parameter
        self.dropout = nn.Dropout(tagm_dropout)
        self.fc0 = nn.Linear(hidden_size, num_classes)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)

        self.init_weigths()

    def init_weigths(self):
        # TODO : initialize parameters
        pass

    def forward(self, x):
        def recurrence(input_, hidden, att):
            prev_h = hidden

            in_gate = att
            forget_gate = self.one.expand_as(in_gate) - in_gate

            i2h_tilda = self.i2h_tilda(input_)
            h2h_tilda = self.h2h_tilda(hidden)
            zr_gate = self.i2h_zr(input_) + self.h2h_zr(hidden)
            r_gate, z_gate = zr_gate.chunk(2, 1)

            r_gate = F.sigmoid(r_gate)
            z_gate = F.sigmoid(z_gate)
            h_tilda = F.tanh(i2h_tilda + r_gate * h2h_tilda)

            next_h = z_gate * h_tilda + (self.one_.expand_as(prev_h) - z_gate) * prev_h
            next_h = att.unsqueeze(1).expand_as(next_h) * next_h # 48.05

            return next_h

        # def recurrence(input_, hidden, att): # 48.03
        #     prev_h = hidden
        #
        #     in_gate = att
        #     forget_gate = self.one.expand_as(in_gate) - in_gate
        #
        #     i2h_tilda = self.i2h_tilda(input_)
        #     h2h_tilda = self.h2h_tilda(hidden)
        #     zr_gate = self.i2h_zr(input_) + self.h2h_zr(hidden)
        #     r_gate, z_gate = zr_gate.chunk(2, 1)
        #
        #     r_gate = F.sigmoid(r_gate)
        #     z_gate = F.sigmoid(z_gate)
        #     h_tilda = F.tanh(i2h_tilda + r_gate * h2h_tilda)
        #
        #     next_h = z_gate * h_tilda + (self.one_.expand_as(prev_h) - z_gate) * prev_h
        #     next_h = att.unsqueeze(1).expand_as(next_h) * next_h
        #
        #     return next_h

        # apply TAGM
        # TODO : bidirectional
        # hidden = Variable(torch.zeros(self.num_layers * self.tagm_num_direction, x.size(0), self.hidden_size),
        #                   requires_grad=False).cuda() # 48

        hidden = Variable(torch.zeros(x.size(0), self.hidden_size), requires_grad=False).cuda()
        attention = self.attention_net(x)

        if self.batch_first:
            x = x.transpose(0, 1)

        x_cell = []
        for i in xrange(x.size(0)):
            hidden = recurrence(x[i], hidden, attention[:, i])
            x_cell.append(hidden.unsqueeze(0))

        # for i in xrange(x.size(0)):
        #     if self.mode == 'lstm':
        #         hidden = recurrence(x[i], hidden, attention[:, i])
        #         x_cell.append(hidden[0])
        #     else:
        #         hidden = recurrence(x[i], hidden, attention[:, i])
        #         x_cell.append(hidden)

        x_cell = torch.cat(x_cell, 0)
        if self.batch_first:
            x_cell = x_cell.transpose(0, 1)

        # Top classifier
        x_cell = self.dropout(x_cell)
        out = self.fc0(x_cell[:, -1, :])

        return out, attention
        # return F.log_softmax(out, 1), attention
