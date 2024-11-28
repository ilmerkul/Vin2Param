import torch
import torch.nn as nn


class Vin2ParamLoss(nn.Module):
    def __init__(self):
        super(Vin2ParamLoss, self).__init__()

        self.carbrand_loss = nn.CrossEntropyLoss()
        self.carmodel_loss = nn.CrossEntropyLoss()
        self.color_loss = nn.CrossEntropyLoss()

    def forward(self, carbrand_logit, carmodel_logit, color_logit,
                carbrand_label, carmodel_label, color_label):
        loss = self.carbrand_loss(carbrand_logit, carbrand_label) + \
               self.carmodel_loss(carmodel_logit, carmodel_label) + \
               self.color_loss(color_logit, color_label)
        return loss


class Vin2ParamGRU(nn.Module):
    def __init__(self, inp_size, emb_size, pad_id, hid_size,
                 dropout, carbrand_n, carmodel_n, color_n):
        super(Vin2ParamGRU, self).__init__()

        self.emb = nn.Embedding(inp_size, emb_size, padding_idx=pad_id)

        self.gru = nn.LSTM(emb_size, hid_size,
                           batch_first=True, dropout=dropout,
                           bidirectional=True, num_layers=2)

        self.temp_lin = nn.Linear(2 * hid_size, hid_size)
        self.act = nn.ReLU()

        self.carbrand_lin = nn.Linear(hid_size, carbrand_n)
        self.carmodel_lin = nn.Linear(hid_size, carmodel_n)
        self.color_lin = nn.Linear(hid_size, color_n)

    def forward(self, x):
        # x (batch_size, seq_len)
        # output (batch_size, seq_len, emb_size)
        emb_output = self.emb(x)

        # gru_output (batch_size, seq_len, 2*hid_size)
        # hidden (2, batch_size, 2*hid_size)
        gru_output, hidden = self.gru(emb_output)

        hidden = torch.max(gru_output, dim=1).values
        hidden = self.temp_lin(hidden)
        hidden = self.act(hidden)

        # output (batch_size, hid_size)
        output = hidden.squeeze()

        carbrand_output = self.carbrand_lin(output)
        carmodel_output = self.carmodel_lin(output)
        color_output = self.color_lin(output)

        return carbrand_output, carmodel_output, color_output


class Vin2ParamConv(nn.Module):
    def __init__(self, inp_size, emb_size, pad_id,
                 conv_size0, conv_size1, conv_size2,
                 carbrand_n, carmodel_n, color_n):
        super(Vin2ParamConv, self).__init__()

        self.emb = nn.Embedding(inp_size, emb_size, padding_idx=pad_id)

        self.conv0 = nn.Conv1d(emb_size, conv_size0, kernel_size=(5,))
        self.conv1 = nn.Conv1d(conv_size0, conv_size1, kernel_size=(3,))
        self.conv2 = nn.Conv1d(conv_size1, conv_size2, kernel_size=(3,))

        self.act = nn.ReLU()

        self.carbrand_lin = nn.Linear(conv_size2, carbrand_n)
        self.carmodel_lin = nn.Linear(conv_size2, carmodel_n)
        self.color_lin = nn.Linear(conv_size2, color_n)

    def forward(self, x):
        emb_out = self.emb(x)

        output = emb_out.permute(0, 2, 1)

        output = self.conv0(output)
        output = self.act(output)
        output = self.conv1(output)
        output = self.act(output)
        output = self.conv2(output)

        output = torch.max(output, dim=-1).values

        carbrand_output = self.carbrand_lin(output)
        carmodel_output = self.carmodel_lin(output)
        color_output = self.color_lin(output)

        return carbrand_output, carmodel_output, color_output
