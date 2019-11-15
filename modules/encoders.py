import torch
import torch.nn as nn
from modules.layers import ScaleLinear
from modules.transformers import TransformerEncoderLayer


class RNNencoder(nn.Module):
    def __init__(self, encoder_args):
        """
        RNN encoder
        """
        super(RNNencoder, self).__init__()

        for a in encoder_args.keys():
            setattr(self, a, encoder_args[a])

        assert self.rnn_type == 'LSTM' or self.rnn_type == 'GRU'
        if self.rnn_type == 'LSTM':
            rnn_cls = nn.LSTM
        else:
            rnn_cls = nn.GRU

        self.rnn = rnn_cls(
            self.in_dim, self.num_hid, self.nlayers,
            bidirectional=True,
            dropout=self.dropout,
            batch_first=True)

        self.ndirections = 1 + int(self.bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * 2, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (weight.new(*hid_shape).zero_(),weight.new(*hid_shape).zero_())
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        if self.training:
            self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        return output

    def call(self, x):
        # x: [batch, sequence, in_dim]
        output = self.forward(x)
        forward_vals = output[:, :, :self.num_hid]
        backward_vals = output[:, :, self.num_hid:]

        last_forward_vals = output[:, -1, :self.num_hid]
        last_backward_vals = output[:, 0, self.num_hid:]

        if self.ndirections == 1:
            return backward_vals, last_backward_vals

        return torch.cat((forward_vals, backward_vals), dim=-1), torch.cat((last_forward_vals, last_backward_vals), dim=-1)


class TPRencoder_lstm(nn.Module):
    def __init__(self, encoder_args):
        """
        TPR-LSTM encoder
        """
        for a in encoder_args.keys():
            setattr(self, a, encoder_args[a])

        self.out_dim = self.dSymbols * self.dRoles

        super(TPRencoder_lstm, self).__init__()
        assert self.rnn_type == 'LSTM' or self.rnn_type == 'GRU'
        if self.rnn_type == 'LSTM':
            rnn_cls = nn.LSTM
        else:
            rnn_cls = nn.GRU

        self.rnn_aF = rnn_cls(
            self.in_dim, self.out_dim, self.nlayers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True)
        self.rnn_aR = rnn_cls(
            self.in_dim, self.out_dim, self.nlayers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True)

        self.scale = nn.Parameter(torch.tensor(self.scale_val, dtype=self.get_dtype()), requires_grad=self.train_scale)
        print('self.scale requires grad is: {}'.format(self.scale.requires_grad))

        self.ndirections = 1 + int(self.bidirect)
        self.F = ScaleLinear(self.nSymbols, self.dSymbols, scale_val=self.scale_val)

        if self.fixed_Role:
            self.R = nn.Parameter(torch.eye(self.nRoles), requires_grad=False)
        else:
            self.R = ScaleLinear(self.nRoles, self.dRoles, scale_val=self.scale_val)
        self.WaF = nn.Linear(self.out_dim, self.nSymbols)
        self.WaR = nn.Linear(self.out_dim, self.nRoles)
        self.softmax = nn.Softmax(dim=2)


    def get_dtype(self):
        return next(self.parameters()).dtype

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers, batch, self.out_dim)
        if self.rnn_type == 'LSTM':
            return (weight.new(*hid_shape).zero_(), weight.new(*hid_shape).zero_())
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        seq = x.size(1)
        hidden_aF = self.init_hidden(batch)  # includes both hidden state and cell state (h_n,c_n).
        hidden_aR = self.init_hidden(batch)
        if self.training:
            self.rnn_aF.flatten_parameters()
            self.rnn_aR.flatten_parameters()
        for i in range(seq):
            aF, hidden_aF = self.rnn_aF(x[:, [i], :], hidden_aF)
            aR, hidden_aR = self.rnn_aR(x[:, [i], :], hidden_aR)
            aF = self.WaF(aF)
            aR = self.WaR(aR)
            aF = self.softmax(aF / self.temperature)
            aR = self.softmax(aR / self.temperature)
            itemF = self.F(aF)
            # itemF = aF
            if not self.fixed_Role:
                itemR = self.R(aR)
            else:
                itemR = aR
            T = (torch.bmm(torch.transpose(itemF, 1, 2), itemR)).view(batch, -1)

            if self.rnn_type == 'LSTM':
                hidden_aF = (T.unsqueeze(0), hidden_aF[1])
                hidden_aR = (T.unsqueeze(0), hidden_aR[1])
            else:
                hidden_aF = T.unsqueeze(0)
                hidden_aR = T.unsqueeze(0)
            if i == 0:
                out = T.unsqueeze(1)
                aFs = aF
                aRs = aR
            else:
                out = torch.cat([out, T.unsqueeze(1)], 1)
                aFs = torch.cat([aFs, aF], 1)
                aRs = torch.cat([aRs, aR], 1)

        return out, aFs, aRs

    def backward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        seq = x.size(1)
        hidden_aF = self.init_hidden(batch)  # includes both hidden state and cell state (h_n,c_n).
        hidden_aR = self.init_hidden(batch)
        if self.training:
            self.rnn_aF.flatten_parameters()
            self.rnn_aR.flatten_parameters()
        for i in range(seq - 1, -1, -1):
            aF, hidden_aF = self.rnn_aF(x[:, [i], :], hidden_aF)
            aR, hidden_aR = self.rnn_aR(x[:, [i], :], hidden_aR)
            aF = self.WaF(aF)
            aR = self.WaR(aR)
            aF = self.softmax(aF / self.temperature)
            aR = self.softmax(aR / self.temperature)
            itemF = self.F(aF)
            if not self.fixed_Role:
                itemR = self.R(aR)
            else:
                itemR = aR

            T = (torch.bmm(torch.transpose(itemF, 1, 2), itemR)).view(batch, -1)

            if self.rnn_type == 'LSTM':
                hidden_aF = (T.unsqueeze(0), hidden_aF[1])
                hidden_aR = (T.unsqueeze(0), hidden_aR[1])
            else:
                hidden_aF = T.unsqueeze(0)
                hidden_aR = T.unsqueeze(0)
            if i == seq - 1:
                out = T.unsqueeze(1)
                aFs = aF
                aRs = aR
            else:
                out = torch.cat([T.unsqueeze(1), out], 1)
                aFs = torch.cat([aF, aFs], 1)
                aRs = torch.cat([aR, aRs], 1)

        return out, aFs, aRs

    def call(self, x):

        R_flat = self.R.weight.view(self.R.weight.shape[0], -1)
        R_loss = torch.norm(torch.mm(R_flat, R_flat.t()) - torch.eye(R_flat.shape[0]).type(x.type())).pow(2) +\
                     torch.norm(torch.mm(R_flat.t(), R_flat) - torch.eye(R_flat.shape[1]).type(x.type())).pow(2)

        if self.ndirections == 1:
            out, aFs, aRs = self.backward(x)
            return out, (out[:, 0, :], aFs, aRs), R_loss

        else:
            out_f, aFs_f, aRs_f = self.forward(x)
            out_b, aFs_b, aRs_b = self.backward(x)

            out = torch.cat((out_f[:, :, :], out_b[:, :, :]), dim=-1)
            last_out = torch.cat((out_f[:, -1, :], out_b[:, 0, :]), dim=-1)
            aFs = torch.stack((aFs_f, aFs_b), dim=-1)
            aRs = torch.stack((aRs_f, aRs_b), dim=-1)
            return out, (last_out, aFs, aRs), R_loss


class TPRencoder_transformers(nn.Module):
    def __init__(self, encoder_args):
        """
        TPR_Transformer encoder
        """
        super(TPRencoder_transformers, self).__init__()

        for a in encoder_args.keys():
            setattr(self, a, encoder_args[a])

        self.enc_aF = TransformerEncoderLayer(self.in_dim, self.num_heads, self.num_hid, self.dropout)
        self.enc_aR = TransformerEncoderLayer(self.in_dim, self.num_heads, self.num_hid, self.dropout)
        if self.extra_layers != 0:
            self.T_list = nn.ModuleList([TransformerEncoderLayer(self.dRoles * self.dSymbols, self.num_heads, self.dRoles * self.dSymbols, self.dropout)
                                         for _ in range(self.extra_layers)])
            self.scale = nn.Parameter(torch.tensor(self.scale_val, dtype=self.get_dtype()), requires_grad=self.train_scale)
            print('self.scale requires grad is: {}'.format(self.scale.requires_grad))

        self.F = nn.Linear(self.nSymbols, self.dSymbols)

        if self.fixed_Role:
            self.R = nn.Parameter(torch.eye(self.nRoles), requires_grad=False)
        else:
            self.R = nn.Linear(self.nRoles, self.dRoles)

        self.WaF = nn.Linear(self.in_dim, self.nSymbols)
        self.WaR = nn.Linear(self.in_dim, self.nRoles)
        self.softmax = nn.Softmax(dim=2)

    def get_dtype(self):
        return next(self.parameters()).dtype

    def get_device(self):
        return next(self.parameters()).device


    def forward(self, x, src_key_padding_mask=None):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        seq = x.size(1)

        src_mask = None
        src_key_pad_mask = None
        # if self.do_src_mask:
            # src_key_pad_mask = src_key_padding_mask

        aF = self.enc_aF(x.transpose(0, 1), src_mask=src_mask, src_key_padding_mask=src_key_pad_mask)
        aR = self.enc_aR(x.transpose(0, 1), src_mask=src_mask, src_key_padding_mask=src_key_pad_mask)

        aF = self.WaF(aF)
        aR = self.WaR(aR)
        aF = self.softmax(aF / self.temperature)
        aR = self.softmax(aR / self.temperature)
        itemF = self.F(aF)

        if not self.fixed_Role:
            itemR = self.R(aR)
        else:
            itemR = aR
        T = torch.einsum('tbf,tbr->tbfr', [itemF, itemR]).view(seq, batch, -1)

        for i in range(self.extra_layers):
            T = T * self.scale
            T = self.T_list[i](T, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        final_T = T.transpose(0, 1)

        return final_T, aF.transpose(0, 1), aR.transpose(0, 1)

    def call(self, x, src_key_padding_mask=None):

        R_flat = self.R.weight.view(self.R.weight.shape[0], -1)
        R_loss = torch.norm(torch.mm(R_flat, R_flat.t()) - torch.eye(R_flat.shape[0]).type(x.type())).pow(2) +\
                     torch.norm(torch.mm(R_flat.t(), R_flat) - torch.eye(R_flat.shape[1]).type(x.type())).pow(2)

        out, aFs, aRs = self.forward(x, src_key_padding_mask)
        return out, aFs, aRs, R_loss
