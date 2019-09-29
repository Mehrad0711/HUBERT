# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from tpr import RNNencoder, TPRencoder_lstm, TPRencoder_transformers


class BertForSequenceClassification_tpr(BertPreTrainedModel):
    """
    BERT model for classification (+ tpr)
    """
    def __init__(self, config, nSymbols, nRoles, dSymbols, dRoles, temperature, max_seq_len, num_labels, **kwargs):
        super(BertForSequenceClassification_tpr, self).__init__(config)
        self.num_labels = num_labels
        self.sub_word_masking = kwargs['sub_word_masking']
        self.ortho_reg = kwargs.get('ortho_reg', 0.0)
        self.bert = BertModel(config)

        # freeze bert layers
        if kwargs['freeze_bert']:
            self.freeze_layers(12)

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_layers = kwargs['num_rnn_layers']
        self.num_directs = 1 + int(kwargs['bidirect'])
        self.cls = kwargs.get('cls', 'v1')
        self.encoder = kwargs['encoder']
        self.pooling = kwargs['pooling']

        hidden_size = config.hidden_size

        if self.encoder == 'lstm':
            self.rnn = RNNencoder(hidden_size, hidden_size, nlayers=self.num_layers, bidirect=kwargs['bidirect'], dropout=0.0, rnn_type='LSTM')
            if self.cls == 'v1':
                self.classifier = nn.Linear(hidden_size * self.num_directs, num_labels)
            elif self.cls == 'v2':
                self.activation1 = nn.LeakyReLU()
                self.linear1 = nn.Linear(hidden_size * self.num_directs, hidden_size * self.num_directs // 10)
                self.activation2 = nn.LeakyReLU()
                self.linear2 = nn.Linear(hidden_size * self.num_directs // 10, hidden_size * self.num_directs // 100,)
                self.classifier = nn.Sequential(self.linear1, self.activation1, self.linear2, self.activation2)

            if self.pooling == 'concat':
                self.proj = nn.Linear(max_seq_len * hidden_size * self.num_directs, hidden_size * self.num_directs)

        elif self.encoder == 'tpr_lstm':
            self.rnn = TPRencoder_lstm(hidden_size, nSymbols, nRoles, dSymbols, dRoles, temperature,
                                  self.num_layers, bidirect=kwargs['bidirect'], dropout=0.0, fixed_Role=kwargs['fixed_Role'], scale_val=kwargs['scale_val'], train_scale=kwargs['train_scale'], rnn_type='LSTM')

            if self.cls == 'v1':
                self.classifier = nn.Linear(dSymbols * dRoles * self.num_directs, num_labels)
            elif self.cls == 'v2':
                self.activation1 = nn.LeakyReLU()
                self.linear1 = nn.Linear(dSymbols * dRoles * self.num_directs, dSymbols * dRoles * self.num_directs // 10)
                self.activation2 = nn.LeakyReLU()
                self.linear2 = nn.Linear(dSymbols * dRoles * self.num_directs // 10, dSymbols * dRoles * self.num_directs // 100)
                self.classifier = nn.Sequential(self.linear1, self.activation1, self.linear2, self.activation2)

            if self.pooling == 'concat':
                self.proj = nn.Linear(max_seq_len * dSymbols * dRoles * self.num_directs, dSymbols * dRoles * self.num_directs)


        elif self.encoder == 'tpr_transformers':
            args = {'in_dim': hidden_size, 'num_hid': hidden_size, 'num_heads': kwargs['num_heads'], 'nSymbols': nSymbols, 'nRoles': nRoles, 'dSymbols': dSymbols, 'dRoles': dRoles,
                    'temperature': temperature, 'nlayers': self.num_layers, 'dropout': 0.0, 'fixed_Role': kwargs['fixed_Role'], 'train_scale': kwargs['train_scale'], 'scale_val': kwargs['scale_val']}
            args.update({'do_src_mask': kwargs['do_src_mask']})
            self.rnn = TPRencoder_transformers(args)

            if self.cls == 'v1':
                self.classifier = nn.Linear(dSymbols * dRoles, num_labels)
            elif self.cls == 'v2':
                self.activation1 = nn.LeakyReLU()
                self.linear1 = nn.Linear(dSymbols * dRoles, dSymbols * dRoles // 10)
                self.activation2 = nn.LeakyReLU()
                self.linear2 = nn.Linear(dSymbols * dRoles // 10, dSymbols * dRoles // 100)
                self.classifier = nn.Sequential(self.linear1, self.activation1, self.linear2, self.activation2)

            if self.pooling == 'none':
                print('transformer_tpr outputs should be pooled! since no pooling is provided #concat# will be used by default')
                self.pooling = 'concat'
            if self.pooling == 'concat':
                self.proj = nn.Linear(max_seq_len * dSymbols * dRoles, dSymbols * dRoles)

        else:
            self.classifier = nn.Linear(768, num_labels)

        if hasattr(self, 'rnn'):
            print('num_elems:', sum([p.nelement() for p in self.rnn.parameters() if p.requires_grad]))

        self.apply(self.init_bert_weights)

    def nbert_layer(self):
        return len(self.bert.encoder.layer)

    def freeze_layers(self, max_n):
        assert max_n <= self.nbert_layer()
        for p in self.bert.pooler.parameters():
            p.requires_grad = False
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for i in range(0, max_n):
            self.freeze_layer(i)

    def freeze_layer(self, n):
        assert n < self.nbert_layer()
        encode_layer = self.bert.encoder.layer[n]
        for p in encode_layer.parameters():
            p.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sub_word_masks=None, labels=None):
        batch_size = input_ids.size(0)
        R_loss = None


        yb, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        final_mask = attention_mask.unsqueeze(2).type(yb.type())
        yb_masked = yb * final_mask
        if self.encoder == 'lstm':
            yb_rnn, last_yb_rnn = self.rnn.call(yb_masked)

        elif self.encoder == 'tpr_lstm':
            yb_rnn, (last_yb_rnn, aFs, aRs), R_loss = self.rnn.call(yb_masked) # aFs/ aRs: [batch, seq_len, nSymbols, num_directions]

        elif self.encoder == 'tpr_transformers':
            yb_rnn, aFs, aRs, R_loss = self.rnn.call(yb_masked, attention_mask)

        else:
            last_yb_rnn = pooled_output


        if self.pooling == 'sum':
            cls_input = torch.sum(yb_rnn, dim=1)
        elif self.pooling == 'mean':
            cls_input = torch.mean(yb_rnn, dim=1)
        elif self.pooling == 'concat':
            yb_rnn_flattened = yb_rnn.contiguous().view(batch_size, -1)
            cls_input = self.proj(yb_rnn_flattened)
        else:
            cls_input = last_yb_rnn

        logits = self.classifier(cls_input)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss + self.ortho_reg * R_loss if R_loss is not None else loss
        else:
            return logits