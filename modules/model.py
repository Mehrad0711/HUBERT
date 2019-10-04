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
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from modules.encoders import RNNencoder, TPRencoder_lstm, TPRencoder_transformers


class BertForSequenceClassification_tpr(BertPreTrainedModel):
    """
    BERT model for classification (+ tpr)
    """

    def __init__(self, config, nSymbols, nRoles, dSymbols, dRoles, temperature, max_seq_len, num_labels, task_type, **kwargs):
        super(BertForSequenceClassification_tpr, self).__init__(config)
        self.num_labels = num_labels
        self.task_type = task_type
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
        self.aggregate = kwargs['aggregate']

        hidden_size = config.hidden_size

        # define the classifier
        if self.encoder == 'lstm':
            hidden_dimension = hidden_size * self.num_directs
        elif self.encoder == 'tpr_lstm':
            hidden_dimension = dSymbols * dRoles * self.num_directs
        elif self.encoder == 'tpr_transformers':
            hidden_dimension = dSymbols * dRoles
        else:
            hidden_dimension = hidden_size

        if self.cls == 'v1':
            self.classifier = nn.Linear(hidden_dimension, num_labels)
        elif self.cls == 'v2':
            self.activation1 = nn.LeakyReLU()
            self.linear1 = nn.Linear(hidden_dimension, hidden_dimension // 10)
            self.activation2 = nn.LeakyReLU()
            self.linear2 = nn.Linear(hidden_dimension // 10, hidden_dimension // 100, )
            self.classifier = nn.Sequential(self.linear1, self.activation1, self.linear2, self.activation2)

        # define aggregation layer
        if self.aggregate == 'none' and self.encoder == 'tpr_transformers':
            print('tpr_transformers outputs should be aggregated! since no aggregation is provided #concat# will be used by default')
            self.aggregate = 'concat'

        if self.aggregate == 'concat':
            self.proj = nn.Linear(max_seq_len * hidden_dimension, hidden_dimension)

        # define the encoder and aggregation layer
        if self.encoder == 'lstm':
            encoder_args = {'in_dim': hidden_size, 'num_hid': hidden_size, 'nlayers': self.num_layers,
                            'dropout': 0.0, 'bidirect': kwargs['bidirect'], 'rnn_type': 'LSTM'}
            self.head = RNNencoder(encoder_args)

        elif self.encoder == 'tpr_lstm':
            encoder_args = {'in_dim': hidden_size, 'nSymbols': nSymbols, 'nRoles': nRoles,
                            'dSymbols': dSymbols, 'dRoles': dRoles, 'temperature': temperature,
                           'nlayers': self.num_layers, 'bidirect': kwargs['bidirect'], 'dropout': 0.0,
                           'fixed_Role': kwargs['fixed_Role'], 'scale_val': kwargs['scale_val'],
                           'train_scale': kwargs['train_scale'], 'rnn_type': 'LSTM'}
            self.head = TPRencoder_lstm(encoder_args)

        elif self.encoder == 'tpr_transformers':
            encoder_args = {'in_dim': hidden_size, 'num_hid': hidden_size, 'num_heads': kwargs['num_heads'],
                    'nSymbols': nSymbols, 'nRoles': nRoles, 'dSymbols': dSymbols, 'dRoles': dRoles,
                    'temperature': temperature, 'dropout': 0.0,
                    'fixed_Role': kwargs['fixed_Role'], 'train_scale': kwargs['train_scale'],
                    'scale_val': kwargs['scale_val'], 'do_src_mask': kwargs['do_src_mask']}
            self.head = TPRencoder_transformers(encoder_args)

        else:
            self.classifier = nn.Linear(768, num_labels)

        self.apply(self._init_weights)

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

        yb, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        final_mask = attention_mask.unsqueeze(2).type(yb.type())
        yb_masked = yb * final_mask
        if self.encoder == 'lstm':
            yb_rnn, last_yb_rnn = self.head.call(yb_masked)

        elif self.encoder == 'tpr_lstm':
            yb_rnn, (last_yb_rnn, aFs, aRs), R_loss = self.head.call(
                yb_masked)  # aFs/ aRs: [batch, seq_len, nSymbols, num_directions]

        elif self.encoder == 'tpr_transformers':
            yb_rnn, aFs, aRs, R_loss = self.head.call(yb_masked, attention_mask)

        else:
            last_yb_rnn = pooled_output

        if self.aggregate == 'sum':
            cls_input = torch.sum(yb_rnn, dim=1)
        elif self.aggregate == 'mean':
            cls_input = torch.mean(yb_rnn, dim=1)
        elif self.aggregate == 'concat':
            yb_rnn_flattened = yb_rnn.contiguous().view(batch_size, -1)
            cls_input = self.proj(yb_rnn_flattened)
        else:
            cls_input = last_yb_rnn

        logits = self.classifier(cls_input)

        if labels is not None:
            if self.task_type == 0:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss + self.ortho_reg * R_loss if R_loss is not None else loss
            else:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))
                return loss + self.ortho_reg * R_loss if R_loss is not None else loss

        else:
            return logits
