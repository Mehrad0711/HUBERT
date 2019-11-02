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
from transformers.modeling_utils import PreTrainedModel
from modules.encoders import RNNencoder, TPRencoder_lstm, TPRencoder_transformers


class HUTransformer(PreTrainedModel):
    """
    HUTransformer model class
    """

    def __init__(self, config, transformer_model, num_labels, task_type, temperature, max_seq_len, **kwargs):
        super(HUTransformer, self).__init__(config)
        self.num_labels = num_labels
        self.task_type = task_type
        self.sub_word_masking = kwargs['sub_word_masking']
        self.ortho_reg = kwargs.get('ortho_reg', 0.0)

        self.transformer = transformer_model

        nRoles, nSymbols, dRoles, dSymbols = kwargs['nRoles'], kwargs['nSymbols'], kwargs['dRoles'], kwargs['dSymbols']

        # freeze transformer layers
        if kwargs['freeze_transformer']:
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

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            initializer_range = self.config.initializer_range if hasattr(self.config, 'initializer_range') else self.config.embed_init_std
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def number_transformer_layer(self):
        return len(self.transformer.encoder.layer)

    def freeze_layers(self, max_n):
        assert max_n <= self.number_transformer_layer()
        for p in self.transformer.pooler.parameters():
            p.requires_grad = False
        for p in self.transformer.embeddings.parameters():
            p.requires_grad = False
        for i in range(0, max_n):
            self.freeze_layer(i)

    def freeze_layer(self, n):
        assert n < self.number_transformer_layer()
        encode_layer = self.transformer.encoder.layer[n]
        for p in encode_layer.parameters():
            p.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sub_word_masks=None, labels=None):

        batch_size = input_ids.size(0)
        R_loss = None

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids
        output = self.transformer(**inputs)

        sequence_output = output[0]

        final_mask = attention_mask.unsqueeze(2).type(sequence_output.type())
        sequence_output_masked = sequence_output * final_mask

        # obtain output of encoder
        if self.encoder == 'lstm':
            output, last_output = self.head.call(sequence_output_masked)

        elif self.encoder == 'tpr_lstm':
            # aFs/ aRs: [batch, seq_len, nSymbols, num_directions]
            output, (last_output, aFs, aRs), R_loss = self.head.call(sequence_output_masked)

        elif self.encoder == 'tpr_transformers':
            output, aFs, aRs, R_loss = self.head.call(sequence_output_masked, attention_mask)

        else:
            output = sequence_output_masked

        # aggregate encoder output values
        if self.aggregate == 'sum':
            cls_input = torch.sum(output, dim=1)

        elif self.aggregate == 'mean':
            cls_input = torch.mean(output, dim=1)

        elif self.aggregate == 'concat':
            output_flattened = output.contiguous().view(batch_size, -1)
            cls_input = self.proj(output_flattened)

        else:
            cls_input = last_output if 'lstm' in self.encoder else output[:, 0]


        logits = self.classifier(cls_input)
        total_loss = None

        if labels is not None:
            if self.task_type == 0:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                total_loss = loss + self.ortho_reg * R_loss if R_loss is not None else loss
            else:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))
                total_loss = loss + self.ortho_reg * R_loss if R_loss is not None else loss

        return logits, total_loss
