from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os

from modules.model import BertForSequenceClassification_tpr
from utils.data_utils import convert_examples_to_features, logger
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.modeling_bert import BertModel
from transformers.optimization import AdamW, WarmupLinearSchedule
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def prepare_data_loader(args, processor, label_list, task_type, task, tokenizer, split, examples=None, single_sentence=False,
                        return_pos_tags=False, return_ner_tags=False, return_dep_parse=False, return_const_parse=False):

    data_dir = os.path.join(args.data_dir, task)

    if examples is None:
        if split == 'train':
            examples = processor.get_train_examples(data_dir)
        if split == 'dev':
            examples = processor.get_dev_examples(data_dir)
        if split == 'test':
            examples = processor.get_test_examples(data_dir)

    features, structure_features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, single_sentence,
                                                                  return_pos_tags, return_ner_tags, return_dep_parse, return_const_parse)
    token_pos, token_ner, token_dep, token_const = structure_features

    logger.info("***** preparing data *****")
    logger.info("  Num examples = %d", len(examples))
    batch_size = args.train_batch_size if split == 'train' else args.eval_batch_size
    logger.info("  Batch size = %d", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.uint8)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_sub_word_masks = torch.tensor([f.sub_word_masks for f in features], dtype=torch.uint8)
    all_orig_to_token_maps = torch.tensor([f.orig_to_token_map for f in features], dtype=torch.long)

    if split == 'test':
        if task.lower() == 'snli':
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks, all_orig_to_token_maps, all_label_ids)
        else:
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks, all_orig_to_token_maps)
    else:
        if task_type != 1:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        else:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float32)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks, all_orig_to_token_maps, all_label_ids)

    if split == 'train' and not args.save_tpr_attentions:
        if args.local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    else:
        sampler = SequentialSampler(data)

    all_guids = [f.guid for f in features]
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader, all_guids, structure_features


def prepare_optim(args, num_train_steps, param_optimizer):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    scheduler = None
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        if args.optimizer == 'adam':
            optimizer = AdamW(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 correct_bias=False)
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_proportion * t_total, t_total=t_total)

        elif args.optimizer == 'radam':
            optimizer = RAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer = SGD(optimizer_grouped_parameters,
                                 lr=args.learning_rate)

    return optimizer, scheduler, t_total


def prepare_model(args, opt, num_labels, task_type, device, n_gpu, loading_path=None):

    # Load config and pre-trained model
    pre_trained_model = BertModel.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    bert_config = pre_trained_model.config

    # modify config
    bert_config.num_hidden_layers = args.num_bert_layers
    model = BertForSequenceClassification_tpr(bert_config,
                                              num_labels=num_labels,
                                              task_type=task_type,
                                              temperature=args.temperature,
                                              max_seq_len=args.max_seq_length,
                                              **opt)

    # load desired layers from config
    model.bert.load_state_dict(pre_trained_model.state_dict(), strict=False)

    # initialize Symbol and Filler parameters from a checkpoint instead of randomly initializing them
    if loading_path:
        logger.info('loading model checkpoint from {}'.format(loading_path))
        output_model_file = os.path.join(loading_path)
        states = torch.load(output_model_file, map_location=device)
        model_state_dict = states['state_dict']
        # options shouldn't be loaded from the pre-trained model
        # opt = states['options']
        desired_keys = []
        if args.load_role:
            logger.info('loading roles from checkpoint model')
            desired_keys.extend(['head.R.weight', 'head.R.bias'])
        if args.load_filler:
            logger.info('loading fillers from checkpoint model')
            desired_keys.extend(['head.F.weight', 'head.F.bias'])
        if args.load_role_selector:
            logger.info('loading role selectors from checkpoint model')
            desired_keys.extend(['head.WaR.weight', 'head.WaR.bias'])
            desired_keys.extend([name for name in model_state_dict.keys() if name.startswith('head.enc_aR')])
        if args.load_filler_selector:
            logger.info('loading filler selectors from checkpoint model')
            desired_keys.extend(['head.WaF.weight', 'head.WaF.bias'])
            desired_keys.extend([name for name in model_state_dict.keys() if name.startswith('head.enc_aF')])
        if args.load_bert_params:
            logger.info('loading bert params from checkpoint model')
            desired_keys.extend([name for name in model_state_dict.keys() if name.startswith('bert')])
        if args.load_classifier:
            logger.info('loading classifier params from checkpoint model')
            desired_keys.extend([name for name in model_state_dict.keys() if name.startswith('classifier')])
        if args.load_LSTM_params:
            logger.info('loading LSTM params from checkpoint model')
            desired_keys.extend([name for name in model_state_dict.keys() if name.startswith('head.rnn')])

        state = dict()

        for key, val in model_state_dict.items():
            if key in desired_keys:
                state[key] = val

        model.load_state_dict(state, strict=False)

        frozen_keys = []
        if args.freeze_role:
            logger.info('freezing roles if loaded from ckpt model')
            frozen_keys.extend(['head.R.weight', 'head.R.bias'])
        if args.freeze_filler:
            logger.info('freezing fillers if loaded from ckpt model')
            frozen_keys.extend(['head.F.weight', 'head.F.bias'])
        if args.freeze_role_selector:
            logger.info('freezing role selectors if loaded from ckpt model')
            frozen_keys.extend(['head.WaR.weight', 'head.WaR.bias'])
            frozen_keys.extend([name for name in model_state_dict.keys() if name.startswith('head.enc_aR')])
        if args.freeze_filler_selector:
            logger.info('freezing filler selectors if loaded from ckpt model')
            frozen_keys.extend(['head.WaF.weight', 'head.WaF.bias'])
            frozen_keys.extend([name for name in model_state_dict.keys() if name.startswith('head.enc_aF')])
        if args.freeze_bert_params:
            logger.info('freezing bert params if loaded from ckpt model')
            frozen_keys.extend([name for name in model_state_dict.keys() if name.startswith('bert')])
        if args.freeze_classifier:
            logger.info('freezing classifier params if loaded from ckpt model')
            frozen_keys.extend([name for name in model_state_dict.keys() if name.startswith('classifier')])
        if args.freeze_LSTM_params:
            logger.info('freezing LSTM params if loaded from ckpt model')
            frozen_keys.extend([name for name in model_state_dict.keys() if name.startswith('head.rnn')])

        for name, param in model.named_parameters():
            if name in frozen_keys:
                param.requires_grad = False

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, bert_config


def prepare_structure_values(args, eval_task_name, all_ids, F_list, R_list, token_pos, token_ner, token_dep, token_const):

    values = {}
    if args.single_sentence or eval_task_name.lower() in ['sst', 'cola']:
        index = 0
        tokens = [[subval[0] for subval in val[index]] for val in token_pos]
        pos_tags = [[subval[1] for subval in val[index]] for val in token_pos]
        ner_tags = [[subval[1] for subval in val[index]] for val in token_ner]
        dep_parse_tokens = [[subval[0] for subval in val[index]] for val in token_dep]
        dep_parses = [[subval[1] for subval in val[index]] for val in token_dep]
        const_parses = [[subval[1] for subval in val[index]] for val in token_const]
        parse_tree_depths = [[len(subval[1]) for subval in val[index]] for val in token_const]

    else:
        tokens = []
        pos_tags = []
        ner_tags = []
        dep_parse_tokens = []
        dep_parse = []
        const_parse = []
        parse_tree_depth = []
        index = 0
        tokens_a = [[subval[0] for subval in val[index]] for val in token_pos]
        pos_tags_a = [[subval[1] for subval in val[index]] for val in token_pos]
        ner_tags_a = [[subval[1] for subval in val[index]] for val in token_ner]
        dep_parse_tokens_a = [[subval[0] for subval in val[index]] for val in token_dep]
        dep_parses_a = [[subval[1] for subval in val[index]] for val in token_dep]
        const_parses_a = [[subval[1] for subval in val[index]] for val in token_const]
        parse_tree_depths_a = [[len(subval[1]) for subval in val[index]] for val in token_const]
        index = 1
        tokens_b = [[subval[0] for subval in val[index]] for val in token_pos]
        pos_tags_b = [[subval[1] for subval in val[index]] for val in token_pos]
        ner_tags_b = [[subval[1] for subval in val[index]] for val in token_ner]
        dep_parse_tokens_b = [[subval[0] for subval in val[index]] for val in token_dep]
        dep_parses_b = [[subval[1] for subval in val[index]] for val in token_dep]
        const_parses_b = [[subval[1] for subval in val[index]] for val in token_const]
        parse_tree_depths_b = [[len(subval[1]) for subval in val[index]] for val in token_const]

        for token_a, token_b in zip(tokens_a, tokens_b):
            tokens.append(token_a + ['[SEP]'] + token_b)
        for pos_tag_a, pos_tag_b in zip(pos_tags_a, pos_tags_b):
            pos_tags.append(pos_tag_a + ['SEP'] + pos_tag_b)
        for ner_tag_a, ner_tag_b in zip(ner_tags_a, ner_tags_b):
            ner_tags.append(ner_tag_a + ['[SEP]'] + ner_tag_b)
        for dep_parse_token_a, dep_parse_token_b in zip(dep_parse_tokens_a, dep_parse_tokens_b):
            dep_parse_tokens.append(dep_parse_token_a + ['[SEP]'] + dep_parse_token_b)
        for dep_parse_a, dep_parse_b in zip(dep_parses_a, dep_parses_b):
            dep_parse.append(dep_parse_a + ['[SEP]'] + dep_parse_b)
        for const_parse_a, const_parse_b in zip(const_parses_a, const_parses_b):
            const_parse.append(const_parse_a + ['[SEP]'] + const_parse_b)
        for parse_tree_depth_a, parse_tree_depth_b in zip(parse_tree_depths_a, parse_tree_depths_b):
            parse_tree_depth.append(parse_tree_depth_a + ['[SEP]'] + parse_tree_depth_b)

    bad_sents_count = 0
    for i in range(len(all_ids)):
        try:
            assert len(tokens[i]) == len(F_list[i]) == len(R_list[i])
            val_i = {'tokens': tokens[i], 'all_aFs': F_list[i], 'all_aRs': R_list[i]}
            if args.return_POS:
                assert len(pos_tags[i]) == len(tokens[i])
                val_i.update({'pos_tags': pos_tags[i]})
            if args.return_NER:
                assert len(ner_tags[i]) == len(tokens[i])
                val_i.update({'ner_tags': ner_tags[i]})
            if args.return_DEP:
                assert len(dep_parse_tokens[i]) == len(dep_parse[i])
                val_i.update({'dep_parse_tokens': dep_parse_tokens[i],'dep_edge': dep_parse[i]})
            if args.return_CONST:
                assert len(const_parse[i]) == len(tokens[i])
                val_i.update({'cont_parse_path': const_parse[i]})
                assert len(parse_tree_depth[i]) == len(tokens[i])
                val_i.update({'tree_depth': parse_tree_depth[i]})

            values[all_ids[i]] = val_i
        except:
            bad_sents_count += 1
    logger.info('Could not parse {:.2f}% of the sentences out of all {} data points'.format(bad_sents_count/ len(all_ids)*100,  len(all_ids)))

    return values