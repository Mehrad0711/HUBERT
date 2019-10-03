from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from modules.model import BertForSequenceClassification_tpr
from utils.data_utils import *
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.modeling_bert import BertModel
from transformers.optimization import AdamW, WarmupLinearSchedule
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def prepare_data_loader(args, processor, label_list, task_type, task, tokenizer, split, examples=None):

    data_dir = os.path.join(args.data_dir, task)

    if examples is None:
        if split == 'train':
            examples = processor.get_train_examples(data_dir)
        if split == 'dev':
            examples = processor.get_dev_examples(data_dir)
        if split == 'test':
            examples = processor.get_test_examples(data_dir)

    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** preparing data *****")
    logger.info("  Num examples = %d", len(examples))
    batch_size = args.train_batch_size if split == 'train' else args.eval_batch_size
    logger.info("  Batch size = %d", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_sub_word_masks = torch.tensor([f.sub_word_masks for f in features], dtype=torch.long)

    if split == 'test':
        if args.task_name.lower() == 'snli':
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks, all_label_ids)
        else:
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks)
    else:
        if task_type != 1:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        else:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float32)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks, all_label_ids)

    if split == 'train':
        if args.local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    else:
        sampler = SequentialSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if split == 'test':
        all_guids = [f.guid for f in features]
        return dataloader, all_guids
    else:
        return dataloader

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
                                              nSymbols=args.nSymbols,
                                              nRoles=args.nRoles,
                                              dSymbols=args.dSymbols,
                                              dRoles=args.dRoles,
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
        if args.freeze_mat:
            logger.info('freezing all params loaded from ckpt model')
            for name, param in model.named_parameters():
                if name in desired_keys:
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
