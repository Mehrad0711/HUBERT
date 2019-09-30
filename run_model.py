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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import shutil

import numpy as np
import torch
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.modeling_bert import BertModel
from transformers.optimization import AdamW, WarmupLinearSchedule
from transformers.tokenization_bert import BertTokenizer
from tensorboardX import SummaryWriter
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from arguments import define_args
from modeling_tpr import BertForSequenceClassification_tpr
from utils.data_utils import *
from utils.evaluation import evaluate
from utils.prediction import predict

PROCESSORS = {
    'dnc_acc': ACCProcessor,
    'dnc_nli': NLIProcessor,
    'hans': HANSProcessor,
    'mnli': MNLIProcessor,
    'snli': SNLIProcessor,
    'qqp': QQPProcessor,
    'qnli': QNLIProcessor,
    'wnli': WNLIProcessor,
    'rte': RTEProcessor,
    'mrpc': MRPCProcessor,
    'sst': SSTProcessor,
    'sts': STSProcessor,
    'cola': COLAProcessor,
    'copa': COPAProcessor
}

NUM_LABELS_TASK = {
 'dnc_acc': 2,
 'dnc_nli': 3,
 'hans': 3, # 3-way prediction followed by combining contradiction and neutral into one
 'mnli': 3,
 'snli': 3,
 'qqp': 2,
 'qnli': 2,
 'wnli': 2,
 'rte': 2,
 'mrpc': 2,
 'sst': 2,
 'sts': 1,
 'cola': 2,
 'copa': 2
}

# 0 for classification and 1 for regression
TASK_TYPE = {
 'dnc_acc': 0,
 'dnc_nli': 0,
 'hans': 0,
 'mnli': 0,
 'snli': 0,
 'qqp': 0,
 'qnli': 0,
 'wnli': 0,
 'rte': 0,
 'mrpc': 0,
 'sst': 0,
 'sts': 1,
 'cola': 0,
 'copa': 0
}


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def decay(value, mode, final_ratio, global_step, t_total):
    assert final_ratio <= 1.0
    if mode == 'exp':
        alpha = np.log(final_ratio)
        new_value = value * np.exp(-alpha * global_step / t_total)

    elif mode == 'lin':
        alpha = 1 - final_ratio
        new_value = value * (1 - alpha * global_step / t_total)

    return new_value


def main(args):
    if os.path.exists(args.output_dir) and args.do_train:
        if args.delete_ok:
            shutil.rmtree(args.output_dir)
        else:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.log_dir) and args.do_train:
        if args.delete_ok:
            shutil.rmtree(args.log_dir)
        else:
            raise ValueError("Logging directory ({}) already exists and is not empty.".format(args.log_dir))
    os.makedirs(args.log_dir, exist_ok=True)

    # create logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.log_dir, 'log.txt')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('** output_dir is {} **'.format(args.output_dir))

    tensorboard_writer = SummaryWriter(args.log_dir)

    if args.fixed_Role:
        if args.nRoles != args.dRoles:
            logger.warning('Role dimension should be the same as number of Roles when using one-hot embedding')
            logger.warning('changing Role dimension from {} to {} to match number of Roles'.format(args.dRoles, args.nRoles))
            setattr(args, 'dRoles', args.nRoles)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not any([args.do_train, args.do_eval, args.do_test]):
        raise ValueError("At least one of `do_train` or `do_eval` or 'do_test' must be True.")
    task_name = args.task_name.lower()

    if task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (task_name))

    processor = PROCESSORS[task_name](args.num_ex)
    num_labels = NUM_LABELS_TASK[task_name]
    task_type = TASK_TYPE[task_name]
    label_list = None
    if task_type != 1:
        label_list = processor.get_labels()

    if 'uncased' in args.bert_model and not args.do_lower_case:
        logger.warning('do_lower_case should be True if uncased bert models are used')
        logger.warning('changing do_lower_case from False to True')
        setattr(args, 'do_lower_case', True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.do_train or args.do_eval:

        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** evaluation data *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_sub_word_masks = torch.tensor([f.sub_word_masks for f in eval_features], dtype=torch.long)
        if task_type != 1:
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        else:
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float32)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)


    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # Prepare model
        opt = {'bidirect': args.bidirect, 'sub_word_masking': args.sub_word_masking,
               'encoder': args.encoder, 'fixed_Role': args.fixed_Role, 'scale_val': args.scale_val, 'train_scale': args.train_scale,
               'pooling': args.pooling, 'freeze_bert': args.freeze_bert, 'num_rnn_layers': args.num_rnn_layers,
               'num_heads': args.num_heads, 'do_src_mask': args.do_src_mask, 'ortho_reg': args.ortho_reg, 'cls': args.cls}
        logger.info('*' * 50)
        logger.info('option for training: {}'.format(args))
        logger.info('*' * 50)

        # also print it for philly debugging
        print('option for training: {}'.format(args))

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
        if args.load_ckpt:
            output_model_file = os.path.join(args.load_ckpt)
            states = torch.load(output_model_file, map_location=device)
            model_state_dict = states['state_dict']
            # options shouldn't be loaded from the pre-trained model
            # opt = states['options']
            desired_keys = []
            if args.load_role:
                logger.info('loading roles from checkpoint model')
                desired_keys.extend(['rnn.R.weight', 'rnn.R.bias'])
            if args.load_filler:
                logger.info('loading fillers from checkpoint model')
                desired_keys.extend(['rnn.F.weight', 'rnn.F.bias'])
            if args.load_bert_params:
                logger.info('loading bert params from checkpoint model')
                desired_keys.extend([name for name in model_state_dict.keys() if name.startswith('bert')])
            if args.load_LSTM_params:
                logger.info('loading LSTM params from checkpoint model')
                desired_keys.extend([name for name in model_state_dict.keys() if name.startswith('rnn.rnn')])

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

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
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

        global_step = 0
        best_eval_accuracy = -float('inf')

        train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_sub_word_masks = torch.tensor([f.sub_word_masks for f in train_features], dtype=torch.long)
        if task_type != 1:
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        else:
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float32)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, sub_word_masks, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, sub_word_masks, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                    if args.debug:
                        print('\n')
                        for name, value in model.named_parameters():
                            if value.requires_grad and value.grad is not None:
                                print('{}: {}'.format(name, (torch.max(abs(value.grad)), torch.mean(abs(value.grad)), torch.min(abs(value.grad)))))


                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:

                    # modify scaling factor
                    with torch.no_grad():
                        pre = model.module if hasattr(model, 'module') else model
                        if args.do_decay and hasattr(pre.rnn, 'scale'):
                            pre.rnn.scale.copy_(torch.tensor(decay(pre.rnn.scale.cpu().numpy(), args.mode, args.final_ratio, global_step, t_total), dtype=pre.rnn.scale.dtype))

                    if args.debug:
                        pre = 'module' if hasattr(model, 'module') else ''

                        cls_w = dict(model.named_parameters())[pre + 'classifier.weight'].clone()
                        cls_b = dict(model.named_parameters())[pre + 'classifier.bias'].clone()

                    if args.optimizer == 'adam':
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                    else:
                        optimizer.step()

                    if args.debug:
                        new_cls_w = dict(model.named_parameters())[pre + 'classifier.weight'].clone()
                        new_cls_b = dict(model.named_parameters())[pre + 'classifier.bias'].clone()

                        print('diff weight is: {}'.format(new_cls_w - cls_w))
                        print('diff bias is: {}'.format(new_cls_b - cls_b))

                    optimizer.zero_grad()
                    global_step += 1

                if (global_step % args.log_every == 0) and (global_step != 0):
                    tensorboard_writer.add_scalar('train/loss', tr_loss, global_step)

            # Save a trained model after each epoch
            if not args.save_best_only or not args.do_eval:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model_{}.bin".format(epoch))
                logger.info("Saving checkpoint pytorch_model_{}.bin to {}".format(epoch, args.output_dir))
                torch.save({'state_dict': model_to_save.state_dict(), 'options': opt, 'bert_config': bert_config}, output_model_file)

            if args.do_eval:
                # evaluate model after every epoch
                model.eval()
                result = evaluate(args, model, eval_dataloader, device, task_type, global_step, tr_loss, nb_tr_steps)
                for key in sorted(result.keys()):
                    if key == 'eval_loss':
                        tensorboard_writer.add_scalar('eval/loss', result[key], global_step)
                    elif key == 'eval_accuracy':
                        tensorboard_writer.add_scalar('eval/accuracy', result[key], global_step)
                    logger.info("  %s = %s", key, str(result[key]))

                if result['eval_accuracy'] >= best_eval_accuracy:
                    best_eval_accuracy = result['eval_accuracy']
                    # Save the best model
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model_best.bin")
                    logger.info("Saving checkpoint pytorch_model_best.bin to {}".format(args.output_dir))
                    torch.save({'state_dict': model_to_save.state_dict(), 'options': opt, 'bert_config': bert_config}, output_model_file)


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Load a trained model for evaluation
        if args.do_train:
            output_model_file = os.path.join(args.output_dir, 'pytorch_model_best.bin')
        else:
            output_model_file = os.path.join(args.load_ckpt)
        states = torch.load(output_model_file, map_location=device)
        model_state_dict = states['state_dict']
        opt = states['options']
        bert_config = states['bert_config']
        if 'rnn.scale' in model_state_dict.keys():
            print('scale value is:', model_state_dict['rnn.scale'])
        logger.info('*' * 50)
        logger.info('option for evaluation: {}'.format(args))
        logger.info('*' * 50)
        # also print it for philly debugging
        print('option for evaluation: {}'.format(args))
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
        if args.classifier_ckpt:
            logger.info('loading classifier from checkpoint')
            classifier_state = torch.load(args.classifier_ckpt, map_location=device)['state_dict']
            for k in classifier_state.keys():
                if k.startswith('classifier'):
                    model_state_dict[k] = classifier_state[k]

        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        result = evaluate(args, model, eval_dataloader, device, task_type)


        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        logger.info("***** Eval results *****")
        logger.info("eval output file is in {}".format(output_eval_file))
        with open(output_eval_file, "w") as writer:
            writer.write('exp_{:s}_{:.3f}_{:.6f}_{:.0f}_{:.1f}_{:.0f}_{:.0f}_{:.0f}_{:.0f}\n'
                         .format(task_name, args.temperature, args.learning_rate, args.train_batch_size,
                                 args.num_train_epochs, args.dSymbols, args.dRoles, args.nSymbols, args.nRoles))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # Load a trained model for evaluation
        if args.do_train:
            output_model_file = os.path.join(args.output_dir, 'pytorch_model_best.bin')
        else:
            output_model_file = os.path.join(args.load_ckpt)

        states = torch.load(output_model_file, map_location=device)
        model_state_dict = states['state_dict']
        opt = states['options']
        bert_config = states['bert_config']
        if 'rnn.scale' in model_state_dict.keys():
            print('scale value is:', model_state_dict['rnn.scale'])
        logger.info('*' * 50)
        logger.info('option for evaluation: {}'.format(args))
        logger.info('*' * 50)

        # also print it for philly debugging
        print('option for evaluation: {}'.format(args))

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
        if args.classifier_ckpt:
            logger.info('loading classifier from checkpoint')
            classifier_state = torch.load(args.classifier_ckpt, map_location=device)['state_dict']
            for k in classifier_state.keys():
                if k.startswith('classifier'):
                    model_state_dict[k] = classifier_state[k]

        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        # prepare test data
        if task_name.startswith('dnc'):
            test_features = defaultdict()
            test_examples = processor.get_all_examples(args.data_dir)
            for k in test_examples.keys():
                test_features = convert_examples_to_features(test_examples[k], label_list, args.max_seq_length, tokenizer)
                logger.info("***** Running Testing *****")
                logger.info("  Num examples = %d", len(test_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
                all_sub_word_masks = torch.tensor([f.sub_word_masks for f in test_features], dtype=torch.long)

                all_guids = [f.guid for f in test_features]
                test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks)
                test_sampler = SequentialSampler(test_data)
                test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
                result = predict(args, model, test_dataloader, all_guids, device, task_type)

                output_test_file = os.path.join(args.output_dir, "{}-test_predictions.txt".format(k))
                logger.info("***** Test predictions *****")
                logger.info("test output file is in {}".format(output_test_file))
                with open(output_test_file, "w") as writer:
                    writer.write("index\tpredictions\n")
                    for id, pred in zip(result['input_ids'], result['predictions']):
                        writer.write("%s\t%s\n" % (id, label_list[pred]))

        else:
            test_examples = processor.get_test_examples(args.data_dir)
            test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
            logger.info("***** Running Testing *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
            all_sub_word_masks = torch.tensor([f.sub_word_masks for f in test_features], dtype=torch.long)

            all_guids = [f.guid for f in test_features]

            if task_name == 'snli':
                all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
                test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks, all_label_ids)
            else:
                test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks)

            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

            result = predict(args, model, test_dataloader, all_guids, device, task_type)

            output_test_file = os.path.join(args.output_dir, "test_predictions.txt")
            logger.info("***** Test predictions *****")
            logger.info("test output file is in {}".format(output_test_file))
            with open(output_test_file, "w") as writer:
                writer.write("index\tpredictions\n")
                for id, pred in zip(result['input_ids'], result['predictions']):
                    if task_name == 'hans':
                        if pred == 2: pred = 0  #consider neutral as non-entailment
                        writer.write("%s,%s\n" % (id, label_list[pred]))
                    elif task_type == 1:
                        writer.write("%s\t%s\n" % (id, pred))
                    else:
                        writer.write("%s\t%s\n" % (id, label_list[pred]))
                if task_name == 'snli':
                    writer.write("test_accuracy:\t%s\n" % (result['test_accuracy']))


if __name__ == "__main__":
    args = define_args()
    main(args)
