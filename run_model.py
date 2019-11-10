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
import logging
import os
import sys
import json
import pickle

import numpy as np
import torch

from transformers.tokenization_bert import BertTokenizer
from tensorboardX import SummaryWriter
from transformers.configuration_bert import PretrainedConfig

from tqdm import tqdm, trange

from arguments import define_args
from utils.tasks import PROCESSORS, NUM_LABELS_TASK, TASK_TYPE
from modules.model import BertForSequenceClassification_tpr
from utils.evaluation import evaluate
from utils.prediction import predict
from utils.prepare import prepare_data_loader, prepare_model, prepare_optim, modify_model

import warnings
warnings.simplefilter("ignore", UserWarning)

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

    args.log_dir = args.output_dir #TODO
    if os.path.exists(args.output_dir):
        if args.delete_ok:
            shutil.rmtree(args.output_dir)
        else:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=False)

    # create logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.log_dir, 'log.txt')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    if len(args.cont_task_names) != len(set(args.cont_task_names)) or args.task_name in args.cont_task_names:
        logger.error('Please make sure all continual tasks are distinct and also different from source_task')
        sys.exit('Exiting the program...')

    logger.info('** output_dir is {} **'.format(args.output_dir))

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

    if 'uncased' in args.bert_model and not args.do_lower_case:
        logger.warning('do_lower_case should be True if uncased bert models are used')
        logger.warning('changing do_lower_case from False to True')
        setattr(args, 'do_lower_case', True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    all_tasks = [args.task_name] + args.cont_task_names

    # check for NAN values and end experiment promptly
    torch.autograd.set_detect_anomaly(True)

    if args.do_train:

        loading_path = args.load_ckpt

        for i, task in enumerate(all_tasks):

            if task.lower() not in PROCESSORS:
                raise ValueError("Task not found: %s" % (task.lower()))
            logger.info('*** Start training for {} ***'.format(task))

            processor = PROCESSORS[task.lower()](args.num_ex)
            num_labels = NUM_LABELS_TASK[task.lower()]
            task_type = TASK_TYPE[task.lower()]
            label_list = None
            if task_type != 1:
                label_list = processor.get_labels()

            # make output_dir
            os.makedirs(os.path.join(args.output_dir, task), exist_ok=False)

            # init tensorboard writer
            tensorboard_writer = SummaryWriter(os.path.join(args.log_dir, task))

            train_examples = processor.get_train_examples(os.path.join(args.data_dir, task))
            num_train_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

            if args.do_eval:
                # prepare eval data
                eval_dataloader = prepare_data_loader(args, processor, label_list, task_type, task, tokenizer, split='dev')

            # Prepare model
            opt = {'bidirect': args.bidirect, 'sub_word_masking': args.sub_word_masking, 'nRoles': args.nRoles, 'nSymbols': args.nSymbols,
                   'dRoles': args.dRoles, 'dSymbols': args.dSymbols, 'encoder': args.encoder, 'fixed_Role': args.fixed_Role,
                   'scale_val': args.scale_val, 'train_scale': args.train_scale, 'aggregate': args.aggregate, 'freeze_bert': args.freeze_bert,
                   'num_rnn_layers': args.num_rnn_layers, 'num_extra_layers': args.num_extra_layers, 'num_heads': args.num_heads, 'do_src_mask': args.do_src_mask,
                   'ortho_reg': args.ortho_reg, 'cls': args.cls}
            logger.info('*' * 50)
            logger.info('option for training: {}'.format(args))
            logger.info('*' * 50)
            # also print it for philly debugging
            print('option for training: {}'.format(args))

            model, bert_config = prepare_model(args, opt, num_labels, task_type, device, n_gpu, loading_path)

            print('num_elems:', sum([p.nelement() for p in model.parameters() if p.requires_grad]))

            # Prepare optimizer
            optimizer, scheduler, t_total = prepare_optim(args, num_train_steps, param_optimizer=list(model.named_parameters()))

            global_step = 0
            best_eval_accuracy = -float('inf')
            best_model = None

            train_dataloader = prepare_data_loader(args, processor, label_list, task_type, task, tokenizer, split='train')

            for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
                model.train()
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, sub_word_masks, orig_to_token_maps, label_ids = batch
                    _, loss, _ = model(input_ids, segment_ids, input_mask, sub_word_masks, label_ids)
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
                            if args.do_decay and hasattr(pre.head, 'scale'):
                                pre.head.scale.copy_(torch.tensor(decay(pre.head.scale.cpu().numpy(), args.mode, args.final_ratio, global_step, t_total), dtype=pre.head.scale.dtype))

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
                    output_model_file = os.path.join(*[args.output_dir, task, "pytorch_model_{}.bin".format(epoch)])
                    loading_path = output_model_file
                    logger.info("Saving checkpoint pytorch_model_{}.bin to {}".format(epoch, output_model_file))
                    torch.save({'state_dict': model_to_save.state_dict(), 'options': opt, 'bert_config': bert_config}, output_model_file)


                if args.do_eval:
                    # evaluate model after every epoch
                    model.eval()
                    result, _ = evaluate(args, model, eval_dataloader, device, task_type, global_step, tr_loss, nb_tr_steps)
                    for key in sorted(result.keys()):
                        if key == 'eval_loss':
                            tensorboard_writer.add_scalar('eval/loss', result[key], global_step)
                        elif key == 'eval_accuracy':
                            tensorboard_writer.add_scalar('eval/accuracy', result[key], global_step)
                        logger.info("  %s = %s", key, str(result[key]))

                    if result['eval_accuracy'] >= best_eval_accuracy:
                        best_eval_accuracy = result['eval_accuracy']
                        best_model = model
                        # Save the best model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(*[args.output_dir, task, "pytorch_model_best.bin"])
                        loading_path = output_model_file
                        logger.info("Saving checkpoint pytorch_model_best.bin to {}".format(output_model_file))
                        torch.save({'state_dict': model_to_save.state_dict(), 'options': opt, 'bert_config': bert_config}, output_model_file)

            if args.do_prev_eval:
                if best_model is None:
                    best_model = model
                best_model.eval()

                # evaluate best model on current task
                dev_task = task
                result, _ = evaluate(args, best_model, eval_dataloader, device, task_type)
                logger.info("train_task: {}, eval_task: {}".format(task, dev_task))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

                # evaluate new model on all previous tasks
                pre = best_model.module if hasattr(best_model, 'module') else best_model
                for j in range(i):
                    dev_task = all_tasks[j]

                    with torch.no_grad():
                        modify_model(best_model, dev_task, args)

                    processor = PROCESSORS[dev_task.lower()](args.num_ex)
                    num_labels = NUM_LABELS_TASK[dev_task.lower()]
                    task_type = TASK_TYPE[dev_task.lower()]
                    label_list = None
                    if task_type != 1:
                        label_list = processor.get_labels()
                    pre.num_labels = num_labels
                    pre.task_type = task_type
                    eval_dataloader = prepare_data_loader(args, processor, label_list, task_type, dev_task, tokenizer, split='dev')

                    result, _ = evaluate(args, best_model, eval_dataloader, device, task_type)
                    logger.info("train_task: {}, eval_task: {}".format(task, dev_task))
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))


            tensorboard_writer.close()


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_task_name = all_tasks[-1]
        logger.info('*** Start evaluating for {} ***'.format(eval_task_name))
        processor = PROCESSORS[eval_task_name.lower()](args.num_ex)
        num_labels = NUM_LABELS_TASK[eval_task_name.lower()]
        task_type = TASK_TYPE[eval_task_name.lower()]
        label_list = None
        if task_type != 1:
            label_list = processor.get_labels()

        # Load a trained model for evaluation
        if args.do_train:
            output_model_file = os.path.join(*[args.output_dir, all_tasks[-1], 'pytorch_model_best.bin'])
        else:
            output_model_file = os.path.join(args.load_ckpt)

        #prepare data
        split = args.data_split_attention if args.save_tpr_attentions else 'dev'
        only_b = True if args.save_tpr_attentions else False

        eval_dataloader = prepare_data_loader(args, processor, label_list, task_type, all_tasks[-1], tokenizer,
                                              split=split, single_sentence=args.single_sentence, only_b=only_b)

        states = torch.load(output_model_file, map_location=device)
        model_state_dict = states['state_dict']
        opt = states['options']
        if 'nRoles' not in opt:
            for val in ['nRoles', 'nSymbols', 'dRoles', 'dSymbols']:
                opt[val] = getattr(args, val)

        bert_config = states['bert_config']
        if not isinstance(bert_config, PretrainedConfig):
            bert_dict = bert_config.to_dict()
            bert_dict['layer_norm_eps'] = 1e-12
            bert_config = PretrainedConfig.from_dict(bert_dict)

        if 'head.scale' in model_state_dict.keys():
            print('scale value is:', model_state_dict['head.scale'])
        logger.info('*' * 50)
        logger.info('option for evaluation: {}'.format(args))
        logger.info('*' * 50)
        # also print it for philly debugging
        print('option for evaluation: {}'.format(args))
        model = BertForSequenceClassification_tpr(bert_config,
                                                  num_labels=num_labels,
                                                  task_type=task_type,
                                                  temperature=args.temperature,
                                                  max_seq_len=args.max_seq_length,
                                                  **opt)

        model.load_state_dict(model_state_dict, strict=False)

        if args.reset_temp_ratio != 1.0 and hasattr(model.head, 'temperature'):
            new_temp = model.head.temperature / args.reset_temp_ratio
            model.head.temperature = new_temp

        model.to(device)
        model.eval()
        result, (all_ids, F_list, R_list) = evaluate(args, model, eval_dataloader, device, task_type, data_split=args.data_split_attention)

        if not os.path.exists(os.path.join(args.output_dir, eval_task_name)):
            os.makedirs(os.path.join(args.output_dir, eval_task_name))

        if args.save_tpr_attentions:
            output_attention_file = os.path.join(*[args.output_dir, eval_task_name, "tpr_attention.txt"])
            vals = {}
            for i in range(len(all_ids)):
                vals[all_ids[i]] = {'all_aFs': F_list[i], 'all_aRs': R_list[i]}
            logger.info('saving tpr_attentions to {} '.format(output_attention_file))
            with open(output_attention_file, "w") as fp:
                json.dump(vals, fp)

        output_eval_file = os.path.join(*[args.output_dir, eval_task_name, "eval_results.txt"])
        logger.info("***** Eval results *****")
        logger.info("  eval output file is in {}".format(output_eval_file))
        with open(output_eval_file, "w") as writer:
            writer.write('exp_{:s}_{:.3f}_{:.6f}_{:.0f}_{:.1f}_{:.0f}_{:.0f}_{:.0f}_{:.0f}\n'
                         .format(eval_task_name, args.temperature, args.learning_rate, args.train_batch_size,
                                 args.num_train_epochs, args.dSymbols, args.dRoles, args.nSymbols, args.nRoles))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        test_task_name = all_tasks[-1]
        logger.info('*** Start testing for {} ***'.format(test_task_name))
        processor = PROCESSORS[test_task_name.lower()](args.num_ex)
        num_labels = NUM_LABELS_TASK[test_task_name.lower()]
        task_type = TASK_TYPE[test_task_name.lower()]
        label_list = None
        if task_type != 1:
            label_list = processor.get_labels()

        # Load a trained model for evaluation
        if args.do_train:
            output_model_file = os.path.join(*[args.output_dir, test_task_name, 'pytorch_model_best.bin'])
        else:
            output_model_file = os.path.join(args.load_ckpt)

        states = torch.load(output_model_file, map_location=device)
        model_state_dict = states['state_dict']
        opt = states['options']
        if 'nRoles' not in opt:
            print(args.nRoles)
            for val in ['nRoles', 'nSymbols', 'dRoles', 'dSymbols']:
                opt[val] = getattr(args, val)
        bert_config = states['bert_config']

        if not isinstance(bert_config, PretrainedConfig):
            bert_dict = bert_config.to_dict()
            bert_dict['layer_norm_eps'] = 1e-12
            bert_config = PretrainedConfig.from_dict(bert_dict)

        if 'head.scale' in model_state_dict.keys():
            print('scale value is:', model_state_dict['head.scale'])
        logger.info('*' * 50)
        logger.info('option for evaluation: {}'.format(args))
        logger.info('*' * 50)

        # also print it for philly debugging
        print('option for evaluation: {}'.format(args))
        model = BertForSequenceClassification_tpr(bert_config,
                                                  num_labels=num_labels,
                                                  task_type=task_type,
                                                  temperature=args.temperature,
                                                  max_seq_len=args.max_seq_length,
                                                  **opt)

        model.load_state_dict(model_state_dict, strict=True)
        model.to(device)
        model.eval()

        if args.reset_temp_ratio != 1.0 and hasattr(model.head, 'temperature'):
            new_temp = model.head.temperature / args.reset_temp_ratio
            model.head.temperature = new_temp

        if test_task_name.lower().startswith('dnc'):
            test_examples = processor.get_all_examples(args.data_dir)
            # prepare test data
            for k in test_examples.keys():

                test_dataloader, all_guids = prepare_data_loader(args, processor, label_list, task_type, test_task_name, tokenizer, split='test', examples=test_examples)

                result = predict(args, model, test_dataloader, all_guids, device, task_type)

                if not os.path.exists(os.path.join(args.output_dir, test_task_name)): os.makedirs(os.path.join(args.output_dir, test_task_name))
                output_test_file = os.path.join(*[args.output_dir, test_task_name, "{}-test_predictions.txt".format(k)])
                logger.info("***** Test predictions *****")
                logger.info("  test output file is in {}".format(output_test_file))
                with open(output_test_file, "w") as writer:
                    writer.write("index\tpredictions\n")
                    for id, pred in zip(result['input_ids'], result['predictions']):
                        writer.write("%s\t%s\n" % (id, label_list[pred]))

        else:
            # prepare test data
            test_dataloader, all_guids = prepare_data_loader(args, processor, label_list, task_type, test_task_name, tokenizer, split='test')

            result = predict(args, model, test_dataloader, all_guids, device, task_type)

            if not os.path.exists(os.path.join(args.output_dir, test_task_name)): os.makedirs(os.path.join(args.output_dir, test_task_name))
            output_test_file = os.path.join(*[args.output_dir, test_task_name, "test_predictions.txt"])
            logger.info("***** Test predictions *****")
            logger.info("  test output file is in {}".format(output_test_file))
            with open(output_test_file, "w") as writer:
                writer.write("index\tpredictions\n")
                for id, pred in zip(result['input_ids'], result['predictions']):
                    if test_task_name.lower() == 'hans':
                        if pred == 2: pred = 0  #consider neutral as non-entailment
                        writer.write("%s,%s\n" % (id, label_list[pred]))
                    elif task_type == 1:
                        writer.write("%s\t%s\n" % (id, pred))
                    else:
                        writer.write("%s\t%s\n" % (id, label_list[pred]))
                if test_task_name.lower() == 'snli':
                    writer.write("test_accuracy:\t%s\n" % (result['test_accuracy']))


if __name__ == "__main__":
    args = define_args()
    main(args)
