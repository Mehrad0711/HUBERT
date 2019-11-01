import torch
from tqdm import tqdm
import logging
from utils.metrics import class_acc, reg_acc

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_dataloader, device, task_type, global_step=None, tr_loss=None, nb_tr_steps=None):

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, sub_word_masks, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        sub_word_masks = sub_word_masks.to(device)
        label_ids = label_ids.to(device)
        if args.model_type != 'DistilBERT':
            segment_ids = segment_ids if args.model_type in ['BERT', 'XLNet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

        with torch.no_grad():
            logits, tmp_eval_loss = model(input_ids, segment_ids, input_mask, sub_word_masks, label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        if task_type == 0:
            tmp_eval_accuracy = class_acc(logits, label_ids)
        else:
            tmp_eval_accuracy = reg_acc(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    loss = tr_loss / nb_tr_steps if tr_loss else None
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': global_step if args.do_train else 0,
              'loss': loss}

    return result