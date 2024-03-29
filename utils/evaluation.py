import torch
from tqdm import tqdm
import logging
from utils.metrics import class_acc, reg_acc
from utils.visual_utils import get_attention

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_dataloader, device, task_type, global_step=None, tr_loss=None, nb_tr_steps=None, data_split='dev', save_tpr_attentions=False):

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    F_list, R_list = [], []
    F_full, R_full = [], []

    for input_ids, input_mask, segment_ids, sub_word_masks, orig_to_token_maps, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        sub_word_masks = sub_word_masks.to(device)
        orig_to_token_maps = orig_to_token_maps.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits, tmp_eval_loss, (aFs, aRs) = model(input_ids, segment_ids, input_mask, sub_word_masks, label_ids)

        if save_tpr_attentions:
            F_list, R_list, F_full, R_full = get_attention(args, aFs, aRs, orig_to_token_maps)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()
        if task_type == 0:
            tmp_eval_accuracy = class_acc(logits, label_ids)
        else:
            tmp_eval_accuracy = reg_acc(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    all_ids = [data_split + '_' + str(i) for i in range(logits.shape[0])]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    loss = tr_loss / nb_tr_steps if tr_loss else None
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': global_step if args.do_train else 0,
              'loss': loss}

    return result, (all_ids, F_list, R_list, F_full, R_full)
