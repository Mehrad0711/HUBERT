import logging
import numpy as np
import torch
from tqdm import tqdm
from utils.metrics import class_acc, reg_acc

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def predict(args, model, test_dataloader, all_guids, device, task_type):

    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0

    all_ids = []
    all_predictions = []

    for data in tqdm(test_dataloader, desc="predicting"):
        input_ids, input_mask, segment_ids, sub_word_masks = data[:4]
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        sub_word_masks = sub_word_masks.to(device)

        if args.task_name.lower() == 'snli':
            label_ids = data[4]
            label_ids = label_ids.to(device)

        with torch.no_grad():
            logits, _, _ = model(input_ids, segment_ids, input_mask, sub_word_masks)

        logits = logits.detach().cpu().numpy()

        if task_type != 1:
            predictions = np.argmax(logits, axis=1)
        else:
            predictions = logits
        nb_test_examples += input_ids.size(0)

        if args.task_name.lower() == 'snli':
            label_ids = label_ids.to('cpu').numpy()
            if task_type == 0:
                tmp_test_accuracy = class_acc(logits, label_ids)
            else:
                tmp_test_accuracy = reg_acc(logits, label_ids)
            test_accuracy += tmp_test_accuracy

        nb_test_steps += 1
        all_predictions.extend(predictions.flatten().tolist())

    test_accuracy = test_accuracy / nb_test_examples
    if args.task_name.lower() == 'hans':
        all_ids = all_guids
    else:
        all_ids = list(range(len(all_predictions)))
    result = {'input_ids': all_ids,
              'predictions': all_predictions,
              'test_accuracy': test_accuracy
              }

    return result