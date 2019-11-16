import torch
import os

def inductive_bias(input):
    """
    Inductive bias to sparsify aRs and aFs. See equation (4) of https://arxiv.org/abs/1705.08432
    """

    input_squared = input.pow(2)
    item1 = input_squared + input_squared.pow(2) - 2 * input * input_squared
    item1 = torch.sum(item1, 2)
    item2 = (torch.sum(input_squared, 2) - 1).pow(2)
    output = torch.sum(item1 + item2, 1)

    return output


def modify_model(model, dev_task, args):

    # load previous task best-model classifier (and possibly other parameters)
    prev_model_state_dict = torch.load(os.path.join(*[args.output_dir, dev_task, "pytorch_model_best.bin"]))['state_dict']
    pre = model.module if hasattr(model, 'module') else model

    replace_keys = ['classifier.weight', 'classifier.bias']
    if args.replace_filler:
        replace_keys.extend(['head.F.weight', 'head.F.bias'])
    if args.replace_role:
        replace_keys.extend(['head.R.weight', 'head.R.bias'])
    if args.replace_filler_selector:
        replace_keys.extend(['head.WaF.weight', 'head.WaF.bias'])
        replace_keys.extend([key for key in prev_model_state_dict.keys() if key.startswith('head.enc_aF')])
    if args.replace_role_selector:
        replace_keys.extend(['head.WaR.weight', 'head.WaR.bias'])
        replace_keys.extend([key for key in prev_model_state_dict.keys() if key.startswith('head.enc_aR')])

    for key in replace_keys:
        eval('pre.' + key).set_(prev_model_state_dict[key])
