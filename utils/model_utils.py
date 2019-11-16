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

    pre.classifier.weight.set_(prev_model_state_dict['classifier.weight'])
    pre.classifier.bias.set_(prev_model_state_dict['classifier.bias'])
    if args.replace_filler:
        pre.head.F.weight.set_(prev_model_state_dict['head.F.weight'])
        pre.head.F.bias.set_(prev_model_state_dict['head.F.bias'])
    if args.replace_role:
        pre.head.R.weight.set_(prev_model_state_dict['head.R.weight'])
        pre.head.R.bias.set_(prev_model_state_dict['head.R.bias'])
    if args.replace_filler_selector:
        filler_selector = [key for key in prev_model_state_dict.keys() if key.startswith('head.enc_aF')]
        for key in filler_selector:
            eval('pre.' + key).set_(prev_model_state_dict[key])
        pre.head.WaF.weight.set_(prev_model_state_dict['head.WaF.weight'])
        pre.head.WaF.bias.set_(prev_model_state_dict['head.WaF.bias'])
    if args.replace_role_selector:
        role_selector = [key for key in prev_model_state_dict.keys() if key.startswith('head.enc_aF')]
        for key in role_selector:
            eval('pre.' + key).set_(prev_model_state_dict[key])
        pre.head.WaR.weight.set_(prev_model_state_dict['head.WaR.weight'])
        pre.head.WaR.bias.set_(prev_model_state_dict['head.WaR.bias'])
