import torch

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
