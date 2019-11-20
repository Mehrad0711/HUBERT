from hubert.utils.data_utils import *

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