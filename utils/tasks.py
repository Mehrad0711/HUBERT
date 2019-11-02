from utils.data_utils import *
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertModel, BertTokenizer,
                                  RobertaConfig,
                                  RobertaModel,
                                  RobertaTokenizer,
                                  XLMConfig, XLMModel,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetModel,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertModel,
                                  DistilBertTokenizer)


MODEL_CLASSES = {
    'BERT': (BertConfig, BertModel, BertTokenizer),
    'XLNet': (XLNetConfig, XLNetModel, XLNetTokenizer),
    'XLM': (XLMConfig, XLMModel, XLMTokenizer),
    'RoBERTa': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'DistilBert': (DistilBertConfig, DistilBertModel, DistilBertTokenizer)
}

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ())

from arguments import define_args
args = define_args()

PROCESSORS = {
    'dnc_acc': ACCProcessor,
    'dnc_nli': NLIProcessor,
    'hans': HANSProcessor,
    'mnli': lambda num_ex: MNLIProcessor(num_ex, args.model_type),
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