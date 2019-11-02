import argparse
import os
from utils.tasks import MODEL_CLASSES, ALL_MODELS


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unable to parse the argument')


def define_args():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The input data dir. should contain folders with task names having the .tsv files (or other data files) for that task.")

    parser.add_argument("--pretrained_model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=os.path.join(os.getenv("PT_OUTPUT_DIR", ""), 'results'),
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Model
    parser.add_argument("--model_type",
                        type=str,
                        required=True,
                        choices=['BERT', 'RoBERTa', 'XLNet', 'XLM', 'DistilBert'],
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    ## Other parameters
    parser.add_argument("--do_prev_eval",
                        type=str2bool,
                        default=False,
                        help="Whether to eval on previous tasks when doing continual learning")
    parser.add_argument("--cont_task_names",
                        default=[],
                        type=str,
                        nargs='+',
                        help="The name of the tasks to continue training on (for continual learning).")
    parser.add_argument("--eval_task_names",
                        default=[],
                        type=str,
                        nargs='+',
                        help="Task to evaluate your models on")
    parser.add_argument("--test_task_names",
                        default=[],
                        type=str,
                        nargs='+',
                        help="Task to test your models on")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        type=str2bool,
                        default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        type=str2bool,
                        default=False,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        type=str2bool,
                        default=False,
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--do_lower_case",
                        type=str2bool,
                        default=False,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--log_every",
                        default=100,
                        type=int,
                        help="Log results to tensorboard after this many global steps")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        type=str2bool,
                        default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.0,
                        help="clipping grad norms at this value")
    parser.add_argument('--fp16',
                        type=str2bool,
                        default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--nSymbols",
                        default=50,
                        type=int,
                        help="# of symbols")
    parser.add_argument("--nRoles",
                        default=35,
                        type=int,
                        help="# of roles")
    parser.add_argument("--dSymbols",
                        default=30,
                        type=int,
                        help="embedding size of symbols")
    parser.add_argument("--dRoles",
                        default=30,
                        type=int,
                        help="embedding size of roles")
    parser.add_argument("--temperature",
                        default=1.0,
                        type=float,
                        help="softmax temperature for aF and aR")
    parser.add_argument("--sub_word_masking",
                        type=str2bool,
                        default=False,
                        help="whether to feed in only the first sub-word to TPRN (as suggested in BERT paper for NER)")
    parser.add_argument("--bidirect",
                        type=str2bool,
                        default=False,
                        help="whether to use a bidirectional encoder")
    parser.add_argument("--encoder",
                        type=str,
                        default='tpr_transformers',
                        choices=['no_enc', 'lstm', 'tpr_lstm', 'tpr_transformers'],
                        help="which encoder to use")
    parser.add_argument("--load_ckpt",
                        default='',
                        help="path to checkpoint model to load Symbol and Filler matrices")
    parser.add_argument("--load_role",
                        type=str2bool,
                        default=False,
                        help="Load Role matrices from checkpoint")
    parser.add_argument("--load_filler",
                        type=str2bool,
                        default=False,
                        help="Load Filler matrices from checkpoint")
    parser.add_argument("--load_transformer_params",
                        type=str2bool,
                        default=False,
                        help="Load transformer parameters from checkpoint")
    parser.add_argument("--load_classifier",
                        type=str2bool,
                        default=False,
                        help="Load classifier parameters from checkpoint")
    parser.add_argument("--load_LSTM_params",
                        type=str2bool,
                        default=False,
                        help='load LSTM weight and biases')
    parser.add_argument("--fixed_Role",
                        type=str2bool,
                        default=False,
                        help='use identity matrix for Role emebddings instead of learning them')
    parser.add_argument("--delete_ok",
                        type=str2bool,
                        default=False,
                        help='whether to delete the result directory if it already exists')
    parser.add_argument("--freeze_role",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Role/ Filler matrices after loading from a trained model')
    parser.add_argument("--freeze_filler",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Role/ Filler matrices after loading from a trained model')
    parser.add_argument("--freeze_transformer_params",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Role/ Filler matrices after loading from a trained model')
    parser.add_argument("--freeze_classifier",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Role/ Filler matrices after loading from a trained model')
    parser.add_argument("--freeze_LSTM_params",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Role/ Filler matrices after loading from a trained model')
    parser.add_argument("--freeze_mat",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Role/ Filler matrices after loading from a trained model')
    parser.add_argument("--optimizer",
                        type=str,
                        default='adam',
                        choices=['adam', 'radam', 'sgd'],
                        help='whether to freeze Role/ Filler matrices after loading from a trained model')
    parser.add_argument("--num_ex",
                        type=float,
                        default=2000000000,
                        help='number of examples to choose of train/dev/test dataset')
    parser.add_argument("--debug",
                        type=str2bool,
                        default=False,
                        help='debug mode')
    parser.add_argument("--scale_val",
                        type=float,
                        default=1.0,
                        help='initial value of scale factor')
    parser.add_argument("--train_scale",
                        type=str2bool,
                        default=False,
                        help='whether scale factor should be trainable')
    parser.add_argument("--aggregate",
                        type=str,
                        default='none',
                        choices=['sum', 'mean', 'concat', 'none'],
                        help='type of aggregation to perform on top of head layer')
    parser.add_argument("--freeze_transformer",
                        type=str2bool,
                        default=False,
                        help='Freeze transformer layers')
    parser.add_argument("--num_transformer_layers",
                        type=int,
                        default=12,
                        help='num_transformer_layers to use in our model')
    parser.add_argument("--num_rnn_layers",
                        type=int,
                        default=1,
                        help='number of layers for recurrent network')
    parser.add_argument("--num_heads",
                        type=int,
                        default=8,
                        help='number of head for transformer tpr network')
    parser.add_argument("--do_src_mask",
                        type=str2bool,
                        default=True,
                        help='whether to mask the source sentences before feeding to transformer tpr')
    parser.add_argument("--mode",
                        type=str,
                        default='exp',
                        choices=['exp', 'lin'],
                        help='decay method for scaling value')
    parser.add_argument("--final_ratio",
                        type=float,
                        default=1e-2,
                        help='final ratio to decay scale_value to (e.g. final_scale_val = 0.01 * initial_scale_val)')
    parser.add_argument("--do_decay",
                        type=str2bool,
                        default=False,
                        help='decay scale values')
    parser.add_argument("--save_best_only",
                        type=str2bool,
                        default=True,
                        help='save only best checkpoint')
    parser.add_argument("--ortho_reg",
                        type=float,
                        default=0.0,
                        help='regulation for orthonormal R matrix')
    parser.add_argument("--cls",
                        type=str,
                        default='v1',
                        help='which classifier to use')
    parser.add_argument("--replace_filler",
                        type=str2bool,
                        default=False,
                        help='when evaluating a model on previous task (in continual learning settings),'
                             ' replace filler weights with previous values')
    parser.add_argument("--replace_role",
                        type=str2bool,
                        default=False,
                        help='when evaluating a model on previous task (in continual learning settings),'
                             ' replace role weights with previous values')


    args = parser.parse_args()

    return args
