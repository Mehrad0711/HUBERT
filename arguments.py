import argparse
import os

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

    ####################
    ## Required arguments
    ####################
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The input data dir. should contain folders with task names having the .tsv files (or other data files) for that task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
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

    ## Other arguments
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

    parser.add_argument("--delete_ok",
                        type=str2bool,
                        default=False,
                        help='whether to delete the result directory if it already exists')


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




    ####################
    ## Model arguments
    ####################

    ####################
    ## TPR arguments
    ####################

    # model
    parser.add_argument("--nSymbols",
                        default=50,
                        type=int,
                        help="# of symbols")
    parser.add_argument("--nRoles",
                        default=35,
                        type=int,
                        help="# of roles")
    parser.add_argument("--dSymbols",
                        default=32,
                        type=int,
                        help="embedding size of symbols")
    parser.add_argument("--dRoles",
                        default=32,
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

    # model parameters to load during fine-tuning
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
    parser.add_argument("--load_bert_params",
                        type=str2bool,
                        default=False,
                        help="Load bert parameters from checkpoint")
    parser.add_argument("--load_role_selector",
                        type=str2bool,
                        default=False,
                        help="Load Role selector neural network from checkpoint")
    parser.add_argument("--load_filler_selector",
                        type=str2bool,
                        default=False,
                        help="Load Filler selector neural network from checkpoint")
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


    # model parameters to freeze during fine-tuning
    parser.add_argument("--freeze_role",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Role matrices after loading from a trained model')
    parser.add_argument("--freeze_filler",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Filler matrices after loading from a trained model')
    parser.add_argument("--freeze_role_selector",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Role networks after loading from a trained model')
    parser.add_argument("--freeze_filler_selector",
                        type=str2bool,
                        default=False,
                        help='whether to freeze Filler networks after loading from a trained model')
    parser.add_argument("--freeze_bert_params",
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

    # BERT model params
    parser.add_argument("--num_bert_layers",
                        type=int,
                        default=12,
                        help='num_bert_layers to use in our model')
    parser.add_argument("--num_rnn_layers",
                        type=int,
                        default=1,
                        help='number of layers for recurrent network')
    parser.add_argument("--num_extra_layers",
                        type=int,
                        default=0,
                        help='number of extra transformer layers for tpr_transformer network')
    parser.add_argument("--num_heads",
                        type=int,
                        default=8,
                        help='number of head for transformer tpr network')

    ####################
    ## optimizer arguments
    ####################
    parser.add_argument("--optimizer",
                        type=str,
                        default='adam',
                        choices=['adam', 'radam', 'sgd'],
                        help='whether to freeze Role/ Filler matrices after loading from a trained model')

    ####################
    ## training arguments
    ####################
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
    parser.add_argument("--patience",
                        default=6,
                        type=int,
                        help="Number of epochs to allow no further accuracy improvement.")
    parser.add_argument("--tolerance",
                        default=0.0005,
                        type=float,
                        help="Margin for accuracy improvement")
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


    parser.add_argument("--debug",
                        type=str2bool,
                        default=False,
                        help='turn on debug mode')
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
    parser.add_argument("--freeze_bert",
                        type=str2bool,
                        default=False,
                        help='Freeze bert layers')
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
                        help='regularization coefficient for orthonormal R matrix loss')
    parser.add_argument("--inductive_reg",
                        type=float,
                        default=0.0,
                        help='regularization coefficient for inductive loss')
    parser.add_argument("--cls",
                        type=str,
                        default='v1',
                        help='which classifier to use')

    ####################
    ## dataset util parameters
    ####################
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--num_ex",
                        type=float,
                        default=2000000000,
                        help='number of examples to choose of train/dev/test dataset')




    # model parameters to replace in continual learning setting
    parser.add_argument("--replace_filler",
                        type=str2bool,
                        default=False,
                        help='when evaluating a model on previous task replace filler weights with previous values')
    parser.add_argument("--replace_role",
                        type=str2bool,
                        default=False,
                        help='when evaluating a model on previous task replace role weights with previous values')
    parser.add_argument("--replace_filler_selector",
                        type=str2bool,
                        default=False,
                        help='when evaluating a model on previous task replace filler selector weights with previous values')
    parser.add_argument("--replace_role_selector",
                        type=str2bool,
                        default=False,
                        help='when evaluating a model on previous task replace role selector weights with previous values')


    parser.add_argument("--reset_temp_ratio",
                        type=float,
                        default=1.0,
                        help='set temperature to a smaller value during evaluation and testing')


    ####################
    ## aF and aR attention arguments
    ####################
    parser.add_argument("--save_tpr_attentions",
                        type=str2bool,
                        default=False,
                        help='save aFs and aRs')
    parser.add_argument("--save_strategy",
                        type=str,
                        default='topK',
                        help='method to retrieve tpr attention values',
                        choices=['topK', 'sample', 'selectK', 'full'])
    parser.add_argument("--data_split_attention",
                        type=str,
                        default='dev',
                        choices=['train', 'dev', 'test'],
                        help='which split of data to choose for saving aFs and aRs attentions')
    parser.add_argument("--single_sentence",
                        type=str2bool,
                        default=False,
                        help='omit hypothesis for paired-input tasks')
    parser.add_argument("--K",
                        type=int,
                        default=1,
                        help='choose K biggest value from tpr attentions')

    ####################
    ## POS and NER
    ####################
    parser.add_argument("--return_POS",
                        type=str2bool,
                        default=False,
                        help='return POS tags for tokens in the input data')
    parser.add_argument("--return_NER",
                        type=str2bool,
                        default=False,
                        help='return NER for tokens in the input data')
    parser.add_argument("--return_DEP",
                        type=str2bool,
                        default=False,
                        help='return dependency-tree edge for tokens in the input data')
    parser.add_argument("--return_CONST",
                        type=str2bool,
                        default=False,
                        help='return constituency parse paths for tokens in the input data')
    parser.add_argument("--pos_tagger_jar",
                        default='./tests/parser_files/stanford-postagger-3.9.2.jar',
                        help='path to stanford jar file for Stanford POS tagger')
    parser.add_argument("--pos_tagger_model",
                        default='./tests/parser_files/english-bidirectional-distsim.tagger',
                        help='path to stanford model file for Stanford POS tagger')
    parser.add_argument("--ner_tagger_jar",
                        default='./tests/parser_files/stanford-ner-3.9.2.jar',
                        help='path to stanford jar file for Stanford POS tagger')
    parser.add_argument("--ner_tagger_model",
                        default='./tests/parser_files/english.muc.7class.distsim.crf.ser.gz',
                        help='path to stanford model file for Stanford POS tagger')
    parser.add_argument("--dep_parser_jar",
                        default='./tests/parser_files/stanford-parser.jar',
                        help='path to stanford jar file for Stanford POS tagger')
    parser.add_argument("--dep_parser_model",
                        default='./tests/parser_files/stanford-parser-3.9.2-models.jar',
                        help='path to stanford model file for Stanford POS tagger')

    ####################
    ## T-SNE and K-means
    ####################
    parser.add_argument("--do_tsne",
                        type=str2bool,
                        default=False,
                        help='Perform T-SNE on role/ filler vectors')
    parser.add_argument("--metric",
                        type=str,
                        default='euclidean',
                        help='T-SNE distance metric')
    parser.add_argument("--tsne_label",
                        type=str,
                        default='pos',
                        choices=['pos', 'ner', 'dep', 'tree', 'const'],
                        help='T-SNE role label used to color the points')
    parser.add_argument("--perplexity",
                        type=int,
                        default=4,
                        help='T-SNE Number of threads')
    parser.add_argument("--n_jobs",
                        type=int,
                        default=4,
                        help='T-SNE Number of threads')
    parser.add_argument("--n_components",
                        type=int,
                        default=2,
                        help='T-SNE dimensionality')
    parser.add_argument("--n_iter",
                        type=int,
                        default=1000,
                        help='T-SNE number of iterations')
    parser.add_argument("--do_Kmeans",
                        type=str2bool,
                        default=False,
                        help='Perform clustering on T-SNE projections')
    parser.add_argument("--n_clusters",
                        type=int,
                        default=10,
                        help='K-means number of clusters')



    args = parser.parse_args()

    return args


