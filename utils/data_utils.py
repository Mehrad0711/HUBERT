import csv
import json
import logging
import os
from collections import defaultdict
import nltk
from nltk.tag import StanfordPOSTagger, StanfordNERTagger
import re
from arguments import define_args

nltk.download('punkt')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = define_args()
pos_tagger = StanfordPOSTagger(args.stanford_pos_model, args.stanford_pos_jar)
ner_tagger = StanfordNERTagger(args.stanford_ner_model, args.stanford_ner_jar)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, parsed_a=None, parsed_b=None, ner_a=None, ner_b=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.parsed_a = parsed_a
        self.parsed_b = parsed_b
        self.ner_a = ner_a
        self.ner_b = ner_b

class MultiInputExample(object):
    """A single training/test example for more complex sequence classification tasks (e,g, SuperGLUE)."""

    def __init__(self, guid, premise, choices, question=None, label=None):
        """Constructs a InputExample.

        Args:
            ...
        """
        self.guid = guid
        self.premise = premise
        self.choices = choices
        self.question = question
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, sub_word_masks, orig_to_token_map, label_id):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sub_word_masks = sub_word_masks
        self.orig_to_token_map = orig_to_token_map
        self.label_id = label_id

class MultiInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, sub_word_masks, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sub_word_masks = sub_word_masks
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, num_ex):
        self.num_ex = num_ex

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
                if len(lines) > self.num_ex:
                    break
            return lines
    def _read_json(self, input_file):
        """Reads a json file."""
        data = []
        with open(input_file, "r", encoding="utf8") as f:
            for row in f:
                line = json.loads(row)
                data.append(line)
                if len(data) > self.num_ex:
                    break
            return data

class NLIProcessor(DataProcessor):
    """Processor for the general NLI data set (GLUE version)."""

    def __init__(self, num_ex):
        super(NLIProcessor, self).__init__(num_ex)
        self.data = defaultdict()
        # self.metadata = defaultdict()

    def get_all_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(*[data_dir, 'function_words', 'NLI'])
        for file in os.listdir(data_dir):
            category = file.rsplit('_', 1)[0][4:]
            if file.rsplit('.', 1)[0].endswith('metadata'):
                # self.metadata[category] = self._create_examples(self._read_json(os.path.join(data_dir, file)), 'train')
                continue
            else:
                self.data[category] = self._create_examples(self._read_json(os.path.join(data_dir, file)), 'train')


    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line['pair-id'])
            text_a = line['context']
            text_b = line['hypothesis']
            if set_type == 'test':
                label = None
            else:
                label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ACCProcessor(DataProcessor):
    """Processor for the ... data set (GLUE version)."""

    def __init__(self, num_ex):
        super(ACCProcessor, self).__init__(num_ex)
        self.data = defaultdict()
        # self.metadata = defaultdict()

    def get_all_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(*[data_dir, 'function_words', 'ACCEPTABILITY'])
        for file in os.listdir(data_dir):
            category = file.rsplit('_', 1)[0][4:]
            if file.rsplit('.', 1)[0].endswith('metadata'):
                # self.metadata[category] = self._create_examples(self._read_json(os.path.join(data_dir, file)), 'train')
                continue
            else:
                self.data[category] = self._create_examples(self._read_json(os.path.join(data_dir, file)), 'train')
        return self.data

    def get_labels(self):
        """See base class."""
        return ["unacceptable", "acceptable"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line['pair-id'])
            text_a = line['context']
            if line['hypothesis'].strip() == '':
                text_b = None
            else:
                text_b = line['hypothesis']
            if set_type == 'test':
                label = None
            else:
                label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SNLIProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def __init__(self, num_ex):
        super(SNLIProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            if set_type == 'test':
                label = line[-1]
            else:
                label = line[-1]

            text_a_parsed = re.findall(r'\([^\)\(]*\)', line[5])
            text_a_parsed = [tuple([item.split()[1][:-1], item.split()[0][1:]]) for item in text_a_parsed]
            text_b_parsed = re.findall(r'\([^\)\(]*\)', line[6])
            text_b_parsed = [tuple([item.split()[1][:-1], item.split()[0][1:]]) for item in text_b_parsed]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, parsed_a=text_a_parsed, parsed_b=text_b_parsed))
        return examples

class HANSProcessor(DataProcessor):
    """Processor for the HANS data set (GLUE version)."""

    def __init__(self, num_ex):
        super(HANSProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "heuristics_evaluation_set.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "heuristics_evaluation_set.txt")), "eval")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "heuristics_evaluation_set.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["non-entailment", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = str(line[7])
            text_a = line[5]
            text_b = line[6]
            if set_type == 'test':
                label = None
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MNLIProcessor(DataProcessor):
    """Processor for the MNLI data set (GLUE version)."""

    def __init__(self, num_ex):
        super(MNLIProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # get_transitions = lambda parse: ['reduce' if t == ')' else 'shift' for t in parse if t != '(']
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type == 'test':
                label = None
            else:
                label = line[-1]

            text_a_parsed = re.findall(r'\([^\)\(]*\)', line[6])
            text_a_parsed = [tuple([item.split()[1][:-1], item.split()[0][1:]]) for item in text_a_parsed]
            text_b_parsed = re.findall(r'\([^\)\(]*\)', line[7])
            text_b_parsed = [tuple([item.split()[1][:-1], item.split()[0][1:]]) for item in text_b_parsed]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, parsed_a=text_a_parsed, parsed_b=text_b_parsed))
        return examples


class MRPCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, num_ex):
        super(MRPCProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-2]
            text_b = line[-1]
            if set_type == 'test':
                label = None
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class QNLIProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def __init__(self, num_ex):
        super(QNLIProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type == 'test':
                label = None
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QQPProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def __init__(self, num_ex):
        super(QQPProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == 'test':
                text_a = line[-2]
                text_b = line[-1]
                label = None
            else:
                if len(line) < 6:
                    continue
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class RTEProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def __init__(self, num_ex):
        super(RTEProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == 'test':
                text_a = line[-2]
                text_b = line[-1]
                label = None
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class WNLIProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def __init__(self, num_ex):
        super(WNLIProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == 'test':
                text_a = line[-2]
                text_b = line[-1]
                label = None
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SSTProcessor(DataProcessor):
    """Processor for the SST data set (GLUE version)."""

    def __init__(self, num_ex):
        super(SSTProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            if set_type == 'test':
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                label = None
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            else:
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class STSProcessor(DataProcessor):
    """Processor for the STS data set (GLUE version)."""

    def __init__(self, num_ex):
        super(STSProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        print('Any number between 0 and 5')
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == 'test':
                text_a = line[-2]
                text_b = line[-1]
                label = None
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class COLAProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, num_ex):
        super(COLAProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            if set_type == 'test':
                guid = "%s-%s" % (set_type, i)
                text_a = line[-1]
                label = None
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            else:
                guid = "%s-%s" % (set_type, i)
                text_a = line[-1]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class COPAProcessor(DataProcessor):
    """Processor for the COPA data set (SuperGLUE version)."""

    def __init__(self, num_ex):
        super(COPAProcessor, self).__init__(num_ex)

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                MultiInputExample(guid=guid, premise=line['premise'], choices=['{} || {}'.format(line['question'], line['choice1']),
                                                                               '{} || {}'.format(line['question'], line['choice2'])],
                                  question=line['question'], label=str(line['label'])))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 single_sentence=False, return_pos_tags=False, return_ner_tags=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = None
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    token_pos = []
    token_ner = []
    for (ex_index, example) in enumerate(examples):

        do_lower = tokenizer.basic_tokenizer.do_lower_case
        tokens_b = None

        tokens_a = nltk.word_tokenize(example.text_a.lower() if do_lower else example.text_a)
        if example.text_b and not single_sentence:
            tokens_b = nltk.word_tokenize(example.text_b.lower() if do_lower else example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        if tokens_b:
            raw_tokens = tokens_a + ['[SEP]'] + tokens_b
        else:
            raw_tokens = tokens_a

        orig_to_tok_map = []
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        orig_to_tok_map.append(len(tokens)-1)
        for token in tokens_a:
            tokenized_tokens = tokenizer.tokenize(token)
            if len(tokens) + len(tokenized_tokens) >= max_seq_length - 1:
                break
            tokens.extend(tokenized_tokens)
            orig_to_tok_map.append(len(tokens)-1)
            segment_ids.extend([0]*len(tokenized_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)
        orig_to_tok_map.append(len(tokens)-1)

        if tokens_b and len(tokens) < max_seq_length - 1:
            for token in tokens_b:
                tokenized_tokens = tokenizer.tokenize(token)
                if len(tokens) + len(tokenized_tokens) >= max_seq_length - 1:
                    break
                tokens.extend(tokenized_tokens)
                orig_to_tok_map.append(len(tokens)-1)
                segment_ids.extend([1]*len(tokenized_tokens))
            if tokens[-1] != '[SEP]' and len(tokens) <= max_seq_length - 1:
                tokens.append("[SEP]")
                segment_ids.append(1)
                orig_to_tok_map.append(len(tokens)-1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # mask to keep track of the beginning of each sub-word piece
        sub_word_masks = [0 if t.startswith('##') else 1 for t in tokens]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == len(segment_ids) == len(input_mask) == len(sub_word_masks)

        # pad up or cut to the sequence length
        zero_padding = [0] * (max_seq_length - len(input_ids))
        one_padding = [1] * (max_seq_length - len(input_ids))
        minus_one_padding = [-1] * (max_seq_length - len(orig_to_tok_map))
        input_ids += zero_padding
        input_mask += zero_padding
        segment_ids += zero_padding
        sub_word_masks += one_padding
        orig_to_tok_map += minus_one_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(sub_word_masks) == max_seq_length
        assert len(orig_to_tok_map) == max_seq_length

        if label_map:
            if example.label:
                label_id = label_map[example.label]
            else:
                label_id = None
        else:
            if example.label:
                label_id = float(example.label)
            else:
                label_id = None

        return_pos_tags = return_pos_tags
        return_ner_tags = return_ner_tags

        if return_pos_tags and not example.parsed_a:
            example.parsed_a = pos_tagger.tag(tokens_a)
            if tokens_b:
                example.parsed_b = pos_tagger.tag(tokens_b)
        if return_ner_tags:
            example.ner_a = ner_tagger.tag(nltk.word_tokenize(example.text_a))
            if tokens_b:
                example.ner_b = ner_tagger.tag(nltk.word_tokenize(example.text_b))

        if example.parsed_a:
            if example.parsed_b and not single_sentence:
                token_pos.append((example.parsed_a, example.parsed_b))
            else:
                token_pos.append((example.parsed_a,))

        if example.ner_a:
            if example.ner_b and not single_sentence:
                token_ner.append((example.ner_a, example.ner_b))
            else:
                token_ner.append((example.ner_a,))

        guid = example.guid
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            for key in ['label', 'parsed_a', 'parsed_b', 'ner_a', 'ner_b']:
                if getattr(example, key, None):
                    logger.info("{}: {}".format(key, getattr(example, key)))

        features.append(InputFeatures(guid, input_ids, input_mask, segment_ids, sub_word_masks, orig_to_tok_map, label_id))

    return features, token_pos, token_ner


def convert_multi_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = None
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        premise = tokenizer.tokenize(example.premise)

        size = len(example.choices)
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        sub_word_masks_list = []
        label_id_list = []

        for j in range(size):
            choice = tokenizer.tokenize(example.choices[j])
            _truncate_seq_pair(premise, choice, max_seq_length - 3)

            tokens = ["[CLS]"] + premise + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            tokens += choice + ["[SEP]"]
            segment_ids += [1] * (len(choice) + 1)

            # mask to keep track of the beginning of each sub-word piece
            sub_word_masks = [0 if t.startswith('##') else 1 for t in tokens]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            zero_padding = [0] * (max_seq_length - len(input_ids))
            one_padding = [1] * (max_seq_length - len(input_ids))
            input_ids += zero_padding
            input_mask += zero_padding
            segment_ids += zero_padding
            sub_word_masks += one_padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(sub_word_masks) == max_seq_length

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            sub_word_masks_list.append(sub_word_masks)

            if label_map:
                if example.label:
                    label_id = label_map[example.label]
                else:
                    label_id = None
            else:
                if example.label:
                    label_id = float(example.label)
                else:
                    label_id = None

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if example.lable is not None:
                    logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            MultiInputFeatures(input_ids=input_ids_list,
                          input_mask=input_mask_list,
                          segment_ids=segment_ids_list,
                          sub_word_masks=sub_word_masks_list,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
