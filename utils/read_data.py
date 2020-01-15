import os
import csv
import json
import re
import nltk
from collections import defaultdict
from utils.data_utils import InputExample, MultiInputExample
from utils.data_utils import logger


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

            const_parsed_a = nltk.Tree.fromstring(line[5])
            const_parsed_b = nltk.Tree.fromstring(line[6])

            pos_tagged_a = re.findall(r'\([^\)\(]*\)', line[5])
            pos_tagged_a = [tuple([item.split()[1][:-1], item.split()[0][1:]]) for item in pos_tagged_a]
            pos_tagged_b = re.findall(r'\([^\)\(]*\)', line[6])
            pos_tagged_b = [tuple([item.split()[1][:-1], item.split()[0][1:]]) for item in pos_tagged_b]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             pos_tagged_a=pos_tagged_a, pos_tagged_b=pos_tagged_b, const_parsed_a=const_parsed_a, const_parsed_b=const_parsed_b))
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

            const_parsed_a = nltk.Tree.fromstring(line[6])
            const_parsed_b = nltk.Tree.fromstring(line[7])

            pos_tagged_a = re.findall(r'\([^\)\(]*\)', line[6])
            pos_tagged_a = [tuple([item.split()[1][:-1], item.split()[0][1:]]) for item in pos_tagged_a]
            pos_tagged_b = re.findall(r'\([^\)\(]*\)', line[7])
            pos_tagged_b = [tuple([item.split()[1][:-1], item.split()[0][1:]]) for item in pos_tagged_b]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             pos_tagged_a=pos_tagged_a, pos_tagged_b=pos_tagged_b, const_parsed_a=const_parsed_a, const_parsed_b=const_parsed_b))
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