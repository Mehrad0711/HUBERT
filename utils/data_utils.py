import logging
import nltk
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
from arguments import define_args
import benepar

nltk.download('punkt')
benepar.download('benepar_en2')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = define_args()

if args.return_POS:
    pos_tagger = StanfordPOSTagger(args.pos_tagger_model, args.pos_tagger_jar)
if args.return_NER:
    ner_tagger = StanfordNERTagger(args.ner_tagger_model, args.ner_tagger_jar)
if args.return_DEP:
    dep_parser = StanfordDependencyParser(args.dep_parser_model, args.dep_parser_jar)
if args.return_CONST:
    const_parser = benepar.Parser("benepar_en2")


def get_constituency_path_to_root(tree, leaf_index):

    parented_tree = nltk.tree.ParentedTree.convert(tree)
    path_to_leaf = parented_tree.leaf_treeposition(leaf_index)
    path_to_leaf_cut = path_to_leaf[:-1]

    current = parented_tree[path_to_leaf_cut]

    labels = []
    while current is not None:
        labels.append(current.label())
        current = current.parent()

    return labels[:-1]

def process_dep(tokens):
    dep_graph = next(dep_parser.parse(tokens))
    triples = list(dep_graph.triples())
    word2parent = {}
    word2rel = {}
    rels = []
    parent_rels = []

    for anchor, rel, word in triples:
        rels.append((word[0], rel))
        word2rel[word] = rel
        word2parent[word] = anchor[0]

    for anchor, rel, word in triples:
        parent = word2parent[word]
        parent_rels.append((word[0], word2rel.get(parent, None)))

    return rels, parent_rels


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


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 pos_tagged_a=None, pos_tagged_b=None, ner_tagged_a=None, ner_tagged_b=None,
                 dep_parsed_a=None, dep_parsed_b=None, dep_parsed_parents_a=None, dep_parsed_parents_b=None,
                 const_parsed_a=None, const_parsed_b=None):
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
        self.pos_tagged_a = pos_tagged_a
        self.pos_tagged_b = pos_tagged_b
        self.ner_tagged_a = ner_tagged_a
        self.ner_tagged_b = ner_tagged_b
        self.dep_parsed_a = dep_parsed_a
        self.dep_parsed_b = dep_parsed_b
        self.dep_parsed_parents_a = dep_parsed_parents_a
        self.dep_parsed_parents_b = dep_parsed_parents_b
        self.const_parsed_a = const_parsed_a
        self.const_parsed_b = const_parsed_b

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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, single_sentence=False,
                                 return_pos_tags=False, return_ner_tags=False, return_dep_parse=False, return_const_parse=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = None
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    token_pos = []
    token_ner = []
    token_dep = []
    token_const = []

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

        ### extract structure information
        if return_pos_tags and not example.pos_tagged_a:
            example.pos_tagged_a = pos_tagger.tag(tokens_a)
            if tokens_b:
                example.pos_tagged_b = pos_tagger.tag(tokens_b)
        if return_ner_tags:
            # for NER tokenize without lower casing the words
            example.ner_tagged_a = ner_tagger.tag(nltk.word_tokenize(example.text_a))
            if tokens_b:
                example.ner_tagged_b = ner_tagger.tag(nltk.word_tokenize(example.text_b))
        if return_dep_parse and not example.dep_parsed_a:
            example.dep_parsed_a, example.dep_parsed_parents_a = process_dep(tokens_a)
            if tokens_b:
                example.dep_parsed_b, example.dep_parsed_parents_b = process_dep(tokens_b)
        if return_const_parse and not example.const_parsed_a:
            example.const_parsed_a = const_parser.parse(tokens_a)
            if tokens_b:
                example.const_parsed_b = const_parser.parse(tokens_b)
        const_parsed_a = []
        leaves_a = example.const_parsed_a.leaves()
        for i in range(len(leaves_a)):
            const_parsed_a.append((leaves_a[i], get_constituency_path_to_root(example.const_parsed_a, i)))
        if example.const_parsed_b:
            const_parsed_b = []
            leaves_b = example.const_parsed_b.leaves()
            for i in range(len(leaves_b)):
                const_parsed_b.append((leaves_b[i], get_constituency_path_to_root(example.const_parsed_b, i)))
        example.const_parsed_a = const_parsed_a
        if const_parsed_b:
            example.const_parsed_b = const_parsed_b


        if return_const_parse:
            const_parsed_a = []
            leaves_a = example.const_parsed_a.leaves()
            for i in range(len(leaves_a)):
                const_parsed_a.append((leaves_a[i], get_constituency_path_to_root(example.const_parsed_a, i)))
            if example.const_parsed_b:
                const_parsed_b = []
                leaves_b = example.const_parsed_b.leaves()
                for i in range(len(leaves_b)):
                    const_parsed_b.append((leaves_b[i], get_constituency_path_to_root(example.const_parsed_b, i)))
            example.const_parsed_a = const_parsed_a
            if const_parsed_b:
                example.const_parsed_b = const_parsed_b


        if example.pos_tagged_a:
            if example.pos_tagged_b and not single_sentence:
                token_pos.append((example.pos_tagged_a, example.pos_tagged_b))
            else:
                token_pos.append((example.pos_tagged_a,))
        if example.ner_tagged_a:
            if example.ner_tagged_b and not single_sentence:
                token_ner.append((example.ner_tagged_a, example.ner_tagged_b))
            else:
                token_ner.append((example.ner_tagged_a,))
        if example.dep_parsed_a:
            if example.dep_parsed_b and not single_sentence:
                token_dep.append((example.dep_parsed_a, example.dep_parsed_b))
            else:
                token_dep.append((example.dep_parsed_q,))
        if example.const_parsed_a:
            if example.const_parsed_b and not single_sentence:
                token_const.append((example.const_parsed_a, example.const_parsed_b))
            else:
                token_const.append((example.const_parsed_a,))

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
            for key in ['label', 'pos_tagged_a', 'pos_tagged_b', 'ner_tagged_a', 'ner_tagged_b',
                        'dep_parsed_a', 'dep_parsed_b', 'const_parsed_a', 'const_parsed_b']:
                if getattr(example, key, None):
                    logger.info("{}: {}".format(key, getattr(example, key)))

        features.append(InputFeatures(guid, input_ids, input_mask, segment_ids, sub_word_masks, orig_to_tok_map, label_id))

    structure_features = (token_pos, token_ner, token_dep, token_const)
    return features, structure_features


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


