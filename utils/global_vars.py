
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

POS_TAGS_MAP = {
            'VB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], # verb
            'RB': ['RB', 'RBR', 'RBS', 'WRB', '-LRB-', '-RRB-'], # adverb
            'PRP': ['PRP', 'PRP$', 'WP', 'WP$'], # pronoun
            'NN': ['NN', 'NNS', 'NNP', 'NNPS'], # noun
            'DT': ['DT', 'WDT'], # determiner
            'JJ': ['JJ', 'JJR', 'JJS'], #adjective
            'MD': ['MD'], # modal
            'CC': ['CC', 'IN'], # conjunction
            'CD': ['CD'], # Cardinal number
            'EX': ['EX'], # Existential there
            'FW': ['FW'], #Foreign word
            '$': ['$', '``', "''", 'SYM'], # $ sign, quotation marks, symbols, double colon
            '.': ['.', ':', ','], # punctuation
            'RP': ['RP'], # particle
            'TO': ['TO'], # to
            'UH': ['UH'], # interjection
            'PDT': ['PDT'], # Predeterminer
            'LS': ['LS'], # List item marker
            'POS': ['POS'], # Possessive ending
            'SEP': ['SEP'] # SEP token
            }