import os
import argparse
import json
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unable to parse the argument')

def run(args):

    tag2role = defaultdict(list)
    role2tag = defaultdict(list)

    with open(args.input_file) as fin:
        lines = json.load(fin)

    for id, line in lines.items():
        tags = line['tags']
        all_aRs = line['all_aRs']

        for tag, role in zip(tags, all_aRs):
            tag2role[str(tag)].append(str(role))
            role2tag[str(role)].append(str(tag))

    tag2role_mostcommon = defaultdict()
    role2tag_mostcommon = defaultdict()

    for tag, role in tag2role.items():
        tag2role[tag] = Counter(role)
        tag2role_mostcommon[tag] = tag2role[tag].most_common(1)

    for role, tag in role2tag.items():
        role2tag[role] = Counter(tag)
        role2tag_mostcommon[role] = role2tag[role].most_common(1)

    # merge close tags
    tags_map = {
        'VB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], # verb
                'RB': ['RB', 'RBR', 'RBS', 'WRB'], # adverb
                'PRP': ['PRP', 'PRP$', 'WP', 'WP$'], # pronoun
                'NN': ['NN', 'NNS', 'NNP', 'NNPS'], # noun
                'DT': ['DT', 'WDT'], # determiner
                'JJ': ['JJ', 'JJR', 'JJS'], #adjective
                'MD': ['MD'], # modal
                'CC': ['CC', 'IN'], # conjunction
                'CD': ['CD'], # Cardinal number
                'EX': ['EX'], # Existential there
                'FW': ['FW'], #Foreign word
                '$': ['$'], # $ sign
                '.': ['.'], # punctuation
                '``': ['``', "''"], # quotation marks
                'SYM': ['SYM'], # symbols
                'RP': ['RP'], # particle
                'TO': ['TO'], # to
                'UH': ['UH'], # interjection
                'PDT': ['PDT'], # Predeterminer
                'LS': ['LS'], # List item marker
                'POS': ['POS'] # Possessive ending
                }

    prev_tag2role = tag2role.copy()
    tag2role = defaultdict(Counter)
    for k, v in tags_map.items():
        for val in v:
            tag2role[k].update(prev_tag2role[val])

    prev_tag2role = tag2role.copy()
    tag2role = defaultdict(Counter)
    for tag, role in prev_tag2role.items():
        summ = sum(role.values())
        if summ != 0:
            tag2role[tag] = Counter({k: float(v)/summ for k, v in role.items()})


    num_roles = len(role2tag.keys())
    num_tags = len(tag2role.keys())

    # color the bars
    ROLES = list(role2tag.keys())
    TAGS = list(tag2role.keys())


    # Values of each group
    all_bars = [[None]*num_tags for _ in range(num_roles)]

    for i, role in enumerate(ROLES):
        for j, tag in enumerate(TAGS):
            all_bars[i][j] = tag2role[tag][role]

    # Heights
    all_bars = np.array(all_bars)
    all_bars_cumsum = np.cumsum(all_bars, axis=0).tolist()

    # Names of group and bar width
    barWidth = 0.8
    # pal = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
    pal = sns.color_palette("Set1", n_colors=num_roles)

    for i in range(len(all_bars)):
        if i==0:
            bottom = None
        else:
            bottom = all_bars_cumsum[i-1]
        top = all_bars[i]

        plt.bar(range(num_tags), height=top, bottom=bottom, color=pal[i], edgecolor='white', width=barWidth)

    # Custom X axis
    plt.xticks(range(num_tags), TAGS, fontsize=7.5, rotation='vertical')

    plt.xlabel("POS Tags")
    plt.ylabel("Role Frequency")

    # ax = plt.gca()
    # ax.legend(ROLES, fancybox=True, frameon=False, loc='lower center', ncol=5)

    # Show graphic
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('./pos_plot.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default='./results/MNLI/tpr_attention.txt', type=str)
    args = parser.parse_args()

    if not os.path.exists('./workdir/log_eval/'):
        os.makedirs('./workdir/log_eval/', exist_ok=True)

    run(args)