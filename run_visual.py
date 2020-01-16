import os
import argparse
import json
import logging
from collections import defaultdict, Counter, OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from utils.global_vars import POS_TAGS_MAP

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unable to parse the argument')

def run(args, tag_type, target_tag):

    role2token_targeted = defaultdict(list)

    tag2role = defaultdict(list)
    role2tag = defaultdict(list)

    with open(args.input_file) as fin:
        lines = json.load(fin)

    if tag_type == 'const_parse_path':
        tag2idx = defaultdict(int)

    for id, line in lines.items():
        try:
            tags = line[tag_type]
        except:
            logger.error('Specified file does not contain {} information'.format(tag_type))
            logger.error('Skipping this tag and continuing the process...')
            break
        all_aRs = line['all_aRs']
        if tag_type == 'dep_edge':
            dep_tokens = line['dep_parse_tokens']
            orig_tokens = line['tokens']

            # find mapping between dep and orig tokens
            indices = []
            for tok in dep_tokens:
                indices.append(orig_tokens.index(tok))
            all_aRs = [all_aRs[i] for i in indices]
            tokens = dep_tokens
        else:
            tokens = line['tokens']

        assert len(tags) == len(all_aRs) == len(tokens)
        for tag, role, token in zip(tags, all_aRs, tokens):
            if tag == target_tag:
                role2token_targeted[str(role)].append(token)
            if tag == '[SEP]':
                continue
            if tag_type == 'const_parse_path':
                tag = tuple(tag)
                if tag not in tag2idx.keys():
                    tag2idx[tag] = len(tag2idx.keys())
                tag2role[str(tag2idx[tag])].append(str(role))
                role2tag[str(role)].append(str(tag2idx[tag]))
            else:
                tag2role[str(tag)].append(str(role))
                role2tag[str(role)].append(str(tag))


    tag2role_mostcommon = defaultdict()
    role2tag_mostcommon = defaultdict()

    for tag, role in tag2role.items():
        tag2role[tag] = Counter(role)
        tag2role_mostcommon[tag] = tag2role[tag].most_common(1)[0][0]

    for role, tag in role2tag.items():
        role2tag[role] = Counter(tag)
        role2tag_mostcommon[role] = role2tag[role].most_common(1)[0][0]

    # merge close tags
    if args.merge_tags and tag_type == 'pos_tags':
        prev_tag2role = tag2role.copy()
        tag2role = defaultdict(Counter)
        for k, v in POS_TAGS_MAP.items():
            for val in v:
                tag2role[k].update(prev_tag2role[val])

    prev_tag2role = tag2role.copy()
    tag2role = defaultdict(Counter)
    tag2sum = dict()
    for tag, role in prev_tag2role.items():
        summ = sum(role.values())
        tag2sum[tag] = summ
        if summ != 0:
            if args.normalize:
                tag2role[tag] = Counter({k: float(v)/summ for k, v in role.items()})
            else:
                tag2role[tag] = Counter({k: float(v) for k, v in role.items()})

    if args.prune:
        prev_tag2role = tag2role.copy()
        tag2role = defaultdict(Counter)
        for tag, role in prev_tag2role.items():
            tag2role[tag] = Counter({k: v for k, v in role.items() if v > args.threshold})

    num_roles = len(role2tag.keys())
    num_tags = len(tag2role.keys())

    # color the bars
    ROLES = list(role2tag.keys())
    TAGS = list(tag2role.keys())

    # Values of each group
    all_bars = [[None]*num_tags for _ in range(num_roles)]

    TAGS_sorted = sorted(TAGS, key=lambda t: tag2sum[t], reverse=True)
    ROLES_sorted_index = sorted(range(num_roles), key=lambda i: sum([tag2role[t][ROLES[i]] for t in TAGS_sorted]), reverse=True)
    ROLES_sorted = [ROLES[i] for i in ROLES_sorted_index]

    for i, role in enumerate(ROLES_sorted):
        for j, tag in enumerate(TAGS_sorted):
            all_bars[i][j] = tag2role[tag][role]

    # Heights
    all_bars = np.array(all_bars)
    all_bars_cumsum = np.cumsum(all_bars, axis=0).tolist()

    # Names of group and bar width
    barWidth = 0.8
    pal = sns.color_palette("Set1", n_colors=num_roles)
    pal_sorted = [pal[i] for i in ROLES_sorted_index]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    for i in range(len(all_bars)):
        if i==0:
            bottom = None
        else:
            bottom = all_bars_cumsum[i-1]
        top = all_bars[i]
        ax1.bar(range(num_tags), height=top, bottom=bottom, color=pal_sorted[i], edgecolor='white', width=barWidth)

    new_roles = [r for r in ROLES_sorted if r in role2token_targeted.keys()]

    if target_tag:
        updated_output_text_file = "_{}.".format(target_tag).join(args.output_text_file.rsplit('.', 1))
        with open(updated_output_text_file, 'w') as f_out:
            for role in new_roles:
                f_out.write('num of roles for {} is {}'.format(role, tag2role[target_tag][role]))
                f_out.write(str(set(role2token_targeted[role])))
                f_out.write('\n\n')

    # Custom X axis
    ax1.set_xticks(range(num_tags))
    ax1.set_xticklabels(TAGS_sorted, fontsize=7.5, rotation='vertical')
    ax1.set_xlabel(tag_type)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(range(num_tags))
    sum_vals = [str(tag2sum[tag]) for tag in TAGS_sorted]
    ax2.set_xticklabels(sum_vals, fontsize=7.5, rotation='vertical')
    ax2.set_xlabel("Number of Roles")

    plt.ylabel("Role Frequency")

    # Show graphic
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    updated_output_plot_file = "_{}.".format(tag_type).join(args.output_plot_file.rsplit('.', 1))
    plt.savefig(updated_output_plot_file)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default='./results/MNLI/tpr_attention.txt', type=str)
    parser.add_argument('--output_plot_file', default='./plot.png', type=str)
    parser.add_argument('--tag_type', default='pos_tags', choices=['all', 'pos_tags', 'ner_tags', 'dep_edge', 'const_parse_path', 'tree_depth'], type=str)
    parser.add_argument()
    parser.add_argument('--merge_tags', default=False, type=str2bool, help='merge similar tags')
    parser.add_argument('--prune', default=False, type=str2bool, help='prune low frequency values')
    parser.add_argument('--threshold', default=0.0, type=float, help='cutoff value for pruning')
    parser.add_argument('--normalize', default=False, type=str2bool, help='normalize number of roles for each tag')
    parser.add_argument('--target_tag', default=[], type=str, help='target tag to generate roles for', nargs='*')
    parser.add_argument('--output_text_file', default='./role2tokens.txt', type=str)

    args = parser.parse_args()

    if not os.path.exists('./workdir/log_eval/'):
        os.makedirs('./workdir/log_eval/', exist_ok=True)

    if len(args.target_tag) !=0 and len(args.target_tag) != 5 and args.tag_type == 'all':
        raise ValueError('When tag_type is set to all you must provide either to target_tags or one for each tag type')

    all_tag_types = ['pos_tags', 'ner_tags', 'dep_edge', 'const_parse_path', 'tree_depth']
    if args.tag_type == 'all':
        for i in range(len(all_tag_types)):
            tag_type = all_tag_types[i]
            logger.info('Processing tag type: {}'.format(tag_type))
            target_tag = None
            if len(args.target_tag):
                target_tag = args.target_tag[i].strip(',.: ')
            logger.info('Target tag is: {}'.format(target_tag))
            run(args, tag_type, target_tag)
    else:
        run(args, args.tag_type, args.target_tag)
    logger.info('*** Process is completed! ***')