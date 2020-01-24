import torch
import numpy as np
import seaborn as sns
from sklearn import cluster
import logging

from matplotlib import pyplot as plt
from collections import Counter
from utils.global_vars import POS_TAGS_MAP

from MulticoreTSNE import MulticoreTSNE as tsne

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_attention(args, aFs, aRs, orig_to_token_maps):

    F_list, R_list = [], []
    F_full, R_full = [], []

    if args.save_strategy == 'sample':
        for i, (aF_full, aR_all) in enumerate(zip(aFs, aRs)):
            input_F = []
            input_R = []
            for aF, aR in zip(aF_full, aR_all):
                aF, aR = np.array(aF.cpu()), np.array(aR.cpu())
                idx_F = np.random.choice(np.arange(len(aF)), size=1, p=aF/aF.sum())[0]
                idx_R = np.random.choice(np.arange(len(aR)), size=1, p=aR/aR.sum())[0]
                input_F.append(idx_F)
                input_R.append(idx_R)

            map = orig_to_token_maps[i]
            index = map.tolist().index(-1)
            orig_to_token_maps_unpadded = map[:index]
            # dim=0 is seq_length
            # remove first and last token corresponding to [CLS] and [SEP] tokens
            F_list.append(torch.index_select(torch.tensor(input_F).type(aFs.type()), dim=0, index=orig_to_token_maps_unpadded).detach().cpu().tolist()[1:-1])
            R_list.append(torch.index_select(torch.tensor(input_R).type(aRs.type()), dim=0, index=orig_to_token_maps_unpadded).detach().cpu().tolist()[1:-1])

    else:
        if args.save_strategy == 'topK':
            F_selected = torch.topk(aFs, k=args.K, dim=-1)[1] # choose top K indices
            R_selected = torch.topk(aRs, k=args.K, dim=-1)[1] # choose top K indices
        elif args.save_strategy == 'selectK':
            F_selected = torch.topk(aFs, k=args.K, dim=-1)[1][:,:,[-1]] # choose K th best index
            R_selected = torch.topk(aRs, k=args.K, dim=-1)[1][:,:,[-1]] # choose K th best index
        for i, (input_F, input_R) in enumerate(zip(F_selected, R_selected)):
            map = orig_to_token_maps[i]
            index = map.tolist().index(-1)
            orig_to_token_maps_unpadded = map[:index]
            # dim=0 is seq_length
            # remove first and last token corresponding to [CLS] and [SEP] tokens
            F_list.append(torch.index_select(input_F, dim=0, index=orig_to_token_maps_unpadded).detach().cpu().tolist()[1:-1])
            R_list.append(torch.index_select(input_R, dim=0, index=orig_to_token_maps_unpadded).detach().cpu().tolist()[1:-1])

        # also save full embeddings for T-SNE and Kmeans
        for i, (input_F, input_R) in enumerate(zip(aFs, aRs)):
            map = orig_to_token_maps[i]
            index = map.tolist().index(-1)
            orig_to_token_maps_unpadded = map[:index]
            # dim=0 is seq_length
            # remove first and last token corresponding to [CLS] and [SEP] tokens
            F_full.append(torch.index_select(input_F, dim=0, index=orig_to_token_maps_unpadded).detach().cpu().tolist()[1:-1])
            R_full.append(torch.index_select(input_R, dim=0, index=orig_to_token_maps_unpadded).detach().cpu().tolist()[1:-1])

    return F_list, R_list, F_full, R_full

def calculate_purity(cluster_assignments, labels):

    clusters = set(list(cluster_assignments))

    # find out what class is most frequent in each cluster
    cluster_classes = {}
    for cluster in clusters:
        cluster_labels = labels[cluster_assignments == cluster]
        cluster_classes[cluster] = str(Counter(cluster_labels).most_common(1)[0][0])

    # Return the percentage of indices in agreement.
    num_agreed = 0
    for i in range(len(labels)):
        if cluster_classes[cluster_assignments[i]] == labels[i]:
            num_agreed += 1

    return float(num_agreed) / len(labels)


def perform_tsne(args, vals, tsne_label):

    label_mapping = {'pos': 'pos_tags', 'ner': 'ner_tags', 'dep_edge': 'dep_edge',
                     'depth': 'tree_depth', 'const': 'const_parse_path'}

    data = vals.items()
    F_embeddings, R_embeddings, labels = [], [], []

    tag2main = {}
    if tsne_label == 'pos':
        for k, v in POS_TAGS_MAP.items():
            for val in v:
                tag2main[val] = k

    for id, val in data:
        role, aFs, aRs = val[label_mapping[tsne_label]], val['all_aFs_full'], val['all_aRs_full']
        # return main tag for pos label. Otherwise return the tag itself.
        labels.extend([tag2main.get(tag, tag) for tag in role])
        F_embeddings.extend(aFs)
        R_embeddings.extend(aRs)

    assert len(labels) == len(F_embeddings) == len(R_embeddings)
    logger.info('***{} tokens are being processed for visualization***'.format(len(labels)))

    R_embeddings, F_embeddings, labels = np.array(R_embeddings), np.array(F_embeddings), np.array(labels)

    if args.metric == "cosine":
        R_embeddings = R_embeddings / np.linalg.norm(R_embeddings, axis=1, keepdims=True)
        F_embeddings = F_embeddings / np.linalg.norm(F_embeddings, axis=1, keepdims=True)

    R_proj = tsne(n_jobs=args.n_jobs, perplexity=args.perplexity, n_components=args.n_components, random_state=0, metric=args.metric, verbose=True, n_iter=args.n_iter).fit_transform(R_embeddings)
    F_proj = tsne(n_jobs=args.n_jobs, perplexity=args.perplexity, n_components=args.n_components, random_state=0, metric=args.metric, verbose=True, n_iter=args.n_iter).fit_transform(F_embeddings)

    pal = sns.color_palette("Paired", n_colors=len(set(labels)))

    proj = {'role': R_proj, 'filler': F_proj}
    for mode in ('role', 'filler'):
        data = proj[mode]
        fig, ax = plt.subplots()

        for i, label in enumerate(set(labels)):
            vectors = data[labels == label]
            xs, ys = vectors[:, 0], vectors[:, 1]
            ax.scatter(xs, ys, label=label, c=[pal[i]]*len(xs), alpha=0.6)

        # plt.tight_layout()
        plt.title("T-SNE for {} vectors".format(mode))
        plt.legend(ncol=2)
        plt.savefig('./tsne_{}.png'.format(mode))
        plt.show()

        if args.do_Kmeans:
            kmeans = cluster.KMeans(n_clusters=args.n_clusters)
            kmeans.fit(data)
            cluster_assignments = kmeans.labels_
            purity = calculate_purity(np.array(cluster_assignments), labels)
            logger.info('clustering purity for {} vectors is {}'.format(mode, purity))
