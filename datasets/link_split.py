import torch
import numpy as np
import networkx as nx

from networkx.algorithms import tree
from torch_geometric.utils import negative_sampling, to_undirected, to_scipy_sparse_matrix


def undirected_label2directed_label(adj, edge_pairs, task, directed_graph=True, signed_directed=False):
    if len(edge_pairs) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    labels = -np.ones(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = np.array(list(map(list, edge_pairs)))
    edge_pairs = np.array(list(map(list, edge_pairs)))
    if signed_directed:
        directed_pos = (
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten() > 0).tolist()
        directed_neg = (
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten() < 0).tolist()
        inversed_pos = (
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten() > 0).tolist()
        inversed_neg = (
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten() < 0).tolist()
        undirected_pos = np.logical_and(directed_pos, inversed_pos)
        undirected_neg = np.logical_and(directed_neg, inversed_neg)
        undirected_pos_neg = np.logical_and(directed_pos, inversed_neg)
        undirected_neg_pos = np.logical_and(directed_neg, inversed_pos)
        directed_pos = list(map(tuple, edge_pairs[directed_pos].tolist()))
        directed_neg = list(map(tuple, edge_pairs[directed_neg].tolist()))
        inversed_pos = list(map(tuple, edge_pairs[inversed_pos].tolist()))
        inversed_neg = list(map(tuple, edge_pairs[inversed_neg].tolist()))
        undirected = np.logical_or(np.logical_or(np.logical_or(undirected_pos, undirected_neg), undirected_pos_neg), undirected_neg_pos)
        undirected = list(map(tuple, edge_pairs[np.array(undirected)].tolist()))
        edge_pairs = list(map(tuple, edge_pairs.tolist()))
        negative = np.array(
            list(set(edge_pairs) - set(directed_pos) - set(inversed_pos) - set(directed_neg) - set(inversed_neg)))
        directed_pos = np.array(list(set(directed_pos) - set(undirected)))
        inversed_pos = np.array(list(set(inversed_pos) - set(undirected)))
        directed_neg = np.array(list(set(directed_neg) - set(undirected)))
        inversed_neg = np.array(list(set(inversed_neg) - set(undirected)))
        directed = np.vstack([directed_pos, directed_neg])
        undirected = np.array(undirected)
        new_edge_pairs = directed
        new_edge_pairs = np.vstack([new_edge_pairs, new_edge_pairs[:, [1, 0]]])
        new_edge_pairs = np.vstack([new_edge_pairs, negative])
        labels = np.vstack([np.zeros((len(directed_pos), 1), dtype=np.int32),
                            np.ones((len(directed_neg), 1), dtype=np.int32)])
        labels = np.vstack([labels, 2 * np.ones((len(directed_pos), 1), dtype=np.int32),
                            3 * np.ones((len(directed_neg), 1), dtype=np.int32)])
        labels = np.vstack(
            [labels, 4*np.ones((len(negative), 1), dtype=np.int32)])
        label_weight = np.vstack([np.array(adj[directed_pos[:, 0], directed_pos[:, 1]]).flatten()[:, None],
                                np.array(adj[directed_neg[:, 0], directed_neg[:, 1]]).flatten()[:, None]])
        label_weight = np.vstack([label_weight, label_weight])
        label_weight = np.vstack(
            [label_weight, np.zeros((len(negative), 1), dtype=np.int32)])
        assert label_weight[labels==0].min() > 0
        assert label_weight[labels==1].max() < 0
        assert label_weight[labels==2].min() > 0
        assert label_weight[labels==3].max() < 0
        assert label_weight[labels==4].mean() == 0
    elif directed_graph:
        directed = (np.abs(
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()) > 0).tolist()
        inversed = (np.abs(
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten()) > 0).tolist()
        undirected = np.logical_and(directed, inversed)
        directed = list(map(tuple, edge_pairs[directed].tolist()))
        inversed = list(map(tuple, edge_pairs[inversed].tolist()))
        undirected = list(map(tuple, edge_pairs[undirected].tolist()))
        edge_pairs = list(map(tuple, edge_pairs.tolist()))
        negative = np.array(
            list(set(edge_pairs) - set(directed) - set(inversed)))
        directed = np.array(list(set(directed) - set(undirected)))
        inversed = np.array(list(set(inversed) - set(undirected)))
        new_edge_pairs = directed
        new_edge_pairs = np.vstack([new_edge_pairs, new_edge_pairs[:, [1, 0]]])
        new_edge_pairs = np.vstack([new_edge_pairs, negative])
        labels = np.zeros((len(directed), 1), dtype=np.int32)
        labels = np.vstack([labels, np.ones((len(directed), 1), dtype=np.int32)])
        labels = np.vstack(
            [labels, 2*np.ones((len(negative), 1), dtype=np.int32)])
        label_weight = np.array(adj[directed[:, 0], directed[:, 1]]).flatten()[:, None]
        label_weight = np.vstack([label_weight, label_weight])
        label_weight = np.vstack(
            [label_weight, np.zeros((len(negative), 1), dtype=np.int32)])
        assert abs(label_weight[labels==0]).min() > 0
        assert abs(label_weight[labels==1]).min() > 0
        assert label_weight[labels==2].mean() == 0
    else:
        undirected = []
        neg_edges = (
            np.abs(np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()) == 0)
        labels = np.ones(len(edge_pairs), dtype=np.int32)
        labels[neg_edges] = 2
        new_edge_pairs = edge_pairs
        label_weight = np.array(
            adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()
        labels[label_weight < 0] = 0
        if adj.data.min() < 0: # signed graph
            assert label_weight[labels==0].max() < 0
        assert label_weight[labels==1].min() > 0
        assert label_weight[labels==2].mean() == 0
    if task == 'existence':
        labels[labels == 1] = 0
        labels[labels == 2] = 1
        assert label_weight[labels == 1].mean() == 0
        assert abs(label_weight[labels == 0]).min() > 0
    return new_edge_pairs, labels.flatten(), label_weight.flatten(), undirected


def link_class_split(edge_index, A, 
                    cache_edge_split, edge_split_id,
                    size=None,
                    splits=10, 
                    prob_test=0.15, prob_val=0.05, 
                    task='direction', 
                    seed=0, 
                    maintain_connect=True,
                    ratio=1.0):
    cache_edge_split = cache_edge_split + "-" + task
    try:
        cache_edge_split_npy = cache_edge_split + ".npy"
        split_full = np.load(cache_edge_split_npy, allow_pickle=True)
        split_full = dict(enumerate(split_full.flatten(), 1))[1]
        observed_edge_idx = split_full[edge_split_id]['graph']
        observed_edge_weight = split_full[edge_split_id]['weights']
        train_edge_pairs_idx = split_full[edge_split_id]['train']['edges']
        val_edge_pairs_idx = split_full[edge_split_id]['val']['edges']
        test_edge_pairs_idx = split_full[edge_split_id]['test']['edges']
        train_edge_pairs_label = split_full[edge_split_id]['train']['label']
        val_edge_pairs_label = split_full[edge_split_id]['val']['label']
        test_edge_pairs_label = split_full[edge_split_id]['test']['label']
    except:
        print("Execute Edge split, it may take a while...")
        edge_index = edge_index.cpu()
        row, col = edge_index[0], edge_index[1]
        if size is None:
            size = int(max(torch.max(row), torch.max(col))+1)
        len_val = int(prob_val*len(row))
        len_test = int(prob_test*len(row))
        if task not in ["existence", "direction", 'three_class_digraph']:
            pos_ratio = (A>0).sum()/len(A.data)
            neg_ratio = 1 - pos_ratio
            len_val_pos = int(np.around(prob_val*len(row)*pos_ratio))
            len_val_neg = int(np.around(prob_val*len(row)*neg_ratio))
            len_test_pos = int(np.around(prob_test*len(row)*pos_ratio))
            len_test_neg = int(np.around(prob_test*len(row)*neg_ratio))
        undirect_edge_index = to_undirected(edge_index)
        neg_edges = negative_sampling(undirect_edge_index, num_neg_samples=len(
            edge_index.T), force_undirected=False).numpy().T
        neg_edges = map(tuple, neg_edges)
        neg_edges = list(neg_edges)
        all_edge_index = edge_index.T.tolist()
        A_undirected = to_scipy_sparse_matrix(undirect_edge_index)
        if maintain_connect:
            assert ratio == 1, "ratio should be 1.0 if maintain_connect=True"
            G = nx.from_scipy_sparse_matrix(
                A_undirected, create_using=nx.Graph, edge_attribute='weight')
            mst = list(tree.minimum_spanning_edges(
                G, algorithm="kruskal", data=False))
            all_edges = list(map(tuple, all_edge_index))
            mst_r = [t[::-1] for t in mst]
            nmst = list(set(all_edges) - set(mst) - set(mst_r))
            if len(nmst) < (len_val+len_test):
                raise ValueError(
                    "There are no enough edges to be removed for validation/testing. Please use a smaller prob_test or prob_val.")
        else:
            mst = []
            nmst = edge_index.T.tolist()

        rs = np.random.RandomState(seed)
        datasets = {}
        max_samples = int(ratio*len(edge_index.T))+1
        assert ratio <= 1.0 and ratio > 0, "ratio should be smaller than 1.0 and larger than 0"
        assert ratio > prob_val + prob_test, "ratio should be larger than prob_val + prob_test"
        for ind in range(splits):
            rs.shuffle(nmst)
            rs.shuffle(neg_edges)
            if task in ["direction", 'three_class_digraph']:
                ids_test = nmst[:len_test]+neg_edges[:len_test]
                ids_val = nmst[len_test:len_test+len_val] + \
                    neg_edges[len_test:len_test+len_val]
                if len_test+len_val < len(nmst):
                    ids_train = nmst[len_test+len_val:max_samples] + \
                        mst+neg_edges[len_test+len_val:max_samples]
                else:
                    ids_train = mst+neg_edges[len_test+len_val:max_samples]

                ids_test, labels_test, _, _ = undirected_label2directed_label(
                    A, ids_test, task, True)
                ids_val, labels_val, _, _ = undirected_label2directed_label(
                    A, ids_val, task, True)
                ids_train, labels_train, _, undirected_train = undirected_label2directed_label(
                    A, ids_train, task, True)
            elif task == "existence":
                ids_test = nmst[:len_test]+neg_edges[:len_test]
                ids_val = nmst[len_test:len_test+len_val] + \
                    neg_edges[len_test:len_test+len_val]
                if len_test+len_val < len(nmst):
                    ids_train = nmst[len_test+len_val:max_samples] + \
                        mst+neg_edges[len_test+len_val:max_samples]
                else:
                    ids_train = mst+neg_edges[len_test+len_val:max_samples]
                ids_test, labels_test, _, _ = undirected_label2directed_label(
                    A, ids_test, task, False)
                ids_val, labels_val, _, _ = undirected_label2directed_label(
                    A, ids_val, task, False)
                ids_train, labels_train, _, undirected_train = undirected_label2directed_label(
                    A, ids_train, task, False)
                weights = A[ids_val[:, 0], ids_val[:, 1]]
                assert abs(weights[:, labels_val == 1]).mean() == 0

            if task in ['direction']:
                ids_train = ids_train[labels_train < 2]
                labels_train = labels_train[labels_train < 2]
                ids_test = ids_test[labels_test < 2]
                labels_test = labels_test[labels_test < 2]
                ids_val = ids_val[labels_val < 2]
                labels_val = labels_val[labels_val < 2]
            observed_edges = -np.ones((len(ids_train), 2), dtype=np.int32)
            observed_weight = np.zeros((len(ids_train), 1), dtype=np.float32)
            direct = (
                np.abs(A[ids_train[:, 0], ids_train[:, 1]].data) > 0).flatten()
            observed_edges[direct, 0] = ids_train[direct, 0]
            observed_edges[direct, 1] = ids_train[direct, 1]
            observed_weight[direct, 0] = np.array(
                A[ids_train[direct, 0], ids_train[direct, 1]]).flatten()
            valid = (np.sum(observed_edges, axis=-1) >= 0)
            observed_edges = observed_edges[valid]
            observed_weight = observed_weight[valid]
            if len(undirected_train) > 0:
                undirected_train = np.array(undirected_train)
                observed_edges = np.vstack(
                    (observed_edges, undirected_train))
                observed_weight = np.vstack((observed_weight, np.array(A[undirected_train[:, 0],
                                                                    undirected_train[:, 1]]).flatten()[:, None]))

            assert(len(edge_index.T) >= len(observed_edges)), 'The original edge number is {} \
                while the observed graph has {} edges!'.format(len(edge_index.T), len(observed_edges))
            datasets[ind] = {}
            datasets[ind]['graph'] = torch.from_numpy(
                observed_edges.T).long()
            datasets[ind]['weights'] = torch.from_numpy(
                observed_weight.flatten()).float()
            datasets[ind]['train'] = {}
            datasets[ind]['train']['edges'] = torch.from_numpy(
                ids_train).long()
            datasets[ind]['train']['label'] = torch.from_numpy(
                labels_train).long()
            datasets[ind]['val'] = {}
            datasets[ind]['val']['edges'] = torch.from_numpy(
                ids_val).long()
            datasets[ind]['val']['label'] = torch.from_numpy(
                labels_val).long()
            datasets[ind]['test'] = {}
            datasets[ind]['test']['edges'] = torch.from_numpy(
                ids_test).long()
            datasets[ind]['test']['label'] = torch.from_numpy(
                labels_test).long()
        np.save(cache_edge_split, datasets)
        observed_edge_idx = datasets[edge_split_id]['graph']
        observed_edge_weight = datasets[edge_split_id]['weights']
        train_edge_pairs_idx = datasets[edge_split_id]['train']['edges']
        val_edge_pairs_idx = datasets[edge_split_id]['val']['edges']
        test_edge_pairs_idx = datasets[edge_split_id]['test']['edges']
        train_edge_pairs_label = datasets[edge_split_id]['train']['label']
        val_edge_pairs_label = datasets[edge_split_id]['val']['label']
        test_edge_pairs_label = datasets[edge_split_id]['test']['label']
    return observed_edge_idx, observed_edge_weight, train_edge_pairs_idx, val_edge_pairs_idx, test_edge_pairs_idx, train_edge_pairs_label, val_edge_pairs_label, test_edge_pairs_label