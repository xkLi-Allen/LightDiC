import json
import torch
import numpy as np
import os.path as osp
import pickle as pkl
import scipy.sparse as sp

from itertools import chain
from torch_sparse import coalesce
from datasets.base_data import Graph
from datasets.base_dataset import NodeDataset
from datasets.link_split import link_class_split
from datasets.node_split import node_class_split
from datasets.utils import pkl_read_file, download_to, remove_self_loops, coomatrix_to_torch_tensor, set_spectral_adjacency_reg_features


class UsDUwPyGSDDataset(NodeDataset):
    def __init__(self, args, name="coraml", root="./datasets/directed/unweighted/", k=2,
                 node_split="official", node_split_id=0, edge_split="direction", edge_split_id=0):
        super(UsDUwPyGSDDataset, self).__init__(root + 'pygsd/', name, k)
        self.read_file()
        self.node_split = node_split
        self.node_split_id = node_split_id
        self.edge_split = edge_split
        self.edge_split_id = edge_split_id
        self.cache_node_split = osp.join(self.raw_dir, "{}-node-splits".format(self.name))
        self.cache_edge_split = osp.join(self.raw_dir, "{}-edge-splits".format(self.name))

        if self.name == "wikics":
            self.official_split = self.raw_file_paths[0]

        else:
            self.official_split = None
        
        if self.name not in ("wikitalk", "slashdot", "epinions"):
            self.train_idx, self.val_idx, self.test_idx, self.seed_idx, self.stopping_idx = node_class_split(name=name.lower(), data=self.data, 
                                                                                        cache_node_split=self.cache_node_split,
                                                                                        official_split=self.official_split,
                                                                                        split=self.node_split, node_split_id=self.node_split_id, 
                                                                                        train_size_per_class=20, val_size=500)
        
        edge_index = torch.from_numpy(np.vstack((self.edge.row.numpy(), self.edge.col.numpy()))).long()
        self.observed_edge_idx, self.observed_edge_weight, self.train_edge_pairs_idx, self.val_edge_pairs_idx, self.test_edge_pairs_idx, self.train_edge_pairs_label, self.val_edge_pairs_label, self.test_edge_pairs_label\
        = link_class_split(edge_index=edge_index, A=self.edge.sparse_matrix,
                        cache_edge_split=self.cache_edge_split, 
                        task=self.edge_split, edge_split_id=self.edge_split_id,
                        prob_val=0.15, prob_test=0.05, )
        self.num_node_classes = self.num_classes
        if edge_split in ("existence", "direction", "sign"):
            self.num_edge_classes = 2
        elif edge_split in ("three_class_digraph"):
            self.num_edge_classes = 3
        else:
            self.num_edge_classes = None

    @property
    def raw_file_paths(self):
        dataset_name = {
            'coraml': 'cora_ml.npz',
            'citeseerdir': 'citeseer.npz',
            'wikitalk': 'wikitalk.npz',
            'slashdot': 'slashdot.csv',
            'epinions': 'epinions.csv'
        }
        if self.name in ("coraml", "citeseerdir", "wikitalk", "slashdot", "epinions"):
            filename = dataset_name[self.name]
            return [osp.join(self.raw_dir, filename)]
        
        elif self.name == "wikics":
            filenames = ["data.json", "metadata.json", "statistics.json"]
            return [osp.join(self.raw_dir, filename) for filename in filenames]
        
    @property
    def processed_file_paths(self):
        return osp.join(self.processed_dir, f"{self.name}.graph")

    def read_file(self):
        self.data = pkl_read_file(self.processed_file_paths)
        self.edge = self.data.edge
        self.node = self.data.node
        self.x = self.data.x
        self.y = self.data.y
        self.adj = self.data.adj
        self.edge_type = self.data.edge_type
        self.num_features = self.data.num_features 
        self.num_classes = self.data.num_classes
        self.num_node = self.data.num_node
        self.num_edge = self.data.num_edge

    def download(self):

        dataset_drive_url = {
            'coraml': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/cora_ml.npz',
            'citeseerdir': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/citeseer.npz',
            'wikitalk': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/datasets/wikitalk.npz',
            'wikics': 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset',
            'slashdot': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/datasets/slashdot.csv',
            'epinions': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/datasets/epinions.csv',
        }
        file_url = dataset_drive_url[self.name]
        if self.name in ("coraml", "citeseerdir", "wikitalk", "slashdot", "epinions"):
            print("Download:{} to {}".format(file_url, self.raw_file_paths[0]))
            download_to(file_url, self.raw_file_paths[0])

        elif self.name == "wikics":
            print(file_url + "/data.json", self.raw_file_paths[0])
            print("Download:{} to {}".format(file_url + "/metadata.json", self.raw_file_paths[1]))
            download_to(file_url + "/metadata.json", self.raw_file_paths[1])
            print("Download:{} to {}".format(file_url + "/statistics.json", self.raw_file_paths[2]))
            download_to(file_url + "/statistics.json", self.raw_file_paths[2])
            
                                
    def process(self):
        if self.name in ("coraml", "citeseerdir"):
            with np.load(self.raw_file_paths[0], allow_pickle=True) as loader:
                loader = dict(loader)
                edge_index = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                        loader['attr_indptr']), shape=loader['attr_shape'])
                labels = loader.get('labels')

            edge_index = edge_index.tocoo()
            edge_index = coomatrix_to_torch_tensor(edge_index)
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            row, col = undi_edge_index

            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"

            features = torch.from_numpy(features.todense()).float()
            num_node = features.shape[0]
            labels = torch.from_numpy(labels).long()
      
        elif self.name == "wikics":
            with open(self.raw_file_paths[0], 'r') as f:
                ori_data = json.load(f)

            features = torch.tensor(ori_data['features'], dtype=torch.float)
            labels = torch.tensor(ori_data['labels'], dtype=torch.long)
            num_node = features.shape[0]

            edges = [[(i, j) for j in js] for i, js in enumerate(ori_data['links'])]
            edges = list(chain(*edges))
            edges = np.array(edges).transpose()

            edge_index = torch.from_numpy(edges)
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            row, col = undi_edge_index

            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"
     
        elif self.name == "wikitalk":
            adj = sp.load_npz(self.raw_file_paths[0])
            adj_coo = adj.tocoo()
            row, col = adj_coo.row, adj_coo.col
            edge_index = np.vstack((row, col))
            edge_index = torch.from_numpy(edge_index).long()
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            edge_index = undi_edge_index
            row, col = edge_index
            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"
            edge_num_node = edge_index.max().item() + 1
            num_node = edge_num_node
            features = set_spectral_adjacency_reg_features(edge_num_node, edge_index, edge_weight, self.k)
            labels = None

        elif self.name in ("slashdot", "epinions"):
            data = []
            edge_weight = []
            edge_index = []
            node_map = {}
            with open(self.raw_file_paths[0], 'r') as f:
                for line in f:
                    x = line.strip().split(',')
                    if float(x[2]) >= 0:
                        assert len(x) == 3
                        a, b = x[0], x[1]
                        if a not in node_map:
                            node_map[a] = len(node_map)
                        if b not in node_map:
                            node_map[b] = len(node_map)
                        a, b = node_map[a], node_map[b]
                        data.append([a, b])

                        edge_weight.append(float(x[2]))

                edge_index = [[i[0], int(i[1])] for i in data]
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_index = edge_index.t().contiguous()
                undi_edge_index = torch.unique(edge_index, dim=1)
                undi_edge_index = remove_self_loops(undi_edge_index)[0]
                edge_index = undi_edge_index
                row, col = edge_index
                edge_weight = torch.ones(len(row))
                edge_type = "UDUw"
                edge_num_node = edge_index.max().item() + 1
                num_node = edge_num_node
                features = set_spectral_adjacency_reg_features(edge_num_node, edge_index, edge_weight, self.k)
                labels = None
        
        g = Graph(row, col, edge_weight, num_node, edge_type, x=features, y=labels)

        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)
    




