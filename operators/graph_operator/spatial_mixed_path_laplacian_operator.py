import scipy.sparse as sp

from operators.base_operator import MixPathGraphOp
from operators.utils import adj_to_mixed_path_aggregation_conv_norm_rw


class MixPathLaplacianGraphOp(MixPathGraphOp):
    def __init__(self):
        super(MixPathLaplacianGraphOp, self).__init__()

    def construct_adj(self, adj):
        adj = adj.tocoo()
        adj_normalized, adj_normalized_t, source_index, target_index = adj_to_mixed_path_aggregation_conv_norm_rw(adj)
        return adj_normalized, adj_normalized_t, source_index, target_index
