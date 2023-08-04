import scipy.sparse as sp

from operators.base_operator import TwoDirGraphOp
from operators.utils import adj_to_un_in_out_dir_symmetric_norm


class TwoDirLaplacianGraphOp(TwoDirGraphOp):
    def __init__(self, prop_steps, r=0.5):
        super(TwoDirLaplacianGraphOp, self).__init__(prop_steps)
        self.r = r

    def construct_adj(self, adj):
        adj = adj.tocoo()
        un_adj_normalized, in_adj_normalized, out_adj_normalized = adj_to_un_in_out_dir_symmetric_norm(adj, self.r)
        return un_adj_normalized, in_adj_normalized, out_adj_normalized
