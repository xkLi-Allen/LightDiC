import scipy.sparse as sp

from operators.base_operator import LRComGraphOp
from operators.utils import adj_to_directed_symmetric_mag_norm


class LRSymDirMagLaplacianGraphOp(LRComGraphOp):
    def __init__(self, filter_order, r=0.5, q=0.25):
        super(LRSymDirMagLaplacianGraphOp, self).__init__(filter_order, q)
        self.r = r
        self.q = q

    def construct_adj(self, adj):
        adj = adj.tocoo()
        real_adj_normalized, imag_adj_normalized = adj_to_directed_symmetric_mag_norm(adj, self.r, self.q)
        return real_adj_normalized, imag_adj_normalized