import scipy.sparse as sp

from operators.base_operator import ComGraphOp
from operators.utils import adj_to_directed_symmetric_mag_norm, PyGSD_adj_to_directed_symmetric_mag_norm


class SymDirMagLaplacianGraphOp(ComGraphOp):
    def __init__(self, model_name, data_name, prop_steps, r=0.5, q=0.25):
        super(SymDirMagLaplacianGraphOp, self).__init__(model_name, data_name, prop_steps, r, q)
        self.r = r
        self.q = q

    def construct_adj(self, adj):
        adj = adj.tocoo()
        real_adj_normalized, imag_adj_normalized = adj_to_directed_symmetric_mag_norm(adj, self.r, self.q)
        return real_adj_normalized, imag_adj_normalized