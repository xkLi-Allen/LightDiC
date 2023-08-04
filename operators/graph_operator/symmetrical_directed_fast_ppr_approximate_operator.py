import scipy.sparse as sp

from operators.base_operator import SimGraphOp
from operators.utils import adj_to_fast_ppr_approx_symmetric_norm


class SymDirFastPprApproxGraphOp(SimGraphOp):
    def __init__(self, prop_steps, r=0.5, ppr_alpha=0.1):
        super(SymDirFastPprApproxGraphOp, self).__init__(prop_steps)
        self.r = r
        self.ppr_alpha = ppr_alpha

    def construct_adj(self, adj):
        adj = adj.tocoo()
        fast_ppr_approx_sym_adj_normalized = adj_to_fast_ppr_approx_symmetric_norm(adj, self.r, self.ppr_alpha)
        return fast_ppr_approx_sym_adj_normalized