import scipy.sparse as sp

from operators.base_operator import TwoOrderPprApproxGraphOp
from operators.utils import adj_to_slow_first_second_ppr_approx_symmetric_norm


class SymDirTwoOrderPprApproxGraphOp(TwoOrderPprApproxGraphOp):
    def __init__(self, prop_steps, r=0.5, ppr_alpha=0.1):
        super(SymDirTwoOrderPprApproxGraphOp, self).__init__(prop_steps)
        self.r = r
        self.ppr_alpha = ppr_alpha

    def construct_adj(self, adj):
        adj = adj.tocoo()
        one_ppr_approx_sym_adj_normalized, second_ppr_approx_sym_adj_normalized = adj_to_slow_first_second_ppr_approx_symmetric_norm(adj, self.r, self.ppr_alpha)
        return one_ppr_approx_sym_adj_normalized, second_ppr_approx_sym_adj_normalized