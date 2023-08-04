import scipy.sparse as sp

from operators.base_operator import identifyMessageOp
from operators.utils import adj_to_identify


class IdentifyGraphOp(identifyMessageOp):
    def __init__(self):
        super(IdentifyGraphOp, self).__init__()

    def construct_adj(self, adj):
        adj = adj.tocoo()
        adj = adj_to_identify(adj)
        return adj.tocsr()
