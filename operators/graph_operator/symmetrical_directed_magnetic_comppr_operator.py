import scipy.sparse as sp

from operators.base_operator import ComGraphOp
from operators.utils import adj_to_directed_symmetric_mag_norm, PyGSD_adj_to_directed_symmetric_mag_norm


'''
    Simplex Personal PgaeRank (PPR):
        1. original pagerank:   \pi_{p r}=A_{r w} \pi_{p r}, A_{r w}=A D^{-1}
        2. root node pagerank:  \pi_{p p r}\left(i_x\right)=(1-\alpha) \hat{\tilde{\tilde{A}}} \pi_{p p r}\left(i_x\right)+\alpha i_x
                                \pi_{p p r}\left(i_x\right)=\alpha\left(I_n-(1-\alpha) \hat{\tilde{\tilde{A}}}\right)^{-1} i_x
    Approximate Personalized Propagation of Neural Predictions (APPNP)
        \begin{aligned}
        & Z^{(0)}=H=f_\theta(X) \\
        & Z^{(k+1)}=(1-\alpha) \hat{\tilde{A}} Z^{(k)}+\alpha H \\
        & Z^{(K)}=\operatorname{softmax}\left((1-\alpha) \hat{\tilde{A}} Z^{(K-1)}+\alpha H\right)
        \end{aligned}

    Complex Personal PgaeRank (PPR):    (A+iA)X -> \alpha(IX)+(1-\alpha)\left(AX+iAX\right)
        Real:   \alpha(IX)+(1-\alpha)AX
        Imag;   (1-\alpha)iAX

'''
class SymDirMagComPprGraphOp(ComGraphOp):
    def __init__(self, prop_steps, r=0.5, q=0.25, ppr_alpha=0.15):
        super(SymDirMagComPprGraphOp, self).__init__(prop_steps)
        self.r = r
        self.q = q
        self.ppr_alpha = ppr_alpha


    def construct_adj(self, adj):
        adj = adj.tocoo()
        real_adj_normalized, imag_adj_normalized = adj_to_directed_symmetric_mag_norm(adj, self.r, self.q)
        real_adj_normalized = (1 - self.ppr_alpha) * real_adj_normalized + self.ppr_alpha * sp.eye(adj.shape[0])
        imag_adj_normalized = (1 - self.ppr_alpha) * imag_adj_normalized
        return real_adj_normalized, imag_adj_normalized

