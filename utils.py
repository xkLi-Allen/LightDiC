import torch
import random
import numpy as np
import scipy.sparse as sp
from torch_sparse import coalesce
from scipy.sparse import csr_matrix
from torch_scatter import scatter_add
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as acc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

