import math
import torch
import scipy
import numpy as np
import os.path as osp
import scipy.sparse as sp
import numpy.ctypeslib as ctl

from ctypes import c_int
from torch import Tensor
from torch_sparse import coalesce
from scipy.sparse import csr_matrix
from torch_scatter import scatter_add
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import add_self_loops, to_scipy_sparse_matrix


def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]

    ctl_lib = ctl.load_library("./csrc/libmatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )

    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten().astype(np.float32)
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

def cuda_csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    
    ctl_lib = ctl.load_library("./csrc/libcudamatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                         c_int]
    ctl_lib.FloatCSRMulDense.restypes = c_int

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    data_nnz = len(data)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDense(answer, data_nnz, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

def adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized

def bern_adj_to_symmetric_norm(adj, r):
    num_nodes = adj.shape[0]
    edge_weight = torch.tensor(adj.data)
    edge_index = torch.vstack((torch.from_numpy(adj.row), torch.from_numpy(adj.col)))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt_left = torch.pow(deg, r-1)
    deg_inv_sqrt_left.masked_fill_(deg_inv_sqrt_left == float('inf'), 0)
    deg_inv_sqrt_right = torch.pow(deg, -r)
    deg_inv_sqrt_right.masked_fill_(deg_inv_sqrt_right == float('inf'), 0)

    edge_weight = deg_inv_sqrt_left[row] * edge_weight * deg_inv_sqrt_right[col]
    adj_normalized1 = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))

    fill_value = 2
    edge_index, edge_weight = add_self_loops(edge_index, -edge_weight, fill_value, num_nodes)
    row, col = edge_index
    adj_normalized2 = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))

    return adj_normalized1, adj_normalized2

def adj_to_directed_symmetric_mag_norm(adj, r, q):
    num_nodes = adj.shape[0]
    row, col = torch.tensor(adj.row, dtype=torch.long), torch.tensor(adj.col, dtype=torch.long)         # r,c
    edge_weight = torch.tensor(adj.data)                                        # weight

    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)       # r+c, c+r
    edge_index = torch.stack([row, col], dim=0)                                 # r+c, c+r -> [A(u,v), A(v,u)]
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)                     # weight x [A(u,v), A(v,u)]
    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)                  # weight x [A(u,v)}, -A(v,u)]
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)                      # weight x {[A(u,v), A(v,u)], [A(v,u), -A(v,u)]}
    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr,                 # edge_weight_sym -> edge_attr[:, 0]: weight x [A(u,v) + A(v,u)]
                                num_nodes,num_nodes, "add")                     # theta_weight -> edge_attr[:, 1]: weight x [A(u,v) - A(v,u)]
    edge_weight_sym = edge_attr[:, 0]                                           
    edge_weight_sym = edge_weight_sym/2                                         # edge_weight_sym = edge_weight_sym / 2 -> A_s(u,v)
    loop_weight_sym = torch.ones((num_nodes))
    edge_weight_sym = torch.hstack((edge_weight_sym, loop_weight_sym))          # edge_weight_sym = edge_weight_sym + self-loop -> \widetilde{A}_s(u,v)
    loop_edge_u_v = torch.linspace(0, num_nodes-1, steps=num_nodes, dtype=int)
    loop_edge_index_u = torch.hstack((edge_index_sym[0], loop_edge_u_v))
    loop_edge_index_v = torch.hstack((edge_index_sym[1], loop_edge_u_v))
    edge_index_sym = torch.vstack((loop_edge_index_u, loop_edge_index_v))       # edge_index_sym: \widetilde{A}_s(u,v) edge index

    theta_weight = edge_attr[:, 1]
    loop_weight = torch.zeros((num_nodes))
    theta_weight = torch.hstack((theta_weight, loop_weight))                    # theta_weight = theta_weight (no self-loops)

    row, col = edge_index_sym[0], edge_index_sym[1]  
    deg = scatter_add(edge_weight_sym, row, dim=0, dim_size=num_nodes)          # D_s(u,u)

    edge_weight_q = torch.exp(1j * 2 * np.pi * q * theta_weight)                # exp(i\theta^{q}) -> exp(i x 2\pi x q x  weight x [A(u,v) - A(v,u)])
    
    deg_inv_sqrt_left = torch.pow(deg, r-1)
    deg_inv_sqrt_left.masked_fill_(deg_inv_sqrt_left == float('inf'), 0)
    deg_inv_sqrt_right = torch.pow(deg, -r)
    deg_inv_sqrt_right.masked_fill_(deg_inv_sqrt_right == float('inf'), 0)
    edge_weight = deg_inv_sqrt_left[row] * edge_weight_sym * deg_inv_sqrt_right[col] * edge_weight_q


    edge_weight_real = edge_weight.real
    edge_weight_imag = edge_weight.imag

    real_adj_normalized = csr_matrix((edge_weight_real.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))
    imag_adj_normalized = csr_matrix((edge_weight_imag.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))

    return real_adj_normalized, imag_adj_normalized

def PyGSD_adj_to_directed_symmetric_mag_norm(adj, r, q):
    num_nodes = adj.shape[0]
    row, col = torch.tensor(adj.row, dtype=torch.long), torch.tensor(adj.col, dtype=torch.long)         # r,c
    edge_weight = torch.tensor(adj.data)                                        # weight

    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)       # r->r+c, c->c+r
    edge_index = torch.stack([row, col], dim=0)                                 # r|c , c|r (2, edgex2) -> {A(u,v)}, {A(v,u)}
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)                     # weight, weight -> weight x {+A(u,v)}, {+A(v,u)}
    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)                  # weight, -weight -> weight x {+A(u,v)}, {-A(v,u)}
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)                      # -> weight x {+A(u,v), +A(v,u)}, weight x {+A(v,u), -A(v,u)}
    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr,                 # edge_attr[:, 0]: weight x {+A(u,v) + +A(v,u)} = weight x (A(u,v) + A(v,u))
                                num_nodes,num_nodes, "add")                     # edge_attr[:, 1]: weight x {+A(u,v) + -A(v,u)} = weight x (A(u,v) - A(v,u))
                                                                                # edge_index_sym: raw edge_index
    edge_weight_sym = edge_attr[:, 0]                                           
    edge_weight_sym = edge_weight_sym/2                                         # {weight x (A(u,v) + A(v,u))} / 2 -> A_s(u,v)




    row, col = edge_index_sym[0], edge_index_sym[1]  
    deg = scatter_add(edge_weight_sym, row, dim=0, dim_size=num_nodes)          # D_s(u,u)

    edge_weight_q = torch.exp(1j * 2 * np.pi * q * edge_attr[:, 1])             # exp(i\theta^{q}) -> exp(i x 2\pi x q x {weight x (A(u,v) - A(v,u))})
    
    # Compute L_norm = D_sym^{-1/2} A_sym D_sym^{-1/2} Hadamard \exp(i \Theta^{(q)}).
    deg_inv_sqrt_left = torch.pow(deg, r-1)
    deg_inv_sqrt_left.masked_fill_(deg_inv_sqrt_left == float('inf'), 0)
    deg_inv_sqrt_right = torch.pow(deg, -r)
    deg_inv_sqrt_right.masked_fill_(deg_inv_sqrt_right == float('inf'), 0)
    edge_weight = deg_inv_sqrt_left[row] * edge_weight_sym * deg_inv_sqrt_right[col] * edge_weight_q

    # L = I - A_norm.
    edge_index, edge_weight = add_self_loops(edge_index_sym, -edge_weight, fill_value=1., num_nodes=num_nodes)
    
    # L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    # lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
    # lambda_max = float(lambda_max.real)
    lambda_max = 2

    edge_weight_real = edge_weight.real
    edge_weight_real = (2.0 * edge_weight_real) / lambda_max
    edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)
    edge_index_real, edge_weight_real = add_self_loops(edge_index, edge_weight_real, fill_value=-1.0, num_nodes=num_nodes)

    edge_weight_imag = edge_weight.imag
    edge_index_imag = edge_index.clone()
    edge_weight_imag = (2.0 * edge_weight_imag) / lambda_max
    edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)


    real_adj_normalized = csr_matrix((edge_weight_real.numpy(), (edge_index_real[0].numpy(), edge_index_real[1].numpy())),shape=(num_nodes, num_nodes))
    imag_adj_normalized = csr_matrix((edge_weight_imag.numpy(), (edge_index_imag[0].numpy(), edge_index_imag[1].numpy())),shape=(num_nodes, num_nodes))

    return real_adj_normalized, imag_adj_normalized

def MGC_adj_to_linear_rank_directed_symmetric_mag_norm(adj, r, q):
    id = sp.identity(adj.shape[0], format='csc')
    # Symmetrizing an adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if q != 0:
        dir = adj.transpose() - adj
        trs = np.exp(1j * 2 * np.pi * q * dir.toarray())
        trs = sp.csc_matrix(trs)
    else:
        trs = id # Fake
    
    adj = adj.tocoo()
    num_nodes = adj.shape[0]
    edge_weight = torch.ones((len(adj.data), ))
    edge_index = torch.vstack((torch.from_numpy(adj.row), torch.from_numpy(adj.col)))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index

    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt_left = torch.pow(deg, r-1)
    deg_inv_sqrt_left.masked_fill_(deg_inv_sqrt_left == float('inf'), 0)
    deg_inv_sqrt_right = torch.pow(deg, -r)
    deg_inv_sqrt_right.masked_fill_(deg_inv_sqrt_right == float('inf'), 0)
    edge_weight = deg_inv_sqrt_left[row] * edge_weight * deg_inv_sqrt_right[col]
    sym_norm_adj = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))

    if q == 0:
        sym_norm_mag_adj = sym_norm_adj
    elif q == 0.5:
        sym_norm_mag_adj = sym_norm_adj.multiply(trs.real)
    else:
        sym_norm_mag_adj = sym_norm_adj.multiply(trs)
    
    gso = -1 * sym_norm_mag_adj
    gso = gso.tocoo()
    num_nodes = gso.shape[0]
    real_adj_normalized = csr_matrix((gso.data.real, (gso.row, gso.col)),shape=(num_nodes, num_nodes))
    imag_adj_normalized = csr_matrix((gso.data.imag, (gso.col, gso.row)),shape=(num_nodes, num_nodes))
    return real_adj_normalized, imag_adj_normalized

def adj_to_un_in_out_dir_symmetric_norm(adj, r):
    num_nodes = adj.shape[0]
    edge_weight = torch.ones((len(adj.data), ))
    edge_index = torch.vstack((torch.from_numpy(adj.row), torch.from_numpy(adj.col)))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    un_edge_weight = edge_weight
    un_deg_inv_sqrt_left = torch.pow(deg, r-1)
    un_deg_inv_sqrt_left.masked_fill_(un_deg_inv_sqrt_left == float('inf'), 0)
    un_deg_inv_sqrt_right = torch.pow(deg, -r)
    un_deg_inv_sqrt_right.masked_fill_(un_deg_inv_sqrt_right == float('inf'), 0)
    un_edge_weight = un_deg_inv_sqrt_left[row] * un_edge_weight * un_deg_inv_sqrt_right[col]
    un_adj_normalized = csr_matrix((un_edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))

    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    in_L = torch.mm(p_dense.t(), p_dense) 
    out_L = torch.mm(p_dense, p_dense.t())

    # make nan to 0
    in_L[torch.isnan(in_L)] = 0
    # transfer dense L to sparse
    in_L_indices = torch.nonzero(in_L,as_tuple=False).t()
    in_L_values = in_L[in_L_indices[0], in_L_indices[1]]
    in_edge_index = in_L_indices
    in_edge_weight = in_L_values
    
    # row normalization
    in_row, in_col = in_edge_index
    in_deg = scatter_add(in_edge_weight, in_row, dim=0, dim_size=num_nodes)

    in_deg_inv_sqrt_left = torch.pow(in_deg, r-1)
    in_deg_inv_sqrt_left.masked_fill_(in_deg_inv_sqrt_left == float('inf'), 0)
    in_deg_inv_sqrt_right = torch.pow(in_deg, -r)
    in_deg_inv_sqrt_right.masked_fill_(in_deg_inv_sqrt_right == float('inf'), 0)
    in_edge_weight = in_deg_inv_sqrt_left[in_row] * in_edge_weight * in_deg_inv_sqrt_right[in_col]

    # make nan to 0
    out_L[torch.isnan(in_L)] = 0
    # transfer dense L to sparse
    out_L_indices = torch.nonzero(out_L,as_tuple=False).t()
    out_L_values = out_L[out_L_indices[0], out_L_indices[1]]
    out_edge_index = out_L_indices
    out_edge_weight = out_L_values
    
    # row normalization
    out_row, out_col = out_edge_index
    out_deg = scatter_add(out_edge_weight, out_row, dim=0, dim_size=num_nodes)

    out_deg_inv_sqrt_left = torch.pow(out_deg, r-1)
    out_deg_inv_sqrt_left.masked_fill_(out_deg_inv_sqrt_left == float('inf'), 0)
    out_deg_inv_sqrt_right = torch.pow(out_deg, -r)
    out_deg_inv_sqrt_right.masked_fill_(out_deg_inv_sqrt_right == float('inf'), 0)
    out_edge_weight = out_deg_inv_sqrt_left[out_row] * out_edge_weight * out_deg_inv_sqrt_right[out_col]

    in_adj_normalized = csr_matrix((in_edge_weight.numpy(), (in_row.numpy(), in_col.numpy())),shape=(num_nodes, num_nodes))
    out_adj_normalized = csr_matrix((out_edge_weight.numpy(), (out_row.numpy(), out_col.numpy())),shape=(num_nodes, num_nodes))

    return un_adj_normalized, in_adj_normalized, out_adj_normalized

def adj_to_identify(adj):
    num_nodes = adj.shape[0]
    edge_weight = torch.ones((len(adj.data), ))
    adj = csr_matrix((edge_weight.numpy(), (adj.row, adj.col)),shape=(num_nodes, num_nodes))
    return adj

def adj_to_fast_ppr_approx_symmetric_norm(adj, r, ppr_alpha, max_iter=100):
    num_nodes = adj.shape[0]
    edge_weight = torch.ones((len(adj.data), ))
    edge_index = torch.vstack((torch.from_numpy(adj.row), torch.from_numpy(adj.col)))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index

    # from tensor to csr matrix
    sparse_adj = sp.csr_matrix((edge_weight.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())), shape=(num_nodes, num_nodes))

    tol = 1e-6

    r_ = np.asarray(sparse_adj.sum(axis=1)).reshape(-1)
    k = r_.nonzero()[0]
    D_1 = sp.csr_matrix((1 / r_[k], (k, k)), shape=(num_nodes, num_nodes))
    personalize = np.ones(num_nodes)
    personalize = personalize.reshape(num_nodes, 1)
    s = 1/(1+ppr_alpha)/num_nodes * personalize
    z_T = ((ppr_alpha*(1+ppr_alpha)) * (r_ != 0) + ((1-ppr_alpha)/(1+ppr_alpha)+ppr_alpha*(1+ppr_alpha))
           * (r_ == 0))[scipy.newaxis, :]
    W = (1-ppr_alpha) * sparse_adj.T @ D_1
    x = s
    oldx = np.zeros((num_nodes, 1))
    iteration = 0
    while scipy.linalg.norm(x - oldx) > tol:
        oldx = x
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
    x = x / sum(x)
    x = x.reshape(-1)
    p = D_1 * sparse_adj
    pi_sqrt = sp.diags(np.power(x, 0.5))
    pi_inv_sqrt = sp.diags(np.power(x, -0.5))
    L = (pi_sqrt * p * pi_inv_sqrt + pi_inv_sqrt * p.T * pi_sqrt)/2.0
    L.data[np.isnan(L.data)] = 0.0


    L = L.tocoo()
    values = L.data
    indices = np.vstack((L.row, L.col))

    L_indices = torch.LongTensor(indices)
    L_values = torch.FloatTensor(values)

    edge_index = L_indices
    edge_weight = L_values
    row, col = edge_index
    
    adj_normalized = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))
    return adj_normalized

def adj_to_first_second_fast_ppr_approx_symmetric_norm(adj, r, ppr_alpha, max_iter=100):
    num_nodes = adj.shape[0]
    edge_weight = torch.ones((len(adj.data), ))
    edge_index = torch.vstack((torch.from_numpy(adj.row), torch.from_numpy(adj.col)))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index

    # from tensor to csr matrix
    sparse_adj = sp.csr_matrix((edge_weight.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())), shape=(num_nodes, num_nodes))
    tol = 1e-6
    r_ = np.asarray(sparse_adj.sum(axis=1)).reshape(-1)
    k = r_.nonzero()[0]
    D_1 = sp.csr_matrix((1 / r_[k], (k, k)), shape=(num_nodes, num_nodes))
    personalize = np.ones(num_nodes)
    personalize = personalize.reshape(num_nodes, 1)
    s = 1/(1+ppr_alpha)/num_nodes * personalize
    z_T = ((ppr_alpha*(1+ppr_alpha)) * (r_ != 0) + ((1-ppr_alpha)/(1+ppr_alpha)+ppr_alpha*(1+ppr_alpha))
           * (r_ == 0))[scipy.newaxis, :]
    W = (1-ppr_alpha) * sparse_adj.T @ D_1
    x = s
    oldx = np.zeros((num_nodes, 1))
    iteration = 0
    while scipy.linalg.norm(x - oldx) > tol:
        oldx = x
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
    x = x / sum(x)
    x = x.reshape(-1)
    p = D_1 * sparse_adj
    pi_sqrt = sp.diags(np.power(x, 0.5))
    pi_inv_sqrt = sp.diags(np.power(x, -0.5))
    L = (pi_sqrt * p * pi_inv_sqrt + pi_inv_sqrt * p.T * pi_sqrt)/2.0
    L.data[np.isnan(L.data)] = 0.0
    L = L.tocoo()
    values = L.data
    indices = np.vstack((L.row, L.col))
    L_indices = torch.LongTensor(indices)
    L_values = torch.FloatTensor(values)
    edge_index = L_indices
    edge_weight = L_values
    row, col = edge_index
    one_order_adj_normalized = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))



    L_in = p.T * p
    L_out = p * p.T
    L_in_hat = L_in
    L_out_hat = L_out
    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0
    p = (L_in_hat + L_out_hat) / 2.0
    p.data[np.isnan(p.data)] = 0.0
    p = p.tocoo()
    sparse_adj = sp.csr_matrix((p.data, (p.row, p.col)), shape=(num_nodes, num_nodes))
    tol = 1e-6
    r_ = np.asarray(sparse_adj.sum(axis=1)).reshape(-1)
    k = r_.nonzero()[0]
    D_1 = sp.csr_matrix((1 / r_[k], (k, k)), shape=(num_nodes, num_nodes))
    personalize = np.ones(num_nodes)
    personalize = personalize.reshape(num_nodes, 1)
    s = 1/(1+ppr_alpha)/num_nodes * personalize
    z_T = ((ppr_alpha*(1+ppr_alpha)) * (r_ != 0) + ((1-ppr_alpha)/(1+ppr_alpha)+ppr_alpha*(1+ppr_alpha))
           * (r_ == 0))[scipy.newaxis, :]
    W = (1-ppr_alpha) * sparse_adj.T @ D_1
    x = s
    oldx = np.zeros((num_nodes, 1))
    iteration = 0
    while scipy.linalg.norm(x - oldx) > tol:
        oldx = x
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
    x = x / sum(x)
    x = x.reshape(-1)
    p = D_1 * sparse_adj
    pi_sqrt = sp.diags(np.power(x, 0.5))
    pi_inv_sqrt = sp.diags(np.power(x, -0.5))
    L = (pi_sqrt * p * pi_inv_sqrt + pi_inv_sqrt * p.T * pi_sqrt)/2.0
    L.data[np.isnan(L.data)] = 0.0
    L = L.tocoo()
    values = L.data
    
    indices = np.vstack((L.row, L.col))
    L_indices = torch.LongTensor(indices)
    L_values = torch.FloatTensor(values)
    edge_index = L_indices
    edge_weight = L_values
    row, col = edge_index
    two_order_adj_normalized = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))

    return one_order_adj_normalized, two_order_adj_normalized

def adj_to_slow_first_second_ppr_approx_symmetric_norm(adj, r, ppr_alpha):
    num_nodes = adj.shape[0]
    edge_weight = torch.ones((len(adj.data), ))
    edge_index = torch.vstack((torch.from_numpy(adj.row), torch.from_numpy(adj.col)))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(
        edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1, num_nodes+1]))
    p_v[0:num_nodes, 0:num_nodes] = (1-ppr_alpha) * p_dense
    p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes, num_nodes] = ppr_alpha
    p_v[num_nodes, num_nodes] = 0.0
    p_ppr = p_v

    eig_value, left_vector = scipy.linalg.eig(
        p_ppr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    _, ind = eig_value.sort(descending=True)

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi
    # print(pi)
    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    pi_sqrt = pi_sqrt.to(p_ppr.device)
    pi_inv_sqrt = pi_inv_sqrt.to(p_ppr.device)
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) +
         torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0
    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()

    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    pi_deg_inv_sqrt_left = torch.pow(deg, r-1)
    pi_deg_inv_sqrt_left.masked_fill_(pi_deg_inv_sqrt_left == float('inf'), 0)
    pi_deg_inv_sqrt_right = torch.pow(deg, -r)
    pi_deg_inv_sqrt_right.masked_fill_(pi_deg_inv_sqrt_right == float('inf'), 0)

    edge_weight = pi_deg_inv_sqrt_left[row] * edge_weight * pi_deg_inv_sqrt_right[col]
    pi_one_order_adj_normalized = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))


    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())

    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    pi_deg_inv_sqrt_left = torch.pow(deg, r-1)
    pi_deg_inv_sqrt_left.masked_fill_(pi_deg_inv_sqrt_left == float('inf'), 0)
    pi_deg_inv_sqrt_right = torch.pow(deg, -r)
    pi_deg_inv_sqrt_right.masked_fill_(pi_deg_inv_sqrt_right == float('inf'), 0)

    edge_weight = pi_deg_inv_sqrt_left[row] * edge_weight * pi_deg_inv_sqrt_right[col]
    pi_two_order_adj_normalized = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))


    return pi_one_order_adj_normalized, pi_two_order_adj_normalized

def adj_to_mixed_path_aggregation_conv_norm_rw(adj):
    num_nodes = adj.shape[0]
    edge_weight = torch.tensor(adj.data)
    edge_index = torch.vstack((torch.from_numpy(adj.row), torch.from_numpy(adj.col)))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    row_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = row_deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    adj_normalized = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),shape=(num_nodes, num_nodes))
    adj_normalized_t = csr_matrix((edge_weight.numpy(), (col.numpy(), row.numpy())),shape=(num_nodes, num_nodes))
    return adj_normalized, adj_normalized_t, torch.unique(row), torch.unique(col)

def one_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError("This function is designed for list(feature) and tensor(weight)!")
    elif len(feat_list) != weight_list.shape[0]:
        raise ValueError("The feature list and the weight list have different lengths!")
    elif len(weight_list.shape) != 1:
        raise ValueError("The weight list should be a 1d tensor!")

    feat_shape = feat_list[0].shape
    feat_reshape = torch.vstack([feat.view(1, -1).squeeze(0) for feat in feat_list])
    weighted_feat = (feat_reshape * weight_list.view(-1, 1)).sum(dim=0).view(feat_shape)
    return weighted_feat

def two_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError("This function is designed for list(feature) and tensor(weight)!")
    elif len(feat_list) != weight_list.shape[1]:
        raise ValueError("The feature list and the weight list have different lengths!")
    elif len(weight_list.shape) != 2:
        raise ValueError("The weight list should be a 2d tensor!")

    feat_reshape = torch.stack(feat_list, dim=2)
    weight_reshape = weight_list.unsqueeze(dim=2)
    weighted_feat = torch.bmm(feat_reshape, weight_reshape).squeeze(dim=2)
    return weighted_feat

def squeeze_first_dimension(feat_list):
    if isinstance(feat_list, Tensor):
        if len(feat_list.shape) == 3:
            feat_list = feat_list[0]
    elif isinstance(feat_list, list):
        if len(feat_list[0].shape) == 3:
            for i in range(len(feat_list)):
                feat_list[i] = feat_list[i].squeeze(dim=0)
    return feat_list

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)