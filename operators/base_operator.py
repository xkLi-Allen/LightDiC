import os
import platform

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.special import comb
from torch import Tensor

from operators.utils import csr_sparse_dense_matmul
 

class TwoOrderPprApproxGraphOp:
    def __init__(self, prop_steps):
        self.prop_steps = prop_steps
        self.one_adj = None
        self.two_adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.one_adj, self.two_adj = self.construct_adj(adj)
        one_prop_feat_list = []
        two_prop_feat_list = []

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.one_adj.shape[1] != feature.shape[0] or self.two_adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        one_prop_feat_list = [feature]
        two_prop_feat_list = [feature]

        for _ in range(self.prop_steps):
            one_feat_temp = ada_platform_one_step_propagation(self.one_adj, one_prop_feat_list[-1])
            two_feat_temp = ada_platform_one_step_propagation(self.two_adj, two_prop_feat_list[-1])
            one_prop_feat_list.append(one_feat_temp)
            two_prop_feat_list.append(two_feat_temp)

        return [torch.FloatTensor(feat) for feat in one_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in two_prop_feat_list]

# Might include training parameters
class TwoOrderPprApproxMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(TwoOrderPprApproxMessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, one_feat_list, two_feat_list):
        return NotImplementedError

    def aggregate(self, one_feat_list, two_feat_list):
        if not isinstance(one_feat_list, list) or not isinstance(two_feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in one_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The one order feature matrices must be tensors!")
        for feat in two_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The two order feature matrices must be tensors!")
            
        return self.combine(one_feat_list, two_feat_list)


class calculator:
    def __init__(self, value, r_step=0, i_step=0):
        self.value = value
        self.r_step = r_step
        self.i_step = i_step

    def prop_step(self):
        return self.r_step + self.i_step
    
    def reversal(self):
        if self.i_step & 1 == 0 and self.i_step != 0:
            self.value = -self.value
    
    def set_variable(self, value, r=False, i=False):
        self.value = value
        if r:   self.r_step += 1
        elif i: self.i_step += 1


class ComGraphOp:
    def __init__(self, model_name, data_name, prop_steps, r=None, q=None):
        self.data_name = data_name
        self.model_name = model_name
        self.prop_steps = prop_steps
        self.r = r 
        self.q = q
        self.real_adj = None
        self.imag_adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        if not os.path.exists(f"./data_preprocess/{self.model_name}"):
            os.makedirs(f"./data_preprocess/{self.model_name}")
        real_saved = f'./data_preprocess/{self.model_name}/{self.data_name}_realfeat_{self.r}_{self.q}_{self.prop_steps}.pt'
        imag_saved = f'./data_preprocess/{self.model_name}/{self.data_name}_imagfeat_{self.r}_{self.q}_{self.prop_steps}.pt'

        try:
            propagate_real_feature, propagate_imag_feature = torch.load(real_saved), torch.load(imag_saved)

        except:
            self.real_adj, self.imag_adj = self.construct_adj(adj)

            if not isinstance(adj, sp.csr_matrix):
                raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
            elif not isinstance(feature, np.ndarray):
                if isinstance(feature, Tensor):
                    feature = feature.numpy()
                else:
                    raise TypeError("The feature matrix must be a numpy.ndarray!")
            elif self.real_adj.shape[1] != feature.shape[0] or self.imag_adj.shape[1] != feature.shape[0]:
                raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

            init_real_calculator = calculator(feature)
            init_imag_calculator = calculator(feature)

            real_prop_feat_list = [init_real_calculator.value]
            imag_prop_feat_list = [init_imag_calculator.value]
            tmp_prop_feat_calculator_in_list = []
            tmp_prop_feat_calculator_out_list = []

            for steps in range(self.prop_steps):
                if steps == 0:
                    real_feat_temp_value = ada_platform_one_step_propagation(self.real_adj, real_prop_feat_list[-1])
                    init_real_calculator.set_variable(real_feat_temp_value, r=True)
                    tmp_prop_feat_calculator_in_list.append(init_real_calculator)
                    real_prop_feat_list.append(init_real_calculator.value)

                    imag_feat_temp_value = ada_platform_one_step_propagation(self.imag_adj, imag_prop_feat_list[-1])
                    init_imag_calculator.set_variable(imag_feat_temp_value, i=True)
                    tmp_prop_feat_calculator_in_list.append(init_imag_calculator)
                    imag_prop_feat_list.append(init_imag_calculator.value)

                
                else:
                    for k in range(len(tmp_prop_feat_calculator_in_list)):
                        tmp_calculator = tmp_prop_feat_calculator_in_list[k]
                        new_calculator = calculator(tmp_calculator.value, tmp_calculator.r_step, tmp_calculator.i_step)
                        tmp_value = ada_platform_one_step_propagation(self.real_adj, tmp_calculator.value)
                        new_calculator.set_variable(tmp_value, r=True)
                        tmp_prop_feat_calculator_out_list.append(new_calculator)

                    for k in range(len(tmp_prop_feat_calculator_in_list)):
                        tmp_calculator = tmp_prop_feat_calculator_in_list[k]
                        new_calculator = calculator(tmp_calculator.value, tmp_calculator.r_step, tmp_calculator.i_step)
                        tmp_value = ada_platform_one_step_propagation(self.imag_adj, tmp_calculator.value)
                        new_calculator.set_variable(tmp_value, i=True)
                        new_calculator.reversal()
                        tmp_prop_feat_calculator_out_list.append(new_calculator)

                    real_feat, imag_feat = calculate_real_imag_feat(tmp_prop_feat_calculator_out_list)
                    real_prop_feat_list.append(real_feat)
                    imag_prop_feat_list.append(imag_feat)
                    tmp_prop_feat_calculator_in_list = tmp_prop_feat_calculator_out_list
                    tmp_prop_feat_calculator_out_list = []
            
            propagate_real_feature = [torch.FloatTensor(feat) for feat in real_prop_feat_list]
            propagate_imag_feature = [torch.FloatTensor(feat) for feat in imag_prop_feat_list]
            torch.save(propagate_real_feature, real_saved)
            torch.save(propagate_imag_feature, imag_saved)
            
        return propagate_real_feature, propagate_imag_feature


# Might include training parameters
class ComMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(ComMessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, real_feat_list, imag_feat_list):
        return NotImplementedError

    def aggregate(self, real_feat_list, imag_feat_list):
        if not isinstance(real_feat_list, list) or not isinstance(imag_feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in real_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The real feature matrices must be tensors!")
        for feat in imag_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The imag feature matrices must be tensors!")
            
        return self.combine(real_feat_list, imag_feat_list)


class LRComGraphOp:
    def __init__(self, filter_order, q):
        self.filter_order = filter_order
        self.q = q
        self.adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
            
        self.real_adj, self.imag_adj = self.construct_adj(adj)
        if self.real_adj.shape[1] != feature.shape[0] or self.imag_adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        if self.q != 0 and self.q != 0.5:
            # LinearRank
            if self.filter_order == 1:
                real_feat = feature
            else:
                tmp = 2 / (self.filter_order + 1) * feature
                real_feat = tmp # 0-th term
                for k in range(self.filter_order-1):
                    # (k+1)-th term
                    tmp = (self.filter_order - k - 1) / (self.filter_order - k) * ada_platform_one_step_propagation(self.real_adj, tmp)
                    real_feat = real_feat + tmp

            if self.filter_order == 1:
                imag_feat = feature
            else:
                tmp = 2 / (self.filter_order + 1) * feature
                imag_feat = tmp # 0-th term
                for k in range(self.filter_order-1):
                    # (k+1)-th term
                    tmp = (self.filter_order - k - 1) / (self.filter_order - k) * ada_platform_one_step_propagation(self.imag_adj, tmp)
                    imag_feat = imag_feat + tmp
            
            real_feat, imag_feat = torch.FloatTensor(real_feat), torch.FloatTensor(imag_feat)
            feature = real_feat + 1j * imag_feat
        else:
            if self.filter_order == 1:
                feature = feature
            else:
                tmp = 2 / (self.filter_order + 1) * feature
                feature = tmp # 0-th term
                for k in range(self.filter_order-1):
                    # (k+1)-th term
                    tmp = (self.filter_order - k - 1) / (self.filter_order - k) * ada_platform_one_step_propagation(self.real_adj, tmp)
                    feature = feature + tmp
            feature = torch.FloatTensor(feature)
        return feature
    

class TwoDirGraphOp:
    def __init__(self, prop_steps):
        self.prop_steps = prop_steps
        self.un_adj = None
        self.in_adj = None
        self.out_adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.un_adj, self.in_adj, self.out_adj = self.construct_adj(adj)
        un_prop_feat_list = []
        in_prop_feat_list = []
        out_prop_feat_list = []

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.un_adj.shape[1] != feature.shape[0] or self.in_adj.shape[1] != feature.shape[0] or self.out_adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        un_prop_feat_list = [feature]
        in_prop_feat_list = [feature]
        out_prop_feat_list = [feature]

        for _ in range(self.prop_steps):
            un_feat_temp = ada_platform_one_step_propagation(self.un_adj, un_prop_feat_list[-1])
            in_feat_temp = ada_platform_one_step_propagation(self.in_adj, in_prop_feat_list[-1])
            out_feat_temp = ada_platform_one_step_propagation(self.out_adj, out_prop_feat_list[-1])
            un_prop_feat_list.append(un_feat_temp)
            in_prop_feat_list.append(in_feat_temp)
            out_prop_feat_list.append(out_feat_temp)

        return [torch.FloatTensor(feat) for feat in un_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in in_prop_feat_list], \
                [torch.FloatTensor(feat) for feat in out_prop_feat_list]


# Might include training parameters
class TwoDirMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(TwoDirMessageOp, self).__init__()
        self.aggr_type = None

    def aggr_type(self):
        return self.aggr_type

    def combine(self, un_feat_list, in_feat_list, out_feat_list):
        return NotImplementedError

    def aggregate(self, un_feat_list, in_feat_list, out_feat_list):
        if not isinstance(un_feat_list, list) or not isinstance(in_feat_list, list) or not isinstance(out_feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in un_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The un direction feature matrices must be tensors!")
        for feat in in_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The in direction feature matrices must be tensors!")
        for feat in out_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The out direction feature matrices must be tensors!")
            
        return self.combine(un_feat_list, in_feat_list, out_feat_list)


class identifyMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(identifyMessageOp, self).__init__()
        self.aggr_type = None

    def aggr_type(self):
        return self.aggr_type

    def combine(self, un_feat_list, in_feat_list, out_feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The un direction feature matrices must be tensors!")
            
        return self.combine(feat_list)

class MixPathGraphOp:
    def __init__(self):
        self.adj = None
        self.adj_t = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.adj, self.adj_t = self.construct_adj(adj)
        feat_list = []

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.un_adj.shape[1] != feature.shape[0] or self.in_adj.shape[1] != feature.shape[0] or self.out_adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        feat_list = [feature]

        return [torch.FloatTensor(feat) for feat in feat_list]


# Might include training parameters
class MixPathMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(TwoDirMessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The un direction feature matrices must be tensors!")
            
        return self.combine(feat_list)


class MixPathGraphOp:
    def __init__(self):
        self.adj = None
        self.adj_t = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.adj, self.adj_t = self.construct_adj(adj)
        feat_list = []

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.un_adj.shape[1] != feature.shape[0] or self.in_adj.shape[1] != feature.shape[0] or self.out_adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        feat_list = [feature]

        return [torch.FloatTensor(feat) for feat in feat_list]


# Might include training parameters
class MixPathMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(TwoDirMessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The un direction feature matrices must be tensors!")
            
        return self.combine(feat_list)


def ada_platform_one_step_propagation(adj, x):
    if platform.system() == "Linux":
        one_step_prop_x = csr_sparse_dense_matmul(adj, x)
    else:
        one_step_prop_x = adj.dot(x)
    return one_step_prop_x

def calculate_real_imag_feat(tmp_prop_feat_calculator_out_list):
    real_feat_list = []
    imag_feat_list = []
        
    for k in range(len(tmp_prop_feat_calculator_out_list)):
        tmp_calculator = tmp_prop_feat_calculator_out_list[k]
        if tmp_calculator.i_step & 1 == 0 and tmp_calculator.i_step != 0:
            real_feat_list.append(tmp_calculator.value)
        elif tmp_calculator.i_step == 0:
            real_feat_list.append(tmp_calculator.value)
        else:
            imag_feat_list.append(tmp_calculator.value)

    if len(real_feat_list) != len(imag_feat_list):
        raise RuntimeError("Something wrong!")
    

    for k in range(len(real_feat_list)):
        if k == 0:
            real_feat = real_feat_list[k]
            imag_feat = imag_feat_list[k]
        else:
            real_feat += real_feat_list[k]
            imag_feat += imag_feat_list[k]
    return real_feat, imag_feat