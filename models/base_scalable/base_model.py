import time
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import scipy.sparse as sp
import torch.nn.functional as F

from models.utils import scipy_sparse_mat_to_torch_sparse_tensor


class SimBaseSGModel(nn.Module):
    def __init__(self):
        super(SimBaseSGModel, self).__init__()
        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

        self.processed_feat_list = None
        self.processed_feature = None
        self.pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self.pre_graph_op is not None:
            self.processed_feat_list = self.pre_graph_op.propagate(
                adj, feature)
            if self.pre_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                self.pre_msg_learnable = True
            else:
                self.pre_msg_learnable = False
                self.processed_feature = self.pre_msg_op.aggregate(
                    self.processed_feat_list)

        else: 
            if self.naive_graph_op is not None:
                self.base_model.adj = self.naive_graph_op.construct_adj(adj)
                if not isinstance(self.base_model.adj, sp.csr_matrix):
                    raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
                elif self.base_model.adj.shape[1] != feature.shape[0]:
                    raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
                self.base_model.adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj)
            self.pre_msg_learnable = False
            self.processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        processed_feature = None
        if self.base_model.adj != None:
            self.base_model.adj = self.base_model.adj.to(device)
            processed_feature = self.processed_feature.to(device)
            if ori is not None: self.base_model.query_edges = ori

        else:
            if idx is None and self.processed_feature is not None: idx = torch.arange(self.processed_feature.shape[0])
            if self.pre_msg_learnable is False:
                processed_feature = self.processed_feature[idx].to(device)
            else:
                transferred_feat_list = [feat[idx].to(
                    device) for feat in self.processed_feat_list]
                processed_feature = self.pre_msg_op.aggregate(
                    transferred_feat_list)
            
        output = self.base_model(processed_feature)
        return output[idx] if self.base_model.query_edges is None and self.base_model.adj != None else output
    

class SimBasePolyModel(nn.Module):
    def __init__(self, conv):
        super(SimBasePolyModel, self).__init__()
        self.conv = conv
        self.naive_graph_op = None
        self.base_model = None
        self.post_graph_op, self.post_msg_op = None, None

    def preprocess(self, adj, feature):
        if self.naive_graph_op is not None:
            if self.conv == "bern":
                self.base_model.adj1, self.base_model.adj2 = self.naive_graph_op.construct_adj(adj)
                if not isinstance(self.base_model.adj1, sp.csr_matrix) or not isinstance(self.base_model.adj2, sp.csr_matrix):
                    raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
                elif self.base_model.adj1.shape[1] != feature.shape[0] or self.base_model.adj2.shape[1] != feature.shape[0]:
                    raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
                self.base_model.adj1 = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj1)
                self.base_model.adj2 = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj2)

            else:
                self.base_model.adj = self.naive_graph_op.construct_adj(adj)
                if not isinstance(self.base_model.adj, sp.csr_matrix):
                    raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
                elif self.base_model.adj.shape[1] != feature.shape[0]:
                    raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
                self.base_model.adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj)
                
        self.naive_graph_op.init_conv_func()
        self.pre_msg_learnable = False
        self.processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        processed_feature = None
        if self.base_model.adj != None or (self.base_model.adj1 != None and self.base_model.adj2 != None):
            if self.base_model.adj != None: self.base_model.adj = self.base_model.adj.to(device)
            else: self.base_model.adj1, self.base_model.adj2 = self.base_model.adj1.to(device), self.base_model.adj2.to(device)
            processed_feature = self.processed_feature.to(device)
            if ori is not None: self.base_model.query_edges = ori
            
        output = self.base_model(self.naive_graph_op.conv_fn, processed_feature)
        return output[idx] if self.base_model.query_edges is None and ((self.base_model.adj != None) or (self.base_model.adj1 != None and self.base_model.adj2 != None)) else output
    

class ComLRBaseSGModel(nn.Module):
    def __init__(self):
        super(ComLRBaseSGModel, self).__init__()
        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

        self.processed_feat_list = None
        self.processed_feature = None
        self.pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self.pre_graph_op is not None:
            self.processed_feature = self.pre_graph_op.propagate(adj, feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        processed_feature = None
        if ori is not None: self.base_model.query_edges = ori
        if idx is None and self.processed_feature is not None: idx = torch.arange(self.processed_feature.shape[0])
        processed_feature = self.processed_feature[idx].to(device)
        output = self.base_model(processed_feature)
        return output[idx] if (self.base_model.query_edges is None and (self.base_model.real_adj != None or self.base_model.imag_adj != None)) else output


class ComBaseSGModel(nn.Module):
    def __init__(self):
        super(ComBaseSGModel, self).__init__()
        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

        self.real_processed_feat_list = None
        self.imag_processed_feat_list = None
        self.real_processed_feature = None
        self.imag_processed_feature = None
        self.pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self.pre_graph_op is not None:
            self.real_processed_feat_list, self.imag_processed_feat_list = \
            self.pre_graph_op.propagate(adj, feature)
            if self.pre_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                self.pre_msg_learnable = True
            else:
                self.pre_msg_learnable = False
                self.real_processed_feature, self.imag_processed_feature = self.pre_msg_op.aggregate(
                    self.real_processed_feat_list, self.imag_processed_feat_list)

        else: 
            if self.naive_graph_op is not None:
                self.base_model.real_adj, self.base_model.imag_adj = self.naive_graph_op.construct_adj(adj)
                if not isinstance(self.base_model.real_adj, sp.csr_matrix) or not isinstance(self.base_model.imag_adj, sp.csr_matrix):
                    raise TypeError("The real/imag adjacency matrix must be a scipy csr sparse matrix!")
                elif self.base_model.real_adj.shape[1] != feature.shape[0] or self.base_model.imag_adj.shape[1] != feature.shape[0]:
                    raise ValueError("Dimension mismatch detected for the real/imag adjacency and the feature matrix!")
                self.base_model.real_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.real_adj)
                self.base_model.imag_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.imag_adj)
            self.pre_msg_learnable = False
            self.real_processed_feature = torch.FloatTensor(feature)
            self.imag_processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        real_processed_feature = None
        imag_processed_feature = None
        if self.base_model.real_adj != None or self.base_model.imag_adj != None:
            self.base_model.real_adj = self.base_model.real_adj.to(device)
            self.base_model.imag_adj = self.base_model.imag_adj.to(device)
            real_processed_feature = self.real_processed_feature.to(device)
            imag_processed_feature = self.imag_processed_feature.to(device)
            if ori is not None: self.base_model.query_edges = ori

        else:
            if idx is None and self.real_processed_feature is not None: 
                idx = torch.arange(self.real_processed_feature.shape[0]) if isinstance(self.real_processed_feature, Tensor) else torch.arange(self.real_processed_feature[0].shape[0]) 
            if self.pre_msg_learnable is False:
                if isinstance(self.real_processed_feature, Tensor):
                    real_processed_feature = self.real_processed_feature[idx].to(device)
                    imag_processed_feature = self.imag_processed_feature[idx].to(device)
                else:
                    real_processed_feature = [feat[idx].to(device) for feat in self.real_processed_feature]
                    imag_processed_feature = [feat[idx].to(device) for feat in self.imag_processed_feature]
            else:
                real_transferred_feat_list = [feat[idx].to(device) for feat in self.real_processed_feature]
                imag_transferred_feat_list = [feat[idx].to(device) for feat in self.imag_processed_feature]
                real_processed_feature, imag_processed_feature = self.pre_msg_op.aggregate(
                    real_transferred_feat_list, imag_transferred_feat_list)
                
        output = self.base_model(real_processed_feature, imag_processed_feature)
        return output[idx] if (self.base_model.query_edges is None and (self.base_model.real_adj != None or self.base_model.imag_adj != None)) else output


class IdentifyBaseSGModel(nn.Module):
    def __init__(self):
        super(IdentifyBaseSGModel, self).__init__()
        self.processed_feature = None
        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

    def preprocess(self, adj, feature): 
        if self.naive_graph_op is not None:
            self.base_model.adj = self.naive_graph_op.construct_adj(adj)
            if not isinstance(self.base_model.adj, sp.csr_matrix):
                raise TypeError("The real/imag adjacency matrix must be a scipy csr sparse matrix!")
            elif self.base_model.adj.shape[1] != feature.shape[0]:
                raise ValueError("Dimension mismatch detected for the un/in/out adjacency and the feature matrix!")
        else:
            raise ValueError("TwoDirBaseSGModel must predefine the graph structure operator!")
        
        self.pre_msg_learnable = False
        self.processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        processed_feature = None
        processed_feature = self.processed_feature.to(device)
        if ori is not None: self.base_model.query_edges = ori
        output = self.base_model(processed_feature, device)
        return output[idx] if self.base_model.query_edges is None else output


class ComBaseMultiPropSGModel(nn.Module):
    def __init__(self):
        super(ComBaseMultiPropSGModel, self).__init__()
        self.pre_graph_op_list, self.pre_msg_op_list = [], []
        self.post_graph_op, self.post_msg_op = None, None
        self.pre_multi_msg_op = None
        self.base_model = None

        self.real_processed_feat_list = None
        self.real_processed_feat_list_list = []
        self.imag_processed_feat_list = None
        self.imag_processed_feat_list_list = []

        self.real_processed_feature = None
        self.real_processed_feature_list = []
        self.imag_processed_feature = None
        self.imag_processed_feature_list = []
        self.pre_msg_learnable_list = []
        self.pre_multi_msg_learnable = None

    def preprocess(self, adj, feature):
        if len(self.pre_graph_op_list) != 0 and (len(self.pre_graph_op_list) == len(self.pre_msg_op_list)):
            for i in range(len(self.pre_graph_op_list)):
                if self.pre_msg_op_list[i].aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                    self.pre_msg_learnable_list.append(True)
                else:
                    self.pre_msg_learnable_list.append(False)

            if not self.pre_msg_learnable_list.count(self.pre_msg_learnable_list[0]) == len(self.pre_msg_learnable_list):
                raise ValueError("In the current version only multi-operator same message aggregation patterns (learned, unlearnable) are supported!")
            
            if self.pre_msg_op_list[0].aggr_type == "identity" and all(_ == "identity" for _ in self.pre_msg_learnable_list) is False:
                raise ValueError("In the current version the mix of identity mapping operators and other operators is not supported!")

            for i in range(len(self.pre_graph_op_list)):
                self.real_processed_feat_list, self.imag_processed_feat_list = self.pre_graph_op_list[i].propagate(adj, feature)
                self.real_processed_feat_list_list.append(self.real_processed_feat_list)
                self.imag_processed_feat_list_list.append(self.imag_processed_feat_list)

                if self.pre_msg_learnable_list[i] is False:
                    self.real_processed_feature, self.imag_processed_feature = self.pre_msg_op_list[i].aggregate(self.real_processed_feat_list_list[-1], self.imag_processed_feat_list_list[-1])
                    if self.pre_msg_op_list[i].aggr_type in ["identity"]:
                        self.real_processed_feature_list.extend(self.real_processed_feature)
                        self.imag_processed_feature_list.extend(self.imag_processed_feature)
                    else:
                        self.real_processed_feature_list.append(self.real_processed_feature)
                        self.imag_processed_feature_list.append(self.imag_processed_feature)

            if self.pre_multi_msg_op is not None: 
                if self.pre_multi_msg_op.aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                    self.pre_multi_msg_learnable = True
                else:
                    self.pre_multi_msg_learnable = False
                    if self.pre_msg_op_list[0].aggr_type == "identity":
                        self.real_processed_feature, self.imag_processed_feature =  self.pre_multi_msg_op.aggregate(self.real_processed_feature_list, self.imag_processed_feature_list)

            else:
                self.pre_multi_msg_learnable = False
        else:
            raise ValueError("MultiProp must define One-to-One propagation operator!")
        
    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError("Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        real_processed_feature = None
        imag_processed_feature = None
            
        # f f
        if idx is None and self.real_processed_feature is not None: idx = torch.arange(self.real_processed_feature.shape[0])
        if all(_ is True for _ in self.pre_msg_learnable_list) is False and self.pre_multi_msg_learnable is False:
            real_processed_feature = self.real_processed_feature[idx].to(device)
            imag_processed_feature = self.imag_processed_feature[idx].to(device)
        # f t
        elif all(_ is True for _ in self.pre_msg_learnable_list) is False and self.pre_multi_msg_learnable is True:
            real_multi_transferred_feat_list = [feat[idx].to(device) for feat in self.real_processed_feature_list]
            imag_multi_transferred_feat_list = [feat[idx].to(device) for feat in self.imag_processed_feature_list]
            real_processed_feature, imag_processed_feature = self.pre_multi_msg_op.aggregate(real_multi_transferred_feat_list, imag_multi_transferred_feat_list)
        # t f / t t
        else:
            self.real_processed_feature_list = []
            self.imag_processed_feature_list = []
            for i in range(len(self.real_processed_feat_list_list)):
                self.pre_msg_op_list[i] =  self.pre_msg_op_list[i].to(device)
                real_transferred_feat_list = [feat[idx].to(device) for feat in self.real_processed_feat_list_list[i]]
                imag_transferred_feat_list = [feat[idx].to(device) for feat in self.imag_processed_feat_list_list[i]]
                real_processed_feature, imag_processed_feature = self.pre_msg_op_list[i].aggregate(real_transferred_feat_list, imag_transferred_feat_list)
                self.real_processed_feature_list.append(real_processed_feature)
                self.imag_processed_feature_list.append(imag_processed_feature)
            real_processed_feature, imag_processed_feature = self.pre_multi_msg_op.aggregate(self.real_processed_feature_list, self.imag_processed_feature_list)
        
        output = self.base_model(real_processed_feature, imag_processed_feature)
        
        return output
    

class TwoOrderBaseSGModel(nn.Module):
    def __init__(self):
        super(TwoOrderBaseSGModel, self).__init__()
        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

        self.one_processed_feat_list = None
        self.two_processed_feat_list = None
        self.one_processed_feature = None
        self.two_processed_feature = None

        self.pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self.naive_graph_op is not None:
            self.base_model.one_adj, self.base_model.two_adj = self.naive_graph_op.construct_adj(adj)
            if not isinstance(self.base_model.one_adj, sp.csr_matrix) or not isinstance(self.base_model.two_adj, sp.csr_matrix):
                raise TypeError("The real/imag adjacency matrix must be a scipy csr sparse matrix!")
            elif self.base_model.one_adj.shape[1] != feature.shape[0] or self.base_model.two_adj.shape[1] != feature.shape[0]:
                raise ValueError("Dimension mismatch detected for the un/in/out adjacency and the feature matrix!")
            self.base_model.one_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.one_adj)
            self.base_model.two_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.two_adj)
        else:
            raise ValueError("TwoOrderBaseSGModel must predefine the graph structure operator!")
        
        self.pre_msg_learnable = False
        self.original_feature = torch.FloatTensor(feature)
        self.one_processed_feature = torch.FloatTensor(feature)
        self.two_processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        one_processed_feature = None
        two_processed_feature = None
        self.base_model.one_adj = self.base_model.one_adj.to(device)
        self.base_model.two_adj = self.base_model.two_adj.to(device)
        one_processed_feature = self.one_processed_feature.to(device)
        two_processed_feature = self.two_processed_feature.to(device)
        original_feature = self.original_feature.to(device)
        if ori is not None: self.base_model.query_edges = ori
        output = self.base_model(original_feature, one_processed_feature, two_processed_feature)
        return output[idx] if self.base_model.query_edges is None else output
    

class TwoDirBaseSGModel(nn.Module):
    def __init__(self):
        super(TwoDirBaseSGModel, self).__init__()
        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

        self.un_processed_feat_list = None
        self.in_processed_feat_list = None
        self.out_processed_feat_list = None
        self.un_processed_feature = None
        self.in_processed_feature = None
        self.out_processed_feature = None
        self.pre_msg_learnable = False

    def preprocess(self, adj, feature): 
        if self.naive_graph_op is not None:
            self.base_model.un_adj, self.base_model.in_adj, self.base_model.out_adj = self.naive_graph_op.construct_adj(adj)
            if not isinstance(self.base_model.un_adj, sp.csr_matrix) or not isinstance(self.base_model.in_adj, sp.csr_matrix) or not isinstance(self.base_model.out_adj, sp.csr_matrix):
                raise TypeError("The real/imag adjacency matrix must be a scipy csr sparse matrix!")
            elif self.base_model.un_adj.shape[1] != feature.shape[0] or self.base_model.in_adj.shape[1] != feature.shape[0] or self.base_model.out_adj.shape[1] != feature.shape[0]:
                raise ValueError("Dimension mismatch detected for the un/in/out adjacency and the feature matrix!")
            self.base_model.un_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.un_adj)
            self.base_model.in_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.in_adj)
            self.base_model.out_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.out_adj)
        else:
            raise ValueError("TwoDirBaseSGModel must predefine the graph structure operator!")
        
        self.pre_msg_learnable = False
        self.un_processed_feature = torch.FloatTensor(feature)
        self.in_processed_feature = torch.FloatTensor(feature)
        self.out_processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        un_processed_feature = None
        in_processed_feature = None
        out_processed_feature = None
        self.base_model.un_adj = self.base_model.un_adj.to(device)
        self.base_model.in_adj = self.base_model.in_adj.to(device)
        self.base_model.out_adj = self.base_model.out_adj.to(device)
        un_processed_feature = self.un_processed_feature.to(device)
        in_processed_feature = self.in_processed_feature.to(device)
        out_processed_feature = self.out_processed_feature.to(device)
        if ori is not None: self.base_model.query_edges = ori
        output = self.base_model(un_processed_feature, in_processed_feature, out_processed_feature)
        return output[idx] if self.base_model.query_edges is None else output


class MixPathBaseSGModel(nn.Module):
    def __init__(self):
        super(MixPathBaseSGModel, self).__init__()
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

    def preprocess(self, adj, feature): 
        if self.naive_graph_op is not None:
            self.base_model.adj, self.base_model.adj_t, source_index, target_index = self.naive_graph_op.construct_adj(adj)
            if not isinstance(self.base_model.adj, sp.csr_matrix) or not isinstance(self.base_model.adj_t, sp.csr_matrix):
                raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
            elif self.base_model.adj.shape[1] != feature.shape[0] or self.base_model.adj_t.shape[1] != feature.shape[0]:
                raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
            self.base_model.adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj)
            self.base_model.adj_t = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj_t)
        self.pre_msg_learnable = False
        self.processed_souce_feature = torch.FloatTensor(feature[source_index])
        self.processed_target_feature = torch.FloatTensor(feature[target_index])

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        self.base_model.adj = self.base_model.adj.to(device)
        self.base_model.adj_t = self.base_model.adj_t.to(device)
        processed_souce_feature = self.processed_souce_feature.to(device)
        processed_target_feature = self.processed_target_feature.to(device)
        if ori is not None: self.base_model.query_edges = ori
        output = self.base_model(processed_souce_feature, processed_target_feature)
        return output[idx] if self.base_model.query_edges is None else output