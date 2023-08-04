import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from operators.utils import zeros, glorot, squeeze_first_dimension, two_dim_weighted_add
 

class SimOneDimConvolution(nn.Module):
    def __init__(self, num_subgraphs, prop_steps, feat_dim):
        super(SimOneDimConvolution, self).__init__()
        self.adj = None
        self.hop_num = prop_steps
        self.learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            self.learnable_weight.append(nn.Parameter(
                torch.FloatTensor(feat_dim, num_subgraphs)))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.learnable_weight:
            nn.init.xavier_uniform_(weight)

    # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (
                    adopted_feat * (self.learnable_weight[i].unsqueeze(dim=0))).mean(dim=2)

            aggregated_feat_list.append(intermediate_feat)

        return aggregated_feat_list


class SimOneDimConvolutionWeightSharedAcrossFeatures(nn.Module):
    def __init__(self, num_subgraphs, prop_steps):
        super(SimOneDimConvolutionWeightSharedAcrossFeatures, self).__init__()
        self.adj = None
        self.hop_num = prop_steps
        self.learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            # To help xvarient_uniform_ calculate fan in and fan out, "1" should be kept here.
            self.learnable_weight.append(nn.Parameter(
                torch.FloatTensor(1, num_subgraphs)))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.learnable_weight:
            nn.init.xavier_uniform_(weight)

    # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (
                    adopted_feat * (self.learnable_weight[i])).mean(dim=2)

            aggregated_feat_list.append(intermediate_feat)

        return aggregated_feat_list


class SimFastOneDimConvolution(nn.Module):
    def __init__(self, num_subgraphs, prop_steps):
        super(SimFastOneDimConvolution, self).__init__()
        self.adj = None
        self.num_subgraphs = num_subgraphs
        self.prop_steps = prop_steps

        # How to initialize the weight is extremely important.
        # Pure xavier will lead to extremely unstable accuracy.
        # Initialized with ones will not perform as good as this one.        
        self.learnable_weight = nn.Parameter(
            torch.ones(num_subgraphs * prop_steps, 1))

    # feat_list_list: 3-d tensor (num_node, feat_dim, num_subgraphs * prop_steps)
    def forward(self, feat_list_list):
        return (feat_list_list @ self.learnable_weight).squeeze(dim=2)

    def subgraph_weight(self):
        return self.learnable_weight.view(
            self.num_subgraphs, self.prop_steps).sum(dim=1)


class SimIdenticalMapping(nn.Module):
    def __init__(self) -> None:
        super(SimIdenticalMapping, self).__init__()
        self.adj = None

    def forward(self, feature):
        return feature


class SimLogisticRegression(nn.Module):
    def __init__(self, feat_dim, edge_dim, output_dim, dropout, task_level):
        super(SimLogisticRegression, self).__init__()
        self.adj = None
        self.query_edges = None
        self.dropout = nn.Dropout(dropout)
        if task_level == "edge": self.fc_node_edge = nn.Linear(feat_dim, edge_dim) 
        else: self.fc_node_edge = nn.Linear(feat_dim, output_dim)
        self.linear = nn.Linear(2*edge_dim, output_dim)

    def forward(self, feature):
        feature = squeeze_first_dimension(feature)
        if  self.query_edges == None:
            output = self.fc_node_edge(feature)
        else:
            x = self.fc_node_edge(feature)
            x = torch.cat((x[self.query_edges[:, 0]], x[self.query_edges[:, 1]]), dim=-1)
            x = self.dropout(x)
            output = self.linear(x)
        return output


class SimMultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout, bn=False):
        super(SimMultiLayerPerceptron, self).__init__()
        self.adj = None
        self.query_edges = None
        if num_layers < 2:
            raise ValueError("MLP must have at least two layers!")
        self.num_layers = num_layers
        self.fcs_node_edge = nn.ModuleList()
        self.fcs_node_edge.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs_node_edge.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs_node_edge.append(nn.Linear(hidden_dim, output_dim))
        self.linear = nn.Linear(2*output_dim, output_dim)

        self.bn = bn
        if self.bn is True:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs_node_edge:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            nn.init.zeros_(fc.bias)

    def forward(self, feature):
        for i in range(self.num_layers - 1):
            feature = self.fcs_node_edge[i](feature)
            if self.bn is True:
                feature = self.bns[i](feature)
            feature = self.prelu(feature)
            feature = self.dropout(feature)
        if  self.query_edges == None:
            output = self.fcs_node_edge[-1](feature)
        else:
            x = torch.cat((feature[self.query_edges[:, 0]], feature[self.query_edges[:, 1]]), dim=-1)
            x = self.dropout(x)
            output = self.linear(x)
            
        return output


class SimResMultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.8, bn=False):
        super(SimResMultiLayerPerceptron, self).__init__()
        self.adj = None
        self.query_edges = None
        if num_layers < 2:
            raise ValueError("ResMLP must have at least two layers!")
        self.num_layers = num_layers
        self.fcs_node_edge = nn.ModuleList()
        self.fcs_node_edge.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs_node_edge.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs_node_edge.append(nn.Linear(hidden_dim, output_dim))
        self.linear = nn.Linear(2*output_dim, output_dim)

        self.bn = bn
        if self.bn is True:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, feature):
        feature = self.dropout(feature)
        feature = self.fcs_node_edge[0](feature)
        if self.bn is True:
            feature = self.bns[0](feature)
        feature = self.relu(feature)
        residual = feature

        for i in range(1, self.num_layers - 1):
            feature = self.dropout(feature)
            feature = self.fcs_node_edge[i](feature)
            if self.bn is True:
                feature = self.bns[i](feature)
            feature_ = self.relu(feature)
            feature = feature_ + residual
            residual = feature_
        feature = self.dropout(feature)
        if  self.query_edges == None:
            output = self.fcs_node_edge[-1](feature)
        else:
            x = torch.cat((feature[self.query_edges[:, 0]], feature[self.query_edges[:, 1]]), dim=-1)
            output = self.linear(x)
        return output


class Sim2LayerGraphConvolution(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(Sim2LayerGraphConvolution, self).__init__()
        self.adj = None
        self.query_edges = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1_node_edge = nn.Linear(feat_dim, hidden_dim)
        self.fc2_node_edge = nn.Linear(hidden_dim, output_dim)
        self.linear = nn.Linear(2*output_dim, output_dim)

    def forward(self, feature):
        x = feature
        x = self.fc1_node_edge(x)
        x = torch.mm(self.adj, x)
        x = self.relu(x)
        x = self.dropout(x)
        if  self.query_edges == None:
            x = self.fc2_node_edge(x)
            output = torch.mm(self.adj, x) 
        else:
            x = self.fc2_node_edge(x)
            x = torch.mm(self.adj, x) 
            x = torch.cat((x[self.query_edges[:, 0]], x[self.query_edges[:, 1]]), dim=-1)
            output = self.linear(x)  
        return output
    

class Sim2LayerPolyConvolution(nn.Module):
    def __init__(self, conv, init_poly_coeff, poly_order, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(Sim2LayerPolyConvolution, self).__init__()
        self.adj = None
        self.adj1 = None
        self.adj2 = None
        self.query_edges = None
        self.conv = conv
        self.poly_order = poly_order
        self.init_poly_coeff = init_poly_coeff
        self.poly_coeff = nn.ParameterList([nn.Parameter(torch.tensor(float(min(1 / self.init_poly_coeff, 1))),requires_grad=True) for i in range(self.poly_order + 1)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1_node_edge = nn.Linear(feat_dim, hidden_dim)
        self.fc2_node_edge = nn.Linear(hidden_dim, output_dim)
        self.linear = nn.Linear(2*output_dim, output_dim)
        self.learnable_weight =  nn.Linear(output_dim + (poly_order + 1) * output_dim, 1)

    def forward(self, conv_fn, feature):
        x = self.dropout(feature)
        x = self.fc1_node_edge(x)
        x = self.relu(x)
        x = self.fc2_node_edge(x)
        x = self.dropout(x)
        poly_coeff = [self.init_poly_coeff * torch.tanh(_) for _ in self.poly_coeff]

        if self.conv != "bern":
            xs = [conv_fn(0, [x], self.adj, poly_coeff)]
            for L in range(1, self.poly_order + 1):
                tx = conv_fn(L, xs, self.adj, poly_coeff)
                xs.append(tx)
            reference_feat = torch.hstack(xs).repeat(len(xs), 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(xs[0:len(xs)])))
            weight_list = F.softmax(torch.sigmoid(self.learnable_weight(adopted_feat_list).view(-1, len(xs))), dim=1)
            x = two_dim_weighted_add(xs[0:len(xs)], weight_list=weight_list)
        
        else:
            x = conv_fn(self.poly_order, x, self.adj1, self.adj2, poly_coeff)
        
        if  self.query_edges == None:
            output = x
        else:
            x = torch.cat((x[self.query_edges[:, 0]], x[self.query_edges[:, 1]]), dim=-1)
            output = self.linear(x)  
        
        return output
 

class Sim2LayerGeneralizedPageRank(nn.Module):
    def __init__(self, gpr_alpha, poly_order, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(Sim2LayerGeneralizedPageRank, self).__init__()
        self.adj = None
        self.query_edges = None
        self.gpr_alpha = gpr_alpha
        self.poly_order = poly_order
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1_node_edge = nn.Linear(feat_dim, hidden_dim)
        self.fc2_node_edge = nn.Linear(hidden_dim, output_dim)
        self.linear = nn.Linear(2*output_dim, output_dim)
        self.message_ppr = self.gpr_alpha*(1-self.gpr_alpha)**np.arange(self.poly_order+1)
        self.message_ppr[-1] = (1-self.gpr_alpha)**self.poly_order
        self.message_weight = nn.Parameter(torch.tensor(self.message_ppr))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.message_weight)
        for k in range(self.poly_order+1):
            self.message_weight.data[k] = self.gpr_alpha*(1-self.gpr_alpha)**k
        self.message_weight.data[-1] = (1-self.gpr_alpha)**self.poly_order

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.fc1_node_edge(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2_node_edge(x)
        x = self.dropout(x)

        hidden = x*(self.message_weight[0])
        for k in range(self.poly_order):
            x = self.adj @ x
            gamma = self.message_weight[k+1]
            hidden = hidden + gamma*x

        if  self.query_edges == None:
            output = hidden
        else:
            x = torch.cat((hidden[self.query_edges[:, 0]], hidden[self.query_edges[:, 1]]), dim=-1)
            output = self.linear(x)  
        
        return output
    

class Sim2LayerARMAGraphConv(nn.Module):
    def __init__(self, poly_order, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(Sim2LayerARMAGraphConv, self).__init__()
        self.adj = None
        self.query_edges = None
        self.poly_order = poly_order
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(2*output_dim, output_dim)

        self.fc1_node_edge_init = nn.ModuleDict({str(k): nn.Linear(feat_dim, hidden_dim, bias=False) for k in range(self.poly_order)})
        self.fc1_node_edge_deep_v = nn.ModuleDict({str(k): nn.Linear(feat_dim, hidden_dim, bias=False) for k in range(self.poly_order)})
        self.bias1 = nn.Parameter(torch.Tensor(self.poly_order, 1, 1, hidden_dim))

        self.fc2_node_edge_init = nn.ModuleDict({str(k): nn.Linear(hidden_dim, output_dim, bias=False) for k in range(self.poly_order)})
        self.fc2_node_edge_deep_v = nn.ModuleDict({str(k): nn.Linear(hidden_dim, output_dim, bias=False) for k in range(self.poly_order)})
        self.bias2 = nn.Parameter(torch.Tensor(self.poly_order, 1, 1, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.poly_order):
            glorot(self.fc1_node_edge_init[str(k)].weight)
            glorot(self.fc1_node_edge_deep_v[str(k)].weight)
            glorot(self.fc2_node_edge_init[str(k)].weight)
            glorot(self.fc2_node_edge_deep_v[str(k)].weight)
        zeros(self.bias1)
        zeros(self.bias2)

    def forward(self, feature):
        init_x = feature
        output = None
        for k in range(self.poly_order):
            x = torch.mm(self.adj, feature)
            x = self.fc1_node_edge_init[str(k)](x)
            x = x + self.dropout(self.fc1_node_edge_deep_v[str(k)](init_x))
            x = x + self.fc1_node_edge_deep_v[str(k)](self.dropout(init_x))
            x = x + self.bias1[k][0]
            x = self.relu(x)
            if output is None:
                output = x
            else:
                output = output + x
        output = output / self.poly_order 

        output = self.dropout(output)

        feature = output
        init_x = feature
        output = None
        for k in range(self.poly_order):
            x = torch.mm(self.adj, feature)
            x = self.fc2_node_edge_init[str(k)](x)
            x = x + self.dropout(self.fc2_node_edge_deep_v[str(k)](init_x))
            x = x + self.fc2_node_edge_deep_v[str(k)](self.dropout(init_x))
            x = x + self.bias2[k][0]
            x = self.relu(x)
            if output is None:
                output = x
            else:
                output = output + x
        output = output / self.poly_order 

        if  self.query_edges == None:
            output = output
        else:
            x = torch.cat((output[self.query_edges[:, 0]], output[self.query_edges[:, 1]]), dim=-1)
            output = self.linear(x)  
        
        return output