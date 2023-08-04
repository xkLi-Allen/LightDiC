import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter, Linear
from operators.base_operator import ComMessageOp
from operators.utils import one_dim_weighted_add, two_dim_weighted_add, squeeze_first_dimension


class ComLearnableWeightedMessageOp(ComMessageOp):

    # 'simple' needs one additional parameter 'prop_steps';
    # 'simple_allow_neg' allows negative weights, all else being the same as 'simple';
    # 'gate' needs one additional parameter 'feat_dim';
    # 'ori_ref' needs one additional parameter 'feat_dim';
    # 'jk' needs two additional parameter 'prop_steps' and 'feat_dim'
    def __init__(self, start, end, combination_type, *args):
        super(ComLearnableWeightedMessageOp, self).__init__(start, end)
        self.aggr_type = "learnable_weighted"

        if combination_type not in ["simple", "simple_allow_neg", "gate", "ori_ref", "jk"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'simple', 'simple_allow_neg', 'gate', 'ori_ref' or 'jk'.")
        self.combination_type = combination_type

        self.real_learnable_weight = None
        self.imag_learnable_weight = None
        if combination_type == "simple" or combination_type == "simple_allow_neg":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the simple learnable weighted aggregator!")
            prop_steps = args[0]
            # a 2d tensor is required to use xavier_uniform_.
            tmp_2d_tensor = torch.FloatTensor(1, prop_steps + 1)
            nn.init.xavier_normal_(tmp_2d_tensor)
            self.real_learnable_weight = Parameter(tmp_2d_tensor.view(-1))
            self.imag_learnable_weight = Parameter(tmp_2d_tensor.view(-1))

        elif combination_type == "gate":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the gate learnable weighted aggregator!")
            feat_dim = args[0]
            self.real_learnable_weight = Linear(feat_dim, 1)
            self.imag_learnable_weight = Linear(feat_dim, 1)

        elif combination_type == "ori_ref":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the ori_ref learnable weighted aggregator!")
            feat_dim = args[0]
            self.real_learnable_weight = Linear(feat_dim + feat_dim, 1)
            self.imag_learnable_weight = Linear(feat_dim + feat_dim, 1)

        elif combination_type == "jk":
            if len(args) != 2:
                raise ValueError(
                    "Invalid parameter numbers for the jk learnable weighted aggregator!")
            prop_steps, feat_dim = args[0], args[1]
            self.real_learnable_weight = Linear(feat_dim + (prop_steps + 1) * feat_dim, 1)
            self.imag_learnable_weight = Linear(feat_dim + (prop_steps + 1) * feat_dim, 1)

    def combine(self, real_feat_list, imag_feat_list):
        real_weight_list = None
        imag_weight_list = None
        
        real_feat_list = squeeze_first_dimension(real_feat_list)
        imag_feat_list = squeeze_first_dimension(imag_feat_list)

        if self.combination_type == "simple":
            real_weight_list = F.softmax(torch.sigmoid(self.real_learnable_weight[self.start:self.end]), dim=0)
            imag_weight_list = F.softmax(torch.sigmoid(self.imag_learnable_weight[self.start:self.end]), dim=0)
            
        elif self.combination_type == "simple_allow_neg":
            real_weight_list = self.real_learnable_weight[self.start:self.end]
            imag_weight_list = self.imag_learnable_weight[self.start:self.end]

        elif self.combination_type == "gate":
            adopted_feat_list = torch.vstack(real_feat_list[self.start:self.end])
            real_weight_list = F.softmax(torch.sigmoid(self.real_learnable_weight(adopted_feat_list).view(self.end - self.start, -1).T), dim=1)
            adopted_feat_list = torch.vstack(imag_feat_list[self.start:self.end])
            imag_weight_list = F.softmax(torch.sigmoid(self.imag_learnable_weight(adopted_feat_list).view(self.end - self.start, -1).T), dim=1)

        elif self.combination_type == "ori_ref":
            reference_feat = real_feat_list[0].repeat(self.end - self.start, 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(real_feat_list[self.start:self.end])))
            real_weight_list = F.softmax(torch.sigmoid(self.real_learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)
            reference_feat = imag_feat_list[0].repeat(self.end - self.start, 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(imag_feat_list[self.start:self.end])))
            imag_weight_list = F.softmax(torch.sigmoid(self.imag_learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)

        elif self.combination_type == "jk":
            reference_feat = torch.hstack(real_feat_list).repeat(self.end - self.start, 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(real_feat_list[self.start:self.end])))
            real_weight_list = F.softmax(torch.sigmoid(self.real_learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)
            reference_feat = torch.hstack(imag_feat_list).repeat(self.end - self.start, 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(imag_feat_list[self.start:self.end])))
            imag_weight_list = F.softmax(torch.sigmoid(self.imag_learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)

        else:
            raise NotImplementedError

        real_weighted_feat = None
        imag_weighted_feat = None
        if self.combination_type == "simple" or self.combination_type == "simple_allow_neg":
            real_weighted_feat = one_dim_weighted_add(real_feat_list[self.start:self.end], weight_list=real_weight_list)
            imag_weighted_feat = one_dim_weighted_add(imag_feat_list[self.start:self.end], weight_list=imag_weight_list)
        elif self.combination_type in ["gate", "ori_ref", "jk"]:
            real_weighted_feat = two_dim_weighted_add(real_feat_list[self.start:self.end], weight_list=real_weight_list)
            imag_weighted_feat = two_dim_weighted_add(imag_feat_list[self.start:self.end], weight_list=imag_weight_list)
        else:
            raise NotImplementedError

        return real_weighted_feat, imag_weighted_feat
