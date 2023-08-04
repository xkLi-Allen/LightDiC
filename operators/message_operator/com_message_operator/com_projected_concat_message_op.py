import torch
import torch.nn.functional as F

from torch.nn import ModuleList
from operators.base_operator import ComMessageOp
from operators.utils import squeeze_first_dimension
from models.base_scalable.simple_models import SimMultiLayerPerceptron  


class ComProjectedConcatMessageOp(ComMessageOp):
    def __init__(self, start, end, feat_dim, hidden_dim, num_layers, dropout):
        super(ComProjectedConcatMessageOp, self).__init__(start, end)
        self.aggr_type = "proj_concat"

        self.real_learnable_weight = ModuleList()
        for _ in range(end - start):
            self.real_learnable_weight.append(SimMultiLayerPerceptron(
                feat_dim, hidden_dim, num_layers, hidden_dim, dropout))

        self.imag_learnable_weight = ModuleList()
        for _ in range(end - start):
            self.imag_learnable_weight.append(SimMultiLayerPerceptron(
                feat_dim, hidden_dim, num_layers, hidden_dim, dropout))

    def combine(self, real_feat_list, imag_feat_list):
        real_feat_list = squeeze_first_dimension(real_feat_list)
        imag_feat_list = squeeze_first_dimension(imag_feat_list)

        real_adopted_feat_list = real_feat_list[self.start:self.end]
        real_concat_feat = self.real_learnable_weight[0](real_adopted_feat_list[0])
        imag_adopted_feat_list = imag_feat_list[self.start:self.end]
        imag_concat_feat = self.imag_learnable_weight[0](imag_adopted_feat_list[0])
        
        for i in range(1, self.end - self.start):
            real_transformed_feat = F.relu(
                self.real_learnable_weight[i](real_adopted_feat_list[i]))
            real_concat_feat = torch.hstack((real_concat_feat, real_transformed_feat))
        for i in range(1, self.end - self.start):
            imag_transformed_feat = F.relu(
                self.imag_learnable_weight[i](imag_adopted_feat_list[i]))
            imag_concat_feat = torch.hstack((imag_concat_feat, imag_transformed_feat))
        return real_concat_feat, imag_concat_feat
