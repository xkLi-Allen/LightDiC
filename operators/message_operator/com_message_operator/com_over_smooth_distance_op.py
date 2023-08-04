import torch
import torch.nn.functional as F

from operators.base_operator import ComMessageOp

class ComOverSmoothDistanceWeightedOp(ComMessageOp):
    def __init__(self):
        super(ComOverSmoothDistanceWeightedOp, self).__init__()
        self.aggr_type = 'over_smooth_dis_weighted'

    def combine(self, real_feat_list, imag_feat_list):
        real_weight_list = []
        real_features = real_feat_list[0]
        real_norm_fea = torch.norm(real_features, 2, 1).add(1e-10)
        for real_fea in real_feat_list:
            norm_cur = torch.norm(real_fea, 2, 1).add(1e-10)
            real_tmp = torch.div((real_features * real_fea).sum(1), norm_cur)
            real_tmp = torch.div(real_tmp, real_norm_fea)

            real_weight_list.append(real_tmp.unsqueeze(-1))

        real_weight = F.softmax(torch.cat(real_weight_list, dim=1), dim=1)

        real_hops = len(real_feat_list)
        num_nodes = real_features.shape[0]
        real_output = []
        for i in range(num_nodes):
            real_fea = 0.
            for j in range(real_hops):
                real_fea += (real_weight[i][j]*real_feat_list[j][i]).unsqueeze(0)
            real_output.append(real_fea)
        real_output = torch.cat(real_output, dim=0)

        imag_weight_list = []
        imag_features = imag_feat_list[0]
        imag_norm_fea = torch.norm(imag_features, 2, 1).add(1e-10)
        for imag_fea in imag_feat_list:
            norm_cur = torch.norm(imag_fea, 2, 1).add(1e-10)
            imag_tmp = torch.div((imag_features * imag_fea).sum(1), norm_cur)
            imag_tmp = torch.div(imag_tmp, imag_norm_fea)

            imag_weight_list.append(imag_tmp.unsqueeze(-1))

        imag_weight = F.softmax(torch.cat(imag_weight_list, dim=1), dim=1)

        imag_hops = len(imag_feat_list)
        num_nodes = imag_features.shape[0]
        imag_output = []
        for i in range(num_nodes):
            imag_fea = 0.
            for j in range(imag_hops):
                imag_fea += (imag_weight[i][j]*imag_feat_list[j][i]).unsqueeze(0)
            imag_output.append(imag_fea)
        imag_output = torch.cat(imag_output, dim=0)
        return real_output, imag_output