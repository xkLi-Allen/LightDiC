import torch

from operators.base_operator import ComMessageOp


class ComMaxMessageOp(ComMessageOp):
    def __init__(self, start, end):
        super(ComMaxMessageOp, self).__init__(start, end)
        self.aggr_type = "max"

    def combine(self, real_feat_list, imag_feat_list):
        return torch.stack(real_feat_list[self.start:self.end], dim=0).max(dim=0)[0], torch.stack(imag_feat_list[self.start:self.end], dim=0).max(dim=0)[0]