import torch
from operators.base_operator import ComMessageOp


class ComConcatMessageOp(ComMessageOp):
    def __init__(self, start, end):
        super(ComConcatMessageOp, self).__init__(start, end)
        self.aggr_type = "concat"

    def combine(self, real_feat_list, imag_feat_list):
        return torch.hstack(real_feat_list[self.start:self.end]), torch.hstack(imag_feat_list[self.start:self.end])