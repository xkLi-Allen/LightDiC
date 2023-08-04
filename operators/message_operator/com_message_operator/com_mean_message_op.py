from operators.base_operator import ComMessageOp


class ComMeanMessageOp(ComMessageOp):
    def __init__(self, start, end):
        super(ComMeanMessageOp, self).__init__(start, end)
        self.aggr_type = "mean"

    def combine(self, real_feat_list, imag_feat_list):
        return sum(real_feat_list[self.start:self.end]) / (self.end - self.start), sum(imag_feat_list[self.start:self.end]) / (self.end - self.start)
