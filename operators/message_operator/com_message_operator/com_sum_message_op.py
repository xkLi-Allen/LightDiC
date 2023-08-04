from operators.base_operator import ComMessageOp


class ComSumMessageOp(ComMessageOp):
    def __init__(self, start, end):
        super(ComSumMessageOp, self).__init__(start, end)
        self.aggr_type = "sum"
    
    def combine(self, real_feat_list, imag_feat_list):
        return sum(real_feat_list[self.start:self.end]), sum(imag_feat_list[self.start:self.end])
