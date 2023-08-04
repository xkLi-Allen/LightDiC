from operators.base_operator import ComMessageOp


class ComLastMessageOp(ComMessageOp):
    def __init__(self):
        super(ComLastMessageOp, self).__init__()
        self.aggr_type = "last"

    def combine(self, real_feat_list, imag_feat_list):
        return real_feat_list[-1], imag_feat_list[-1]
