from operators.base_operator import ComMessageOp


class ComIdentityMessageOp(ComMessageOp):
    def __init__(self):
        super(ComIdentityMessageOp, self).__init__()
        self.aggr_type = "identity"

    def combine(self, real_feat_list, imag_feat_list):
        return real_feat_list, imag_feat_list
