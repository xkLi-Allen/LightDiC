from operators.base_operator import TwoOrderPprApproxMessageOp


class TwoOrderLastMessageOp(TwoOrderPprApproxMessageOp):
    def __init__(self):
        super(TwoOrderLastMessageOp, self).__init__()
        self.aggr_type = "last"

    def combine(self, one_feat_list, two_feat_list):
        return one_feat_list[-1], two_feat_list[-1]
