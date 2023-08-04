from operators.base_operator import TwoDirMessageOp


class TwoDirLastMessageOp(TwoDirMessageOp):
    def __init__(self):
        super(TwoDirLastMessageOp, self).__init__()
        self.aggr_type = "last"

    def combine(self, un_feat_list, in_feat_list, out_feat_list):
        return un_feat_list[-1], in_feat_list[-1], out_feat_list[-1]
