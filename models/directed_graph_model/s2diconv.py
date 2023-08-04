from models.base_scalable.base_model import ComBaseSGModel
from models.base_scalable.complex_models import ComLogisticRegression
from operators.message_operator.com_message_operator.com_last_message_op import ComLastMessageOp  
from operators.message_operator.com_message_operator.com_identity_message_op import ComIdentityMessageOp
from operators.graph_operator.symmetrical_directed_magnetic_laplacian_operator import SymDirMagLaplacianGraphOp 
 

class S2DiConv(ComBaseSGModel):
    def __init__(self, data_name, prop_steps, r, q, feat_dim, edge_dim, output_dim, task_level):
        super(S2DiConv, self).__init__()
        
        self.pre_graph_op = SymDirMagLaplacianGraphOp("dirsgc", data_name, prop_steps, r=r, q=q)
        self.pre_msg_op = ComLastMessageOp()
        self.base_model = ComLogisticRegression(feat_dim, edge_dim, output_dim, task_level)
