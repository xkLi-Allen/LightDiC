from models.directed_graph_model.s2diconv import S2DiConv


class ModelZoo():
    def __init__(self, logger, args, feat_dim, output_dim, task_level=None):
        super(ModelZoo, self).__init__()
        self.logger = logger
        self.args = args
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.task_level = task_level
        self.q = self.args.edge_q if self.task_level == "edge" else self.args.node_q
        self.log_model()
    def log_model(self):
        if self.args.model_name == "s2diconv":
            self.logger.info(f"model: {self.args.model_name}, prop_steps: {self.args.prop_steps}, r: {self.args.r}, q: {self.q}")

    def model_init(self):
        if self.args.model_name == "s2diconv":
            model = S2DiConv(data_name=self.args.uns_directed_unw_name,prop_steps=self.args.prop_steps, r=self.args.r, q=self.q, 
                        feat_dim=self.feat_dim, edge_dim=self.args.edge_dim, output_dim=self.output_dim, 
                        task_level=self.task_level)
        else:
            return NotImplementedError
        return model