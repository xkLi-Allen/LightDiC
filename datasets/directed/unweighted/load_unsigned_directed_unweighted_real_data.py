from datasets.directed.unweighted.UsDUw_pygsd import UsDUwPyGSDDataset


def load_unsigned_directed_unweighted_dataset(logger, args, name, root, k, node_split, node_split_id, edge_split,
                                              edge_split_id):
    dataset = UsDUwPyGSDDataset(args, name, root, k, node_split, node_split_id, edge_split, edge_split_id)
    return dataset