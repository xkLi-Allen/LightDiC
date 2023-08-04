import os
import time
import torch
import argparse
import datetime
import warnings

from logger import Logger
from utils import seed_everything
from models.model_init import ModelZoo
from configs.data_config import add_data_config
from configs.model_config import add_model_config
from configs.training_config import add_training_config 
from tasks.link_classification import LinkClassification
from tasks.node_classification import NodeClassification
from datasets.directed.unweighted.load_unsigned_directed_unweighted_real_data import load_unsigned_directed_unweighted_dataset


warnings.filterwarnings("ignore")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    add_data_config(parser)
    add_model_config(parser)
    add_training_config(parser)
    args = parser.parse_args()

    dataset_name = args.uns_directed_unw_name
    method_name = args.model_name

    run_id = f"dropout={args.dropout}, prop_steps={args.prop_steps}, \
        node_q={args.node_q}, edge_q={args.edge_q},\
        lr={args.lr}, weight_decay={args.weight_decay}"
    
    # log_dir = os.path.join("log", method_name, dataset_name, args.uns_directed_unw_node_split)
    log_dir = os.path.join("log", method_name, dataset_name, args.uns_directed_unw_edge_split)
    logger_name = os.path.join(log_dir, run_id + ".log")
    logger = Logger(logger_name)

    logger.info(f"program start: {datetime.datetime.now()}")

    # set up seed
    logger.info(f"random seed: {args.seed}")
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu_id) if (args.use_cuda and torch.cuda.is_available()) else 'cpu')

    # set up datasets
    set_up_datasets_start_time = time.time()
    logger.info(f"Load unsigned & directed & unweighted network: {args.uns_directed_unw_name}")
    dataset = load_unsigned_directed_unweighted_dataset(logger, args, name=args.uns_directed_unw_name, root=args.uns_directed_unw_root, k=args.uns_directed_unw_dimension_k,
                                                    node_split=args.uns_directed_unw_node_split, edge_split=args.uns_directed_unw_edge_split,
                                                    node_split_id=args.uns_directed_unw_node_split_id, edge_split_id=args.uns_directed_unw_edge_split_id)
    set_up_datasets_end_time = time.time()
    
    # logger.info(f"datasets: {args.uns_directed_unw_name}, root dir: {args.uns_directed_unw_root}, node-level split method: {args.uns_directed_unw_node_split}, id: {args.uns_directed_unw_node_split_id}, edge-level split method: {args.uns_directed_unw_edge_split}, id: {args.uns_directed_unw_edge_split_id}, the running time is: {round(set_up_datasets_end_time-set_up_datasets_start_time,4)}s")
    # logger.info(f"num_epochs: {args.num_epochs}, lr: {args.lr}, weight_decay: {args.weight_decay}")
    
    # logger.info(f"dataset.x.shape: {dataset.x.shape}")
    # label_info = max(dataset.y)+1 if dataset.y != None else -1
    # logger.info(f"max(dataset.y)+1: {label_info}")
    # logger.info(f"dataset.num_node: {dataset.num_node}")
    # logger.info(f"Real edges -> dataset.num_edge: {dataset.num_edge}")
    # logger.info(f"min(dataset.adj.data): {min(dataset.adj.data)}")
    # logger.info(f"max(dataset.adj.data): {max(dataset.adj.data)}")
    # logger.info(f"dataset.adj.data: {dataset.adj.data}")

    if args.uns_directed_unw_name not in ("wikitalk", "slashdot", "epinions"):
        model_zoo = ModelZoo(logger, args, feat_dim=dataset.num_features, output_dim=dataset.num_node_classes)
        run = NodeClassification(logger, dataset, model_zoo, normalize_times=args.normalize_times, lr=args.lr, weight_decay=args.weight_decay, epochs=args.num_epochs, early_stop=args.early_stop, device=device)

    model_zoo = ModelZoo(logger, args, feat_dim=dataset.num_features, output_dim=dataset.num_edge_classes, task_level="edge")
    run = LinkClassification(logger, dataset, model_zoo, normalize_times=args.normalize_times, lr=args.lr, weight_decay=args.weight_decay, epochs=args.num_epochs, early_stop=args.early_stop, device=device)

