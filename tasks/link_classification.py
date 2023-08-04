import time

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.optim import Adam
from torch.utils.data import DataLoader

from tasks.base_task import BaseTask
from tasks.utils import accuracy, link_cls_evaluate, link_cls_train, link_cls_mini_batch_train, \
    link_cls_mini_batch_evaluate


class LinkClassification(BaseTask):
    def __init__(self, logger, dataset, model, normalize_times,
                 lr, weight_decay, epochs, early_stop, device, loss_fn=nn.CrossEntropyLoss(),
                 train_batch_size=None, eval_batch_size=None):
        super(LinkClassification, self).__init__()
        self.logger = logger
        self.normalize_times = normalize_times
        self.normalize_record = {"val_acc": [], "test_acc": []}

        dataset.adj = csr_matrix((dataset.observed_edge_weight, (dataset.observed_edge_idx[0], dataset.observed_edge_idx[1])), shape=(dataset.num_node, dataset.num_node))
        self.dataset = dataset

        self.train_query_edges = self.dataset.train_edge_pairs_idx
        self.train_labels = self.dataset.train_edge_pairs_label
        self.val_query_edges = self.dataset.val_edge_pairs_idx
        self.val_labels = self.dataset.val_edge_pairs_label
        self.test_query_edges = self.dataset.test_edge_pairs_idx
        self.test_labels = self.dataset.test_edge_pairs_label

        self.model_zoo = model
        self.model = model.model_init()
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.early_stop = early_stop
        self.loss_fn = loss_fn
        self.device = device

        self.mini_batch = False
        if train_batch_size is not None:
            self.mini_batch = True
            logger.info(f"Mini-batch training size: {train_batch_size}, eval and test size: {eval_batch_size}")
            self.train_loader = DataLoader(
                range(self.train_query_edges.shape[0]), batch_size=train_batch_size, shuffle=True, drop_last=False)
            self.val_loader = DataLoader(
                range(self.val_query_edges.shape[0]), batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.test_loader = DataLoader(
                range(self.test_query_edges.shape[0]), batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.all_eval_loader = DataLoader(
                range(self.dataset.num_node), batch_size=eval_batch_size, shuffle=False, drop_last=False)

        for i in range(self.normalize_times):
            if i == 0: normalize_times_st = time.time()
            else: 
                self.model = self.model_zoo.model_init()
                self.optimizer = Adam(self.model.parameters(), lr=lr,
                        weight_decay=weight_decay)
            self.acc = self.execute()
        
        if self.normalize_times > 1:
            logger.info("Optimization Finished!")
            logger.info("Total training time is: {:.4f}s".format(time.time() -   normalize_times_st))
            logger.info("Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(
                round(np.mean(self.normalize_record["val_acc"]), 4),
                round(np.std(self.normalize_record["val_acc"], ddof=1), 4),
                round(np.mean(self.normalize_record["test_acc"]), 4),
                round(np.std(self.normalize_record["test_acc"], ddof=1), 4)))

    def execute(self):
        pre_time_st = time.time()
        self.model.preprocess(self.dataset.adj, self.dataset.x)
        pre_time_ed = time.time()

        if self.normalize_times == 1:
            self.logger.info(f"Preprocessing done in {(pre_time_ed - pre_time_st):.4f}s")

        self.model = self.model.to(self.device)
        self.train_labels = self.train_labels.to(self.device)
        self.val_labels = self.val_labels.to(self.device)
        self.test_labels = self.test_labels.to(self.device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        stop = 0
        for epoch in range(self.epochs):
            if stop > self.early_stop:
                # self.logger.info("Early stop!")
                break
            t = time.time()
            if self.mini_batch is False:
                loss_train, acc_train = link_cls_train(self.model, self.train_query_edges, self.train_labels, self.device,
                                                self.optimizer, self.loss_fn)
                acc_val, acc_test = link_cls_evaluate(self.model, self.val_query_edges, self.test_query_edges, 
                                                self.val_labels, self.test_labels, self.device)
            else:
                loss_train, acc_train = link_cls_mini_batch_train(self.model, self.train_query_edges, self.train_loader,
                                                         self.train_labels, self.device, self.optimizer, self.loss_fn)
                acc_val, acc_test = link_cls_mini_batch_evaluate(self.model, self.val_query_edges, self.val_loader,
                                                        self.test_query_edges, self.test_loader, self.val_labels, self.test_labels,
                                                        self.device)

            if self.normalize_times == 1:
                self.logger.info("Epoch: {:03d}, loss_train: {:.4f}, acc_train: {:.4f}, acc_val: {:.4f}, "
                                 "acc_test: {:.4f}, time: {:.4f}s".format(epoch+1, loss_train, acc_train, acc_val, acc_test, time.time() - t))

            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
                stop = 0
            stop += 1

        if self.normalize_times == 1:
            self.logger.info("Optimization Finished!")
            self.logger.info("Total training time is: {:.4f}s".format(time.time() - t_total))
            self.logger.info(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
        self.normalize_record["val_acc"].append(best_val)
        self.normalize_record["test_acc"].append(best_test)

        return best_test
