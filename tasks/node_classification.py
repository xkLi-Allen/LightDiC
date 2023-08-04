import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from tasks.base_task import BaseTask
from tasks.utils import accuracy, node_cls_train, node_cls_mini_batch_train, node_cls_evaluate, \
    node_cls_mini_batch_evaluate


class NodeClassification(BaseTask):
    def __init__(self, logger, dataset, model, normalize_times,
                 lr, weight_decay, epochs, early_stop, device, loss_fn=nn.CrossEntropyLoss(),
                 train_batch_size=None, eval_batch_size=None):
        super(NodeClassification, self).__init__()
        self.logger = logger
        self.normalize_times = normalize_times
        self.normalize_record = {"val_acc": [], "test_acc": []}

        self.dataset = dataset
        self.labels = self.dataset.y

        self.model_zoo = model
        self.model = model.model_init()
        self.optimizer = Adam(self.model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.epochs = epochs
        self.early_stop = early_stop
        self.loss_fn = loss_fn
        self.device = device

        self.mini_batch = False
        if train_batch_size is not None:
            self.mini_batch = True
            logger.info(f"Mini-batch training size: {train_batch_size}, eval and test size: {eval_batch_size}")
            self.train_loader = DataLoader(
                self.dataset.train_idx, batch_size=train_batch_size, shuffle=True, drop_last=False)
            self.val_loader = DataLoader(
                self.dataset.val_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.test_loader = DataLoader(
                self.dataset.test_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
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
            logger.info("Total training time is: {:.4f}s".format(time.time() - normalize_times_st))      
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
        self.labels = self.labels.to(self.device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        stop = 0
        for epoch in range(self.epochs):
            if stop > self.early_stop:
                self.logger.info("Early stop!")
                break
            t = time.time()
            if self.mini_batch is False:
                loss_train, acc_train = node_cls_train(self.model, self.dataset.train_idx, self.labels, self.device,
                                              self.optimizer, self.loss_fn)
                acc_val, acc_test = node_cls_evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                             self.labels, self.device)
            else:
                loss_train, acc_train = node_cls_mini_batch_train(self.model, self.dataset.train_idx, self.train_loader,
                                                         self.labels, self.device, self.optimizer, self.loss_fn)
                acc_val, acc_test = node_cls_mini_batch_evaluate(self.model, self.dataset.val_idx, self.val_loader,
                                                        self.dataset.test_idx, self.test_loader, self.labels,
                                                        self.device)
            if self.normalize_times == 1:
                self.logger.info("Epoch: {:03d}, loss_train: {:.4f}, acc_train: {:.4f}, acc_val: {:.4f}, "
                                 "acc_test: {:.4f}, time: {:.4f}s".format(epoch+1, loss_train, acc_train, acc_val, acc_test, time.time() - t))

            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
                stop = 0
            stop += 1

        acc_val, acc_test = self.postprocess()
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test

        if self.normalize_times == 1:
            self.logger.info("Optimization Finished!")
            self.logger.info("Total training time is: {:.4f}s".format(time.time() - t_total))
            self.logger.info(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
        self.normalize_record["val_acc"].append(best_val)
        self.normalize_record["test_acc"].append(best_test)

        return best_test

    def postprocess(self):
        # self.logger.info("Post Process!")
        self.model.eval()
        if self.model.post_graph_op is not None:
            if self.mini_batch is False:
                outputs = self.model.model_forward(
                    range(self.dataset.num_node), self.device)
            else:
                outputs = None
                for batch in self.all_eval_loader:
                    output = self.model.model_forward(batch, self.device)
                    output = F.softmax(output, dim=1)
                    output = output.cpu().detach().numpy()
                    if outputs is None:
                        outputs = output
                    else:
                        outputs = np.vstack((outputs, output))
            final_output = self.model.postprocess(self.dataset.adj, outputs)
            acc_val = accuracy(final_output[self.dataset.val_idx], self.labels[self.dataset.val_idx])
            acc_test = accuracy(final_output[self.dataset.test_idx], self.labels[self.dataset.test_idx])
        else:
            if self.mini_batch is False:
                acc_val, acc_test = node_cls_evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                             self.labels, self.device)
            else:
                acc_val, acc_test = node_cls_mini_batch_evaluate(self.model, self.dataset.val_idx, self.val_loader,
                                                        self.dataset.test_idx, self.test_loader, self.labels,
                                                        self.device)
        return acc_val, acc_test
