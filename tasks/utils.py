import math
import torch
import random
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score



def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)).item()

def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

def add_labels(features, labels, idx, num_classes):
    onehot = np.zeros([features.shape[0], num_classes])
    onehot[idx, labels[idx]] = 1
    return np.concatenate([features, onehot], axis=-1)

def link_cls_train(model, train_query_edges, train_labels, device, optimizer, loss_fn):
    model.train()
    model.base_model.query_edges = train_query_edges
    optimizer.zero_grad()
    train_output = model.model_forward(None, device)
    loss_train = loss_fn(train_output, train_labels)
    acc_train = accuracy(train_output, train_labels)
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train

def link_cls_mini_batch_train(model, train_query_edges, train_loader, train_labels, device, optimizer, loss_fn):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    for batch in train_loader:
        train_idx = torch.unique(train_query_edges[batch].reshape(-1))
        node_idx_map_dict = {train_idx[i].item():i for i in range(len(torch.unique(train_query_edges[batch].reshape(-1))))}
        row,col = train_query_edges[batch].T
        row = torch.tensor([node_idx_map_dict[row[i].item()] for i in range(len(row))])
        col = torch.tensor([node_idx_map_dict[col[i].item()] for i in range(len(col))])
        model.base_model.query_edges = torch.stack((row,col)).T
        train_output = model.model_forward(train_idx, device, train_query_edges[batch])
        loss_train = loss_fn(train_output, train_labels[batch])
        pred = train_output.max(1)[1].type_as(train_labels)
        correct_num += pred.eq(train_labels[batch]).double().sum()
        loss_train_sum += loss_train.item()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    loss_train = loss_train_sum / len(train_loader)
    acc_train = correct_num / len(train_query_edges)

    return loss_train, acc_train.item()
    
def link_cls_evaluate(model, val_query_edges, test_query_edges, val_labels, test_labels, device):
    model.eval()
    model.base_model.query_edges = val_query_edges
    val_output = model.model_forward(None, device)
    model.base_model.query_edges = test_query_edges
    test_output = model.model_forward(None, device)
    acc_val = accuracy(val_output, val_labels)
    acc_test = accuracy(test_output, test_labels)
    return acc_val, acc_test

def link_cls_mini_batch_evaluate(model, val_query_edges, val_loader, test_query_edges, test_loader, val_labels, test_labels, device):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    for batch in val_loader:
        val_idx = torch.unique(val_query_edges[batch].reshape(-1))
        node_idx_map_dict = {val_idx[i].item():i for i in range(len(torch.unique(val_query_edges[batch].reshape(-1))))}
        row,col = val_query_edges[batch].T
        row = torch.tensor([node_idx_map_dict[row[i].item()] for i in range(len(row))])
        col = torch.tensor([node_idx_map_dict[col[i].item()] for i in range(len(col))])
        model.base_model.query_edges = torch.stack((row,col)).T
        val_output = model.model_forward(val_idx, device, val_query_edges[batch])
        pred = val_output.max(1)[1].type_as(val_labels)
        correct_num_val += pred.eq(val_labels[batch]).double().sum()
    acc_val = correct_num_val / len(val_query_edges)

    for batch in test_loader:
        test_idx = torch.unique(test_query_edges[batch].reshape(-1))
        node_idx_map_dict = {test_idx[i].item():i for i in range(len(torch.unique(test_query_edges[batch].reshape(-1))))}
        row,col = test_query_edges[batch].T
        row = torch.tensor([node_idx_map_dict[row[i].item()] for i in range(len(row))])
        col = torch.tensor([node_idx_map_dict[col[i].item()] for i in range(len(col))])
        model.base_model.query_edges = torch.stack((row,col)).T
        test_output = model.model_forward(test_idx, device, test_query_edges[batch])
        pred = test_output.max(1)[1].type_as(test_labels)
        correct_num_test += pred.eq(test_labels[batch]).double().sum()
    acc_test = correct_num_test / len(test_query_edges)

    return acc_val.item(), acc_test.item()

def node_cls_evaluate(model, val_idx, test_idx, labels, device):
    model.eval()
    val_output = model.model_forward(val_idx, device)
    test_output = model.model_forward(test_idx, device)
    acc_val = accuracy(val_output, labels[val_idx])
    acc_test = accuracy(test_output, labels[test_idx])
    return acc_val, acc_test


def node_cls_mini_batch_evaluate(model, val_idx, val_loader, test_idx, test_loader, labels, device):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    for batch in val_loader:
        val_output = model.model_forward(batch, device)
        pred = val_output.max(1)[1].type_as(labels)
        correct_num_val += pred.eq(labels[batch]).double().sum()
    acc_val = correct_num_val / len(val_idx)

    for batch in test_loader:
        test_output = model.model_forward(batch, device)
        pred = test_output.max(1)[1].type_as(labels)
        correct_num_test += pred.eq(labels[batch]).double().sum()
    acc_test = correct_num_test / len(test_idx)

    return acc_val.item(), acc_test.item()


def node_cls_train(model, train_idx, labels, device, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    train_output = model.model_forward(train_idx, device)
    loss_train = loss_fn(train_output, labels[train_idx])
    acc_train = accuracy(train_output, labels[train_idx])
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), acc_train


def node_cls_mini_batch_train(model, train_idx, train_loader, labels, device, optimizer, loss_fn):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    for batch in train_loader:
        train_output = model.model_forward(batch, device)
        loss_train = loss_fn(train_output, labels[batch])
        pred = train_output.max(1)[1].type_as(labels)
        correct_num += pred.eq(labels[batch]).double().sum()
        loss_train_sum += loss_train.item()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    loss_train = loss_train_sum / len(train_loader)
    acc_train = correct_num / len(train_idx)

    return loss_train, acc_train.item()

    
