#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import * 
from task.BaseTask import BaseTask
from dataset.GraphDataset import GraphDataset
from dataset.SyntheticDataset import SyntheticDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import default_collate
from task.GraphPrediction import GraphPrediction
import torch.distributed as dist

def collate_fn(batch):
    """
    Pads every graph in <batch> to the same (#nodes , #neighbours)
    and stacks them into one mini-batch.
    """
    # ---------------------------------------------------------------
    # figure-out maxima inside the mini-batch
    # ---------------------------------------------------------------
    max_node_num, max_nei_num = 0, 0
    for data in batch:
        max_node_num = max(max_node_num, len(data['adj_mat']))     # #nodes
        for row in data['adj_mat']:
            max_nei_num = max(max_nei_num, len(row))               # #neighbours / node

    # ---------------------------------------------------------------
    # pad every field
    # ---------------------------------------------------------------
    padded_batch = []
    for data in batch:
        # make sure the 'node' field is a NumPy array so that we can
        # query its shape
        node_arr = np.asarray(data['node'], dtype=np.float32)       # (n_nodes, feat_dim)
        cur_node_num, feat_dim = node_arr.shape

        # ---------- node features ----------------------------------
        node_pad = np.zeros((max_node_num, feat_dim), dtype=np.float32)
        node_pad[:cur_node_num] = node_arr

        # ---------- adjacency & edge weights -----------------------
        adj_pad  = np.zeros((max_node_num, max_nei_num), dtype=np.int32)
        wgt_pad  = np.zeros((max_node_num, max_nei_num), dtype=np.float32)

        for i, (nei_row, w_row) in enumerate(zip(data['adj_mat'], data['weight'])):
            adj_pad[i, :len(nei_row)] = nei_row
            wgt_pad[i, :len(w_row)]   = w_row

        # ---------- construct the padded sample --------------------
        padded_batch.append({
            'node'    : node_pad,
            'adj_mat' : adj_pad,
            'weight'  : wgt_pad,
            'mask'    : np.array(cur_node_num, dtype=np.int32),  # #real nodes
            'label'   : np.asarray(data['label'])
        })

    # let PyTorch stack everything and convert to tensors
    return default_collate(padded_batch)

class GraphPredictionTask(BaseTask):

    def __init__(self, args, logger, rgnn, manifold):
        if args.is_regression:
            super(GraphPredictionTask, self).__init__(args, logger, criterion='min')
        else:
            super(GraphPredictionTask, self).__init__(args, logger, criterion='max')
        self.hyperbolic = False if args.select_manifold == "euclidean" else True
        self.rgnn = rgnn
        self.manifold = manifold

    def forward(self, model, sample, loss_function):
        mask   = sample['mask'].cuda()

        scores = model(sample['node'].cuda().float(),
                    sample['adj_mat'].cuda().long(),
                    sample['weight'].cuda().float(),
                    mask)                         # (B , C) or (B , 1)

        if self.args.is_regression:
            target = sample['label'][:, self.args.prop_idx].float().cuda()
            scores_renorm = (scores.view(-1) * self.args.std[self.args.prop_idx]
                                        + self.args.mean[self.args.prop_idx])
            loss = loss_function(scores_renorm, target)
        else:
            target = sample['label'][:, self.args.prop_idx].long().cuda()
            loss   = loss_function(scores, target)

        # -------------- return target so the caller can use it ----------
        return scores, loss, target

    def run_gnn(self):
        train_loader, dev_loader, test_loader = self.load_data()

        task_model = GraphPrediction(self.args, self.logger, self.rgnn, self.manifold).cuda()
        if getattr(self.args, 'world_suze', 1) > 1:
            model = nn.parallel.DistributedDataParallel(task_model,
                                                    device_ids=[self.args.device_id],
                                                    output_device=self.args.device_id)
        else:
            model = task_model
        if self.args.is_regression:
            loss_function = nn.MSELoss(reduction='sum')
        else:
            loss_function = nn.CrossEntropyLoss(reduction='sum')

        optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = \
                                set_up_optimizer_scheduler(self.hyperbolic, self.args, model)
        
        for epoch in range(self.args.max_epochs):
            self.reset_epoch_stats(epoch, 'train')
            model.train()
            for i, sample in enumerate(train_loader):
                model.zero_grad()
                scores, loss, target = self.forward(model, sample, loss_function)
                loss.backward(retain_graph=True)

                if self.args.grad_clip > 0.0:
                    th.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)

                optimizer.step()
                if self.hyperbolic and len(self.args.hyp_vars) != 0:
                    hyperbolic_optimizer.step()
                if self.args.is_regression and self.args.metric == "mae":
                    loss = th.sqrt(loss)
                self.update_epoch_stats(loss, scores, target,
                        is_regression=self.args.is_regression)
                if i % 400 ==0:
                    self.report_epoch_stats()
            
            dev_acc, dev_loss, dev_auc = self.evaluate(epoch, dev_loader, 'dev', model, loss_function)
            test_acc, test_loss, test_auc = self.evaluate(epoch, test_loader, 'test', model, loss_function)
            self.logger.info(f"Epoch {epoch} dev_auc: {dev_auc:.5f}  test_auc: {test_auc:.5f}")

            if self.args.is_regression and not self.early_stop.step(dev_loss, test_loss, epoch):		
                break
            elif not self.args.is_regression and not self.early_stop.step(dev_acc, test_acc, epoch):
                break

            lr_scheduler.step()
            if self.hyperbolic and len(self.args.hyp_vars) != 0:
                hyperbolic_lr_scheduler.step()
            th.cuda.empty_cache()
        self.report_best()

    def evaluate(self, epoch, data_loader, prefix, model, loss_function):
        model.eval()
        with th.no_grad():
            self.reset_epoch_stats(epoch, prefix)
            for i, sample in enumerate(data_loader):
                scores, loss, target = self.forward(model, sample, loss_function)
                if self.args.is_regression and self.args.metric == "mae":
                    loss = th.sqrt(loss)
                self.update_epoch_stats(loss, scores, target,
                        is_regression=self.args.is_regression)
            accuracy, loss, roc_auc = self.report_epoch_stats()
        if self.args.is_regression and self.args.metric == "rmse":
            loss = np.sqrt(loss)
        return accuracy, loss, roc_auc

    def load_data(self):
        if self.args.task == 'synthetic':
            return self.load_dataset(SyntheticDataset, collate_fn)
        else:
            return self.load_dataset(GraphDataset, collate_fn)	
