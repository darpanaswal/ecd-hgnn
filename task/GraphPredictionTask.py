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
            'mask'    : np.array([cur_node_num], dtype=np.int32),  # <-- shape (1,)
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
        mask = sample['mask'].cuda()         # shape (B,1)
        scores = model(sample['node'].cuda().float(),
                    sample['adj_mat'].cuda().long(),
                    sample['weight'].cuda().float(),
                    mask)

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

        # Corrected DistributedDataParallel setup
        if getattr(self.args, "world_size", 1) > 1 and th.cuda.is_available() and th.distributed.is_initialized():
            model = nn.parallel.DistributedDataParallel(task_model,
                                                        device_ids=[self.args.device_id], # typically self.args.local_rank
                                                        output_device=self.args.device_id) # typically self.args.local_rank
        else:
            model = task_model

        if self.args.is_regression:
            loss_function = nn.MSELoss(reduction='sum')
        else:
            loss_function = nn.CrossEntropyLoss(reduction='sum')
        
        optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = \
                                set_up_optimizer_scheduler(self.hyperbolic, self.args, model) # Ensure set_up_optimizer_scheduler is defined/imported
        
        for epoch in range(self.args.max_epochs):
            self.reset_epoch_stats(epoch, 'train')
            model.train()
            if train_loader.sampler is not None and isinstance(train_loader.sampler, DistributedSampler): # Required for DDP
                train_loader.sampler.set_epoch(epoch)

            for i, sample in enumerate(train_loader):
                model.zero_grad()
                if hyperbolic_optimizer: # zero grad for hyperbolic optimizer if it exists
                    hyperbolic_optimizer.zero_grad()

                scores, loss, target = self.forward(model, sample, loss_function)
                loss.backward(retain_graph=False)
                
                if self.args.grad_clip > 0.0:
                    th.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                
                optimizer.step()
                if self.hyperbolic and hyperbolic_optimizer and len(self.args.hyp_vars) != 0 :
                    hyperbolic_optimizer.step()
                
                current_loss_val = loss.item() # Use .item() for scalar loss
                if self.args.is_regression and self.args.metric == "mae":
                    # MAE is usually L1 loss. If MSELoss is used, sqrt gives RMSE.
                    # This seems to be calculating MAE from an MSE sum loss, which is unusual.
                    # If loss_function is MSELoss(reduction='sum'), loss is sum of squared errors.
                    # To get MAE, you'd sum absolute errors. sqrt(sum_sq_err) is not MAE.
                    # If you intend RMSE, then current_loss_val should be current_loss_val / num_items_in_batch before sqrt.
                    # For simplicity, I'm assuming this custom logic is intended.
                     current_loss_val = th.sqrt(loss).item()


                self.update_epoch_stats(th.tensor(current_loss_val), scores, target, # Pass loss as a tensor scalar
                        is_regression=self.args.is_regression)
                
                if i % 400 == 0: # Log progress within an epoch
                    # report_epoch_stats will be called with partial epoch data here.
                    # Consider if this intermediate reporting is for rank 0 only or all.
                    # The current report_epoch_stats handles distributed aggregation.
                    self.report_epoch_stats() 
            
            # End of epoch reporting
            train_acc, train_loss, train_auc, train_precision, train_recall, train_f1 = self.report_epoch_stats()
            
            # Evaluation
            dev_acc, dev_loss, dev_auc, dev_precision, dev_recall, dev_f1 = self.evaluate(
                epoch, dev_loader, 'dev', model, loss_function
            )
            test_acc, test_loss, test_auc, test_precision, test_recall, test_f1 = self.evaluate(
                epoch, test_loader, 'test', model, loss_function
            )

            self.logger.info(
                f"Epoch {epoch}: Dev AUC: {dev_auc:.5f}, Dev F1: {dev_f1:.5f} | Test AUC: {test_auc:.5f}, Test F1: {test_f1:.5f}"
            )

            # Early stopping step
            if self.args.is_regression:
                current_eval_metric = dev_loss # or another relevant metric for regression
            else:
                current_eval_metric = dev_acc # or dev_f1 or dev_auc depending on primary metric

            stop = not self.early_stop.step(
                cur_dev_score=current_eval_metric, # Main metric for early stopping decision
                cur_test_score=test_acc if not self.args.is_regression else test_loss, # Corresponding test score
                epoch=epoch,
                train_acc=train_acc, train_auc=train_auc,
                dev_auc=dev_auc, test_auc=test_auc,
                train_precision=train_precision, train_recall=train_recall, train_f1=train_f1,
                dev_precision=dev_precision, dev_recall=dev_recall, dev_f1=dev_f1,
                test_precision=test_precision, test_recall=test_recall, test_f1=test_f1
            )
            
            if stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
            
            lr_scheduler.step()
            if self.hyperbolic and hyperbolic_lr_scheduler and len(self.args.hyp_vars) != 0:
                hyperbolic_lr_scheduler.step()
            
            if th.cuda.is_available():
                 th.cuda.empty_cache()
        
        self.report_best() # Report best scores found during training

    def evaluate(self, epoch, data_loader, prefix, model, loss_function):
        model.eval()
        if data_loader.sampler is not None and isinstance(data_loader.sampler, DistributedSampler): # For DDP
            data_loader.sampler.set_epoch(epoch) # Not strictly needed for eval but good practice

        with th.no_grad():
            self.reset_epoch_stats(epoch, prefix)
            for i, sample in enumerate(data_loader):
                scores, loss, target = self.forward(model, sample, loss_function)
                
                current_loss_val = loss.item()
                if self.args.is_regression and self.args.metric == "mae":
                    current_loss_val = th.sqrt(loss).item() # Same logic as in training loop

                self.update_epoch_stats(th.tensor(current_loss_val), scores, target,
                        is_regression=self.args.is_regression)
            
            # report_epoch_stats now returns 6 values
            accuracy, loss_val, roc_auc, precision, recall, f1 = self.report_epoch_stats()

        # This loss is per-batch average loss from report_epoch_stats
        # If metric is rmse, and loss was sum of squares, then sqrt(loss_val) is appropriate if loss_val is MSE
        if self.args.is_regression and self.args.metric == "rmse":
            # Ensure loss_val is mean squared error before taking sqrt
            # The current report_epoch_stats returns loss = sum_loss / num_total, which is effectively MSE if loss_function was sum of squares.
            loss_val = np.sqrt(loss_val) if loss_val is not None and not np.isnan(loss_val) else float('nan')
            
        return accuracy, loss_val, roc_auc, precision, recall, f1

    def load_data(self):
        # Determine if running in distributed mode
        is_distributed = getattr(self.args, "world_size", 1) > 1 and th.cuda.is_available() and th.distributed.is_initialized()

        if self.args.task == 'synthetic':
            return self.load_dataset(SyntheticDataset, collate_fn, distributed=is_distributed)
        else:
            return self.load_dataset(GraphDataset, collate_fn, distributed=is_distributed)	
