#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from utils import *
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class BaseTask(object):
    """
    A base class that supports loading datasets, early stop and reporting statistics
    """
    def __init__(self, args, logger, criterion='max'):
        self.args = args
        self.logger = logger
        self.early_stop = EarlyStoppingCriterion(self.args.patience, criterion)

    def reset_epoch_stats(self, epoch, prefix):
        self.epoch_stats = {
            'prefix': prefix,
            'epoch': epoch,
            'loss': 0,
            'num_correct': 0,
            'num_total': 0,
            # For ROC-AUC
            'y_true': [],
            'y_pred': []
        }

    def update_epoch_stats(self, loss, score, label, is_regression=False):
        with th.no_grad():
            self.epoch_stats['loss'] += loss.item()
            self.epoch_stats['num_total'] += label.size(0)
            if not is_regression:
                pred = th.argmax(score, dim=1)
                self.epoch_stats['num_correct'] += th.sum(th.eq(pred, label)).item()
                # For ROC-AUC, store labels and probability scores
                if score.shape[1] == 2:
                    # binary, use prob for class 1
                    prob1 = th.softmax(score, dim=1)[:,1]
                    self.epoch_stats['y_pred'].extend(prob1.detach().cpu().tolist())
                else:
                    # multi-class, take max-prob
                    probs = th.softmax(score, dim=1)
                    self.epoch_stats['y_pred'].extend(probs.detach().cpu().tolist())
                self.epoch_stats['y_true'].extend(label.detach().cpu().tolist())
            else:
                # regression: you can extend here if needed for regression ROC-AUC (rarely used)
                pass

    def gather_epoch_stats_distributed(self, stats):
        # Gather and concat all y_true and y_pred across processes (assuming CUDA tensors)
        # This function should be only called for dev/test stats and be torch.distributed aware
        import torch
        world_size = self.args.world_size
        # Gather y_true
        y_true_tensor = torch.tensor(stats['y_true'], dtype=torch.float32, device='cuda')
        y_pred_tensor = torch.tensor(stats['y_pred'], dtype=torch.float32 if isinstance(stats['y_pred'][0], float) else torch.float32, device='cuda')
        y_true_list = [torch.zeros_like(y_true_tensor) for _ in range(world_size)]
        y_pred_list = [torch.zeros_like(y_pred_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(y_true_list, y_true_tensor)
        torch.distributed.all_gather(y_pred_list, y_pred_tensor)
        full_y_true = torch.cat(y_true_list).cpu().numpy().tolist()
        full_y_pred = torch.cat(y_pred_list).cpu().numpy().tolist()
        return full_y_true, full_y_pred

    def report_epoch_stats(self):
        """ Report accuracy and loss, plus ROC-AUC (if classification)"""
        do_roc_auc = (hasattr(self, "args") and getattr(self.args, "compute_roc_auc", False)) or getattr(self, "compute_roc_auc", False)
        from torch import distributed as dist
        accuracy, loss = None, None
        roc_auc = None

        if (self.epoch_stats['prefix'] == 'train') or (getattr(self.args, "world_size", 1) == 1):
            statistics = [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']]
            y_true = self.epoch_stats['y_true']
            y_pred = self.epoch_stats['y_pred']
        else:
            # aggregate the results from all nodes
            import torch.distributed as dist
            group = dist.new_group(range(self.args.world_size))
            statistics = th.tensor(
                [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']],
                dtype=th.float32
            ).cuda()
            if self.args.dist_method == 'reduce':
                dist.reduce(tensor=statistics, dst=0, op=dist.ReduceOp.SUM, group=group)
            elif self.args.dist_method == 'all_gather':
                all_statistics = [th.zeros((1, 3)).cuda() for _ in range(self.args.world_size)]    
                dist.all_gather(tensor=statistics, tensor_list=all_statistics, group=group)
                statistics = th.sum(th.cat(all_statistics, dim=0), dim=0).cpu().numpy()
            # Gather y_true, y_pred
            y_true, y_pred = self.gather_epoch_stats_distributed(self.epoch_stats)
        accuracy = float(statistics[0]) / statistics[1]
        loss = statistics[2] / statistics[1]

        # ROC-AUC computation if needed
        if do_roc_auc:
            try:
                from sklearn.metrics import roc_auc_score
                # Binary: y_pred is list of prob for class 1
                roc_auc = roc_auc_score(y_true, y_pred)
            except Exception as ex:
                roc_auc = float('nan')
                self.logger.info(f"ROC-AUC calculation failed: {ex}")
        
        if self.epoch_stats['prefix'] != 'test':
            if roc_auc is not None:
                self.logger.info(
                    "rank %d, %s phase of epoch %d: accuracy %.6f, loss %.6f, auc %.6f, num_correct %d, total %d" %
                    (self.args.distributed_rank, self.epoch_stats['prefix'], self.epoch_stats['epoch'], accuracy,
                     loss, roc_auc, statistics[0], statistics[1]))
            else:
                self.logger.info(
                    "rank %d, %s phase of epoch %d: accuracy %.6f, loss %.6f, num_correct %d, total %d" % (
                    self.args.distributed_rank, self.epoch_stats['prefix'], self.epoch_stats['epoch'], accuracy,
                    loss, statistics[0], statistics[1]))
        # Return ROC-AUC for external reference/use
        return accuracy, loss, roc_auc

    def report_best(self):
      self.logger.info("best dev %.6f, best test %.6f" 
        % (self.early_stop.best_dev_score, self.early_stop.best_test_score))

    def load_dataset(self, dataset_class, collate_fn, distributed=True):
      bs = getattr(self.args, 'batch_size', 1)          # <-- new
      train_dataset = dataset_class(self.args, self.logger, split='train')
      dev_dataset = dataset_class(self.args, self.logger, split='dev')
      test_dataset = dataset_class(self.args, self.logger, split='test')
      if distributed:
        world_size = getattr(self.args, "world_size", 1)
        distributed_rank = getattr(self.args, "distributed_rank", 0)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=distributed_rank)
        dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=distributed_rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=distributed_rank)
      else:
        train_sampler, dev_sampler, test_sampler = None, None, None
      train_loader = DataLoader(train_dataset, batch_size=bs,
                                collate_fn=collate_fn, num_workers=0,
                                sampler=train_sampler)
      dev_loader   = DataLoader(dev_dataset,   batch_size=bs,
                                collate_fn=collate_fn, num_workers=0,
                                sampler=dev_sampler)
      test_loader  = DataLoader(test_dataset,  batch_size=bs,
                                collate_fn=collate_fn, num_workers=0,
                                sampler=test_sampler)
      self.logger.info("train data size: %d" % len(train_dataset))
      self.logger.info("dev data size: %d" % len(dev_dataset))
      self.logger.info("test data size: %d" % len(test_dataset))
      return train_loader, dev_loader, test_loader
