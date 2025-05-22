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
    class BaseTask(object):
    def __init__(self, args, logger, criterion='max', early_stop_metric="f1"):
        self.args = args
        self.logger = logger
        self.early_stop = EarlyStoppingCriterion(self.args.patience, criterion, metric_name=early_stop_metric)


    def reset_epoch_stats(self, epoch, prefix):
        self.epoch_stats = {
            'prefix': prefix,
            'epoch': epoch,
            'loss': 0,
            'num_correct': 0,
            'num_total': 0,
            'y_true': [],
            'y_pred': [],
            # Add for metrics
            'y_argmax_pred': [],   # NEW: For precision/recall/f1
            'y_target': [],        # NEW: For precision/recall/f1
        }

    def update_epoch_stats(self, loss, score, label, is_regression=False):
        with th.no_grad():
            self.epoch_stats['loss'] += loss.item()
            self.epoch_stats['num_total'] += label.size(0)
            if not is_regression:
                pred = th.argmax(score, dim=1)
                self.epoch_stats['num_correct'] += th.sum(th.eq(pred, label)).item()
                # For metrics
                self.epoch_stats['y_argmax_pred'].extend(pred.detach().cpu().tolist())  # NEW
                self.epoch_stats['y_target'].extend(label.detach().cpu().tolist())      # NEW
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
        y_true_tensor = torch.tensor(stats['y_true'], dtype=torch.float32, device='cuda')
        y_pred_tensor = torch.tensor(stats['y_pred'], dtype=torch.float32 if isinstance(stats['y_pred'][0], float) else torch.float32, device='cuda')
        # Also gather argmax preds and targets
        y_argmax_pred_tensor = torch.tensor(stats['y_argmax_pred'], dtype=torch.int64, device='cuda')
        y_target_tensor      = torch.tensor(stats['y_target'], dtype=torch.int64, device='cuda')

        y_true_list = [torch.zeros_like(y_true_tensor) for _ in range(world_size)]
        y_pred_list = [torch.zeros_like(y_pred_tensor) for _ in range(world_size)]
        y_argmax_pred_list = [torch.zeros_like(y_argmax_pred_tensor) for _ in range(world_size)]
        y_target_list      = [torch.zeros_like(y_target_tensor) for _ in range(world_size)]

        torch.distributed.all_gather(y_true_list, y_true_tensor)
        torch.distributed.all_gather(y_pred_list, y_pred_tensor)
        torch.distributed.all_gather(y_argmax_pred_list, y_argmax_pred_tensor)
        torch.distributed.all_gather(y_target_list, y_target_tensor)
        full_y_true = torch.cat(y_true_list).cpu().numpy().tolist()
        full_y_pred = torch.cat(y_pred_list).cpu().numpy().tolist()
        full_argmax_pred = torch.cat(y_argmax_pred_list).cpu().numpy().tolist()
        full_target      = torch.cat(y_target_list).cpu().numpy().tolist()
        return full_y_true, full_y_pred, full_argmax_pred, full_target

    def report_epoch_stats(self):
        """ Report accuracy and loss, plus ROC-AUC and precision/recall/F1 (if classification)"""
        do_roc_auc = (hasattr(self, "args") and getattr(self.args, "compute_roc_auc", False)) or getattr(self, "compute_roc_auc", False)
        from torch import distributed as dist
        accuracy, loss = None, None
        roc_auc = None
        precision = None
        recall = None
        f1 = None

        if (self.epoch_stats['prefix'] == 'train') or (getattr(self.args, "world_size", 1) == 1):
            statistics = [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']]
            y_true = self.epoch_stats['y_true']
            y_pred = self.epoch_stats['y_pred']
            # For classification metrics
            y_argmax_pred = self.epoch_stats['y_argmax_pred']
            y_target      = self.epoch_stats['y_target']
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
            # Gather y_true, y_pred, y_argmax_pred, y_target
            y_true, y_pred, y_argmax_pred, y_target = self.gather_epoch_stats_distributed(self.epoch_stats)
        accuracy = float(statistics[0]) / statistics[1]
        loss = statistics[2] / statistics[1]

        # ROC-AUC computation if needed
        if do_roc_auc:
            try:
                from sklearn.metrics import roc_auc_score
                roc_auc = roc_auc_score(y_true, y_pred)
            except Exception as ex:
                roc_auc = float('nan')
                self.logger.info(f"ROC-AUC calculation failed: {ex}")

        # Compute precision/recall/F1 ***NEW***
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            if len(set(y_target)) > 2:   # Multi-class
                average_type = 'micro'
            else:
                average_type = 'binary'
            precision = precision_score(y_target, y_argmax_pred, average=average_type, zero_division=0)
            recall    = recall_score(y_target, y_argmax_pred, average=average_type, zero_division=0)
            f1        = f1_score(y_target, y_argmax_pred, average=average_type, zero_division=0)
        except Exception as ex:
            precision = recall = f1 = float('nan')
            self.logger.info(f"Precision/Recall/F1 calculation failed: {ex}")

        # Print logs, include the new metrics
        msg = (
            "rank %d, %s phase of epoch %d: accuracy %.6f, loss %.6f"
            % (self.args.distributed_rank, self.epoch_stats['prefix'], self.epoch_stats['epoch'], accuracy, loss)
        )
        if roc_auc is not None:
            msg += ", auc %.6f" % roc_auc
        if precision is not None:
            msg += ", prec %.6f, recall %.6f, f1 %.6f" % (precision, recall, f1)
        msg += ", num_correct %d, total %d" % (statistics[0], statistics[1])

        if self.epoch_stats['prefix'] != 'test':
            self.logger.info(msg)
        return accuracy, loss, roc_auc, precision, recall, f1

    def report_best(self):
        self.logger.info(
            (
                "BEST at epoch %d: dev %.6f, test %.6f, train_acc %.6f, train_auc %s, dev_auc %s, test_auc %s"
            )
            % (
                self.early_stop.best_epoch,
                self.early_stop.best_dev_score,
                self.early_stop.best_test_score,
                self.early_stop.best_train_acc,
                str(self.early_stop.best_train_auc),
                str(self.early_stop.best_dev_auc),
                str(self.early_stop.best_test_auc),
            )
        )

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
