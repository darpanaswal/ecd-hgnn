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
            'y_pred_scores': [], # Renamed from y_pred to avoid confusion with class predictions
            'y_pred_labels': [] # For storing predicted labels
        }

    def update_epoch_stats(self, loss, score, label, is_regression=False):
        with th.no_grad():
            self.epoch_stats['loss'] += loss.item()
            self.epoch_stats['num_total'] += label.size(0)
            if not is_regression:
                pred_labels = th.argmax(score, dim=1)
                self.epoch_stats['num_correct'] += th.sum(th.eq(pred_labels, label)).item()
                self.epoch_stats['y_pred_labels'].extend(pred_labels.detach().cpu().tolist())

                # For ROC-AUC, store labels and probability scores
                if score.shape[1] == 2: # binary
                    prob1 = th.softmax(score, dim=1)[:,1]
                    self.epoch_stats['y_pred_scores'].extend(prob1.detach().cpu().tolist())
                else: # multi-class
                    probs = th.softmax(score, dim=1)
                    self.epoch_stats['y_pred_scores'].extend(probs.detach().cpu().tolist()) # For multi-class AUC, store all probs
                self.epoch_stats['y_true'].extend(label.detach().cpu().tolist())
            else:
                # regression: you can extend here if needed
                # For regression, precision, recall, F1 are not typically used.
                # If you have a specific way to calculate them for regression, add it here.
                pass

    def gather_epoch_stats_distributed(self, stats):
        import torch # Ensure torch is imported if not already
        world_size = self.args.world_size

        # Gather y_true
        y_true_tensor = torch.tensor(stats['y_true'], dtype=torch.long, device='cuda') # Assuming labels are long
        y_true_list = [torch.zeros_like(y_true_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(y_true_list, y_true_tensor)
        full_y_true = torch.cat(y_true_list).cpu().numpy().tolist()

        # Gather y_pred_scores (probabilities or scores for AUC)
        # Ensure y_pred_scores is not empty before converting to tensor
        if stats['y_pred_scores']:
            # Determine dtype based on content; assuming float for scores
            if isinstance(stats['y_pred_scores'][0], list): # Multi-class probabilities
                 # Find max length of sublists for padding if necessary, or handle appropriately
                max_len = max(len(p) for p in stats['y_pred_scores']) if stats['y_pred_scores'] else 0
                # Pad score lists to have the same length for tensor conversion
                padded_scores = [p + [0.0] * (max_len - len(p)) for p in stats['y_pred_scores']]
                y_pred_scores_tensor = torch.tensor(padded_scores, dtype=torch.float32, device='cuda')
            else: # Binary probabilities
                y_pred_scores_tensor = torch.tensor(stats['y_pred_scores'], dtype=torch.float32, device='cuda')

            y_pred_scores_list = [torch.zeros_like(y_pred_scores_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(y_pred_scores_list, y_pred_scores_tensor)
            full_y_pred_scores = torch.cat(y_pred_scores_list).cpu().numpy().tolist()
        else:
            full_y_pred_scores = []


        # Gather y_pred_labels (actual predicted class labels for P/R/F1)
        if stats['y_pred_labels']:
            y_pred_labels_tensor = torch.tensor(stats['y_pred_labels'], dtype=torch.long, device='cuda')
            y_pred_labels_list = [torch.zeros_like(y_pred_labels_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(y_pred_labels_list, y_pred_labels_tensor)
            full_y_pred_labels = torch.cat(y_pred_labels_list).cpu().numpy().tolist()
        else:
            full_y_pred_labels = []


        return full_y_true, full_y_pred_scores, full_y_pred_labels

    def report_epoch_stats(self):
        """ Report accuracy, loss, ROC-AUC, Precision, Recall, F1-score (if classification)"""
        do_roc_auc = (hasattr(self, "args") and getattr(self.args, "compute_roc_auc", False)) or getattr(self, "compute_roc_auc", False)
        # Assuming these metrics are also desired if ROC-AUC is computed, or make a separate flag
        compute_prf = not self.args.is_regression

        accuracy, loss = None, None
        roc_auc, precision, recall, f1 = None, None, None, None
        avg_precision, avg_recall, avg_f1 = float('nan'), float('nan'), float('nan') # For overall metrics

        if (self.epoch_stats['prefix'] == 'train') or (getattr(self.args, "world_size", 1) == 1):
            statistics_tensor = th.tensor(
                [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']],
                dtype=th.float32
            ) # Keep as tensor for consistency, convert to list/float later
            y_true = self.epoch_stats['y_true']
            y_pred_scores = self.epoch_stats['y_pred_scores']
            y_pred_labels = self.epoch_stats['y_pred_labels']
            num_correct = statistics_tensor[0].item()
            num_total = statistics_tensor[1].item()
            current_loss_sum = statistics_tensor[2].item()

        else:
            # aggregate the results from all nodes
            import torch.distributed as dist
            group = dist.new_group(list(range(self.args.world_size))) # Ensure group covers all ranks
            statistics_gather = th.tensor(
                [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']],
                dtype=th.float32
            ).cuda() # Ensure it's on CUDA for distributed operations

            if self.args.dist_method == 'reduce':
                dist.reduce(tensor=statistics_gather, dst=0, op=dist.ReduceOp.SUM, group=group)
                # Only rank 0 will have the summed statistics
                if self.args.distributed_rank == 0:
                    num_correct = statistics_gather[0].item()
                    num_total = statistics_gather[1].item()
                    current_loss_sum = statistics_gather[2].item()
                else: # Other ranks won't have the correct sum, so set to defaults or handle
                    num_correct, num_total, current_loss_sum = 0, 0, 0
            elif self.args.dist_method == 'all_gather':
                all_statistics = [th.zeros_like(statistics_gather) for _ in range(self.args.world_size)]
                dist.all_gather(tensor_list=all_statistics, tensor=statistics_gather, group=group)
                # Each rank will have all statistics, sum them up
                # Ensure all_statistics are on CPU before converting to numpy and summing for all ranks
                aggregated_stats = th.sum(th.stack(all_statistics).cpu(), dim=0)
                num_correct = aggregated_stats[0].item()
                num_total = aggregated_stats[1].item()
                current_loss_sum = aggregated_stats[2].item()
            else: # Fallback or error
                num_correct, num_total, current_loss_sum = 0,0,0 # Or raise error

            # Gather y_true, y_pred_scores, y_pred_labels
            # This part needs to run on all ranks to participate in all_gather
            y_true, y_pred_scores, y_pred_labels = self.gather_epoch_stats_distributed(self.epoch_stats)
            # After gather, y_true, y_pred_scores, y_pred_labels will be complete lists on all ranks if all_gather was used for them.
            # If using reduce for statistics, these might only be fully populated on rank 0,
            # or need to be broadcasted if other ranks need them.
            # For simplicity, let's assume gather_epoch_stats_distributed makes them available on all ranks
            # (or at least the rank doing the reporting, typically rank 0).

        # Calculate metrics on the process that has the full data (e.g., rank 0 after reduce, or all ranks after all_gather)
        # We need to ensure these calculations only happen where num_total is valid.
        if num_total > 0 :
            accuracy = float(num_correct) / num_total
            loss = current_loss_sum / num_total

            if compute_prf and y_true and y_pred_labels:
                # For multi-class, specify average method if needed, e.g., 'macro' or 'micro' or 'weighted'
                # For binary, default is fine.
                average_method = 'binary' if self.args.num_class == 2 else 'macro'
                if self.args.num_class == 1 and self.args.is_regression: # Should not happen if compute_prf is false for regression
                     pass
                elif len(set(y_true)) < 2 and average_method == 'binary': # Not enough classes for binary
                    self.logger.info(f"Not enough classes in y_true for binary classification metrics. Unique values: {set(y_true)}")
                    precision, recall, f1 = float('nan'), float('nan'), float('nan')
                else:
                    try:
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred_labels, average=average_method, zero_division=0
                        )
                        avg_precision, avg_recall, avg_f1 = precision, recall, f1 # If single values from average
                    except ValueError as e:
                        self.logger.info(f"Precision/Recall/F1 calculation failed: {e}. y_true: {np.unique(y_true, return_counts=True)}, y_pred_labels: {np.unique(y_pred_labels, return_counts=True)}")
                        precision, recall, f1 = float('nan'), float('nan'), float('nan')


            if do_roc_auc and y_true and y_pred_scores:
                try:
                    if self.args.num_class == 2: # Binary
                        roc_auc = roc_auc_score(y_true, y_pred_scores)
                    else: # Multi-class
                        # Ensure y_pred_scores are probabilities for each class
                        # y_pred_scores should be list of lists/arrays if multi-class
                        if y_pred_scores and isinstance(y_pred_scores[0], list):
                             roc_auc = roc_auc_score(y_true, y_pred_scores, multi_class='ovr', average='macro') # or 'ovo'
                        else: # Should not happen if correctly storing multi-class scores
                            self.logger.info("y_pred_scores not in expected format for multi-class AUC.")
                            roc_auc = float('nan')
                except Exception as ex:
                    roc_auc = float('nan')
                    self.logger.info(f"ROC-AUC calculation failed: {ex}. y_true unique: {np.unique(y_true)}. y_pred_scores example: {y_pred_scores[:5]}")
        else: # num_total is 0, occurs on non-master ranks if reduce was used for stats
            accuracy, loss = float('nan'), float('nan')
            roc_auc, avg_precision, avg_recall, avg_f1 = float('nan'), float('nan'), float('nan'), float('nan')


        # Logging should ideally happen only on rank 0 in distributed setting
        # or be guarded by self.args.distributed_rank == 0
        if getattr(self.args, "world_size", 1) == 1 or self.args.distributed_rank == 0:
            log_msg_parts = [
                f"rank {self.args.distributed_rank}",
                f"{self.epoch_stats['prefix']} phase of epoch {self.epoch_stats['epoch']}",
                f"accuracy {accuracy:.6f}" if accuracy is not None else "accuracy nan",
                f"loss {loss:.6f}" if loss is not None else "loss nan"
            ]
            if compute_prf:
                log_msg_parts.extend([
                    f"precision {avg_precision:.6f}" if avg_precision is not None else "precision nan",
                    f"recall {avg_recall:.6f}" if avg_recall is not None else "recall nan",
                    f"f1 {avg_f1:.6f}" if avg_f1 is not None else "f1 nan"
                ])
            if do_roc_auc:
                 log_msg_parts.append(f"auc {roc_auc:.6f}" if roc_auc is not None else "auc nan")

            log_msg_parts.extend([
                f"num_correct {num_correct}",
                f"total {num_total}"
            ])
            self.logger.info(", ".join(log_msg_parts))

        return accuracy, loss, roc_auc, avg_precision, avg_recall, avg_f1

    def report_best(self):
        self.logger.info(
            (
                "BEST at epoch %d: dev %.6f, test %.6f, train_acc %.6f, "
                "train_auc %s, dev_auc %s, test_auc %s, "
                "dev_f1 %s, test_f1 %s, dev_precision %s, test_precision %s, dev_recall %s, test_recall %s"
            )
            % (
                self.early_stop.best_epoch,
                self.early_stop.best_dev_score,
                self.early_stop.best_test_score,
                self.early_stop.best_train_acc,
                str(self.early_stop.best_train_auc),
                str(self.early_stop.best_dev_auc),
                str(self.early_stop.best_test_auc),
                str(self.early_stop.best_dev_f1),
                str(self.early_stop.best_test_f1),
                str(self.early_stop.best_dev_precision),
                str(self.early_stop.best_test_precision),
                str(self.early_stop.best_dev_recall),
                str(self.early_stop.best_test_recall),
            )
        )

    def load_dataset(self, dataset_class, collate_fn, distributed=True):
        bs = getattr(self.args, 'batch_size', 1)
        train_dataset = dataset_class(self.args, self.logger, split='train')
        dev_dataset = dataset_class(self.args, self.logger, split='dev')
        test_dataset = dataset_class(self.args, self.logger, split='test')
        
        train_sampler, dev_sampler, test_sampler = None, None, None # Initialize
        if distributed and getattr(self.args, "world_size", 1) > 1 : # Check if distributed training is active
        world_size = self.args.world_size
        distributed_rank = self.args.distributed_rank
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=distributed_rank, shuffle=True) # shuffle=True for training
        # For dev/test, shuffling is not strictly necessary and can be off, ensure all GPUs see all data or a consistent part
        dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=distributed_rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=distributed_rank, shuffle=False)
        
        # Determine if shuffle is needed for non-distributed training
        shuffle_train = train_sampler is None # Shuffle if not using DistributedSampler

        train_loader = DataLoader(train_dataset, batch_size=bs,
                                collate_fn=collate_fn, num_workers=0, # Consider increasing num_workers
                                sampler=train_sampler, shuffle=shuffle_train) # Add shuffle
        dev_loader   = DataLoader(dev_dataset,   batch_size=bs,
                                collate_fn=collate_fn, num_workers=0,
                                sampler=dev_sampler, shuffle=False) # No shuffle for dev
        test_loader  = DataLoader(test_dataset,  batch_size=bs,
                                collate_fn=collate_fn, num_workers=0,
                                sampler=test_sampler, shuffle=False) # No shuffle for test

        self.logger.info("train data size: %d (per GPU if distributed)" % len(train_dataset))
        self.logger.info("dev data size: %d (per GPU if distributed)" % len(dev_dataset))
        self.logger.info("test data size: %d (per GPU if distributed)" % len(test_dataset))
        return train_loader, dev_loader, test_loader
