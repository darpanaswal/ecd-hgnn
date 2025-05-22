#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class EarlyStoppingCriterion(object):
    def __init__(self, patience, mode, metric_name="f1", min_delta=0.0):
        assert patience >= 0
        assert mode in {'min', 'max'}
        assert min_delta >= 0.0
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.metric_name = metric_name

        self._count = 0
        self.best_metric = None  # Track best value of selected metric!
        self.best_test_score = None
        self.best_epoch = None
        self.is_improved = None
        
        self.best_train_acc = None
        self.best_train_auc = None
        self.best_dev_auc = None
        self.best_test_auc = None

    def step(self, cur_metric_value, cur_test_score, epoch, train_acc=None, 
        train_auc=None, dev_auc=None, test_auc=None):

        if self.best_metric is None:
            self.best_metric = cur_metric_value
            self.best_test_score = cur_test_score
            self.best_epoch = epoch
            # --- extra ---
            self.best_train_acc = train_acc
            self.best_train_auc = train_auc
            self.best_dev_auc = dev_auc
            self.best_test_auc = test_auc
            return True
        else:
            if self.mode == 'max':
                self.is_improved = (cur_metric_value > self.best_metric + self.min_delta)
            else:
                self.is_improved = (cur_metric_value < self.best_metric - self.min_delta)

            if self.is_improved:
                self._count = 0
                self.best_metric = cur_metric_value
                self.best_test_score = cur_test_score
                self.best_epoch = epoch
                # --- extra ---
                self.best_train_acc = train_acc
                self.best_train_auc = train_auc
                self.best_dev_auc = dev_auc
                self.best_test_auc = test_auc
            else:
                self._count += 1
            return self._count <= self.patience
