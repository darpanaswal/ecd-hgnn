#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class EarlyStoppingCriterion(object):
    """
    Arguments:
        patience (int): The maximum number of epochs with no improvement before early stopping should take place
        mode (str, can only be 'max' or 'min'): To take the maximum or minimum of the score for optimization
        min_delta (float, optional): Minimum change in the score to qualify as an improvement (default: 0.0)
    """

    def __init__(self, patience, mode, min_delta=0.0):
        assert patience >= 0
        assert mode in {'min', 'max'}
        assert min_delta >= 0.0
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._count = 0
        self.best_dev_score = None
        self.best_test_score = None
        self.best_epoch = None
        self.is_improved = None
        
        self.best_train_acc = None
        self.best_train_auc = None
        self.best_dev_auc = None
        self.best_test_auc = None

        self.best_dev_prec = None
        self.best_test_prec = None
        self.best_dev_rec = None
        self.best_test_rec = None
        self.best_dev_f1 = None
        self.best_test_f1 = None

    def step(self, cur_dev_score, cur_test_score, epoch,
        train_acc=None, train_auc=None, dev_auc=None, test_auc=None,
        dev_prec=None, test_prec=None, dev_rec=None, test_rec=None, dev_f1=None, test_f1=None):

        """
        Checks if training should be continued given the current score.

        Arguments:
            cur_dev_score (float): the current development score
            cur_test_score (float): the current test score
        Output:
            bool: if training should be continued
        """
        if self.best_dev_score is None:
            self.best_dev_score = cur_dev_score
            self.best_test_score = cur_test_score
            self.best_epoch = epoch
            # --- extra ---
            self.best_train_acc = train_acc
            self.best_train_auc = train_auc
            self.best_dev_auc = dev_auc
            self.best_test_auc = test_auc
            self.best_dev_prec = dev_prec
            self.best_test_prec = test_prec
            self.best_dev_rec = dev_rec
            self.best_test_rec = test_rec
            self.best_dev_f1 = dev_f1
            self.best_test_f1 = test_f1
            return True
        else:
            if self.mode == 'max':
                self.is_improved = (cur_dev_score > self.best_dev_score + self.min_delta)
            else:
                self.is_improved = (cur_dev_score < self.best_dev_score - self.min_delta)

            if self.is_improved:
                self._count = 0
                self.best_dev_score = cur_dev_score
                self.best_test_score = cur_test_score
                self.best_epoch = epoch
                # --- extra ---
                self.best_train_acc = train_acc
                self.best_train_auc = train_auc
                self.best_dev_auc = dev_auc
                self.best_test_auc = test_auc
                self.best_dev_prec = dev_prec
                self.best_test_prec = test_prec
                self.best_dev_rec = dev_rec
                self.best_test_rec = test_rec
                self.best_dev_f1 = dev_f1
                self.best_test_f1 = test_f1
            else:
                self._count += 1
            return self._count <= self.patience