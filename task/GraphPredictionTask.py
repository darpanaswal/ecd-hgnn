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
    Pads every graph in <batch> to the same number of nodes and the same
    number of neighbours per node, returns a single dictionary whose
    tensors have a leading batch dimension.
    """
    # ---------------------------------------------------------------
    # first pass – obtain the largest (#nodes , #neighbours) in batch
    # ---------------------------------------------------------------
    max_node_num, max_nei_num = 0, 0
    for data in batch:
        max_node_num = max(max_node_num, len(data['adj_mat']))          # #nodes
        for row in data['adj_mat']:                                     # neighbours/row
            max_nei_num = max(max_nei_num, len(row))

    # ---------------------------------------------------------------
    # second pass – pad *every* field to (max_node_num, max_nei_num)
    # ---------------------------------------------------------------
    new_batch = []
    for data in batch:
        cur_node_num = len(data['adj_mat'])
        pad_node   = max_node_num - cur_node_num

        # (1) node feature matrix  ------------------
        node = np.zeros((max_node_num, data['node'].shape[1]), dtype=np.float32)
        node[:cur_node_num] = data['node']

        # (2) adjacency & weights -------------------
        adj   = np.zeros((max_node_num, max_nei_num), dtype=np.int32)
        wght  = np.zeros((max_node_num, max_nei_num), dtype=np.float32)
        for i, (nei_row, w_row) in enumerate(zip(data['adj_mat'], data['weight'])):
            adj[i, :len(nei_row)]  = nei_row
            wght[i, :len(w_row)]   = w_row

        # (3) store a mask = real #nodes ------------
        mask = cur_node_num                                        # scalar

        new_batch.append({
            'node'    : node,
            'adj_mat' : adj,
            'weight'  : wght,
            'mask'    : mask,
            'label'   : data['label']
        })
    # let default_collate stack the list into tensors of shape
    #    node     -> (B , max_node , feat_dim)
    #    adj_mat  -> (B , max_node , max_nei)
    #    ...
    return default_collate(new_batch)

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
        # mask : (B,) – number of real nodes per graph
        mask = sample['mask'].cuda()

        scores = model(sample['node'].cuda().float(),
                       sample['adj_mat'].cuda().long(),
                       sample['weight'].cuda().float(),
                       mask)                                # (B,C) or (B,1)

        if self.args.is_regression:
            target = sample['label'][:,self.args.prop_idx].float().cuda()
            # de-normalise inside the loss if necessary
            scores = scores.view(-1) * self.args.std[self.args.prop_idx] \
                             + self.args.mean[self.args.prop_idx]
            loss   = loss_function(scores, target)
        else:                                   # classification
            target = sample['label'][:,self.args.prop_idx].long().cuda()
            loss   = loss_function(scores, target)

        return scores, loss

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
				scores, loss = self.forward(model, sample, loss_function)
				loss.backward(retain_graph=True)

				if self.args.grad_clip > 0.0:
					th.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)

				optimizer.step()
				if self.hyperbolic and len(self.args.hyp_vars) != 0:
					hyperbolic_optimizer.step()
				if self.args.is_regression and self.args.metric == "mae":
					loss = th.sqrt(loss)
				self.update_epoch_stats(loss, scores, sample['label'].cuda(), is_regression=self.args.is_regression)			
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
				scores, loss = self.forward(model, sample, loss_function)
				if self.args.is_regression and self.args.metric == "mae":
					loss = th.sqrt(loss)
				self.update_epoch_stats(loss, scores, sample['label'].cuda(), is_regression=self.args.is_regression)
			accuracy, loss, roc_auc = self.report_epoch_stats()
		if self.args.is_regression and self.args.metric == "rmse":
			loss = np.sqrt(loss)
		return accuracy, loss, roc_auc

	def load_data(self):
		if self.args.task == 'synthetic':
			return self.load_dataset(SyntheticDataset, collate_fn)
		else:
			return self.load_dataset(GraphDataset, collate_fn)	
