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
from hyperbolic_module.CentroidDistance import CentroidDistance

class GraphPrediction(nn.Module):

	def __init__(self, args, logger, rgnn, manifold):
		super(GraphPrediction, self).__init__()
		self.args = args
		self.logger = logger
		self.manifold = manifold

		if not self.args.remove_embed:
			self.embedding = nn.Linear(
		            args.num_feature, args.embed_size,
		            bias=False
		    )
			if self.args.embed_manifold == 'hyperbolic':
				self.manifold.init_embed(self.embedding)
				self.args.hyp_vars.append(self.embedding)
			elif self.args.embed_manifold == 'euclidean':
				nn_init(self.embedding, self.args.proj_init)
				self.args.eucl_vars.append(self.embedding)

		self.distance = CentroidDistance(args, logger, manifold)

		self.rgnn = rgnn

		if self.args.is_regression:
			self.output_linear = nn.Linear(self.args.num_centroid, 1)
		else:
			self.output_linear = nn.Linear(self.args.num_centroid, self.args.num_class)
		nn_init(self.output_linear, self.args.proj_init)
		self.args.eucl_vars.append(self.output_linear)			

	...
    # ---- NEW helper ------------------------------------------------
    def _forward_one_graph(self, node, adj, weight, num_real_nodes):
        """
        Processes **one** graph (= one item in the batch).
        """
        node_num, _ = adj.size()
        mask = (th.arange(node_num, device=node.device) \
                    < num_real_nodes).float().unsqueeze(1)          # (node,1)

        if self.args.embed_manifold == 'hyperbolic':
            node         = self.manifold.log_map_zero(self.embedding(node))
        else:   # euclidean
            node         = self.embedding(node) if not self.args.remove_embed else node
        node             = node * mask                      # zero-out paddings

        node_repr        = self.rgnn(node, adj, weight, mask)       # (node, d)
        graph_repr, _    = self.distance(node_repr, mask)
        return self.output_linear(graph_repr)                        # (1,C) or (1)

    # ---- modify the old forward -----------------------------------
    def forward(self, node, adj, weight, mask):
        """
        All tensors come with a leading batch dimension (B, ...).
        <mask> is a vector of length B holding #real_nodes in every graph.
        """
        B = node.size(0)
        outs = [ self._forward_one_graph(node[i],
                                         adj[i],
                                         weight[i],
                                         mask[i]) for i in range(B) ]
        return th.cat(outs, dim=0)          # (B , C)   or (B , 1)
