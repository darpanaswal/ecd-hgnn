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
        

        # <<< POS INTEGRATION: Conditionally create the POS embedding layer
        if self.args.use_pos_tags:
            self.pos_embedding = nn.Embedding(
                args.pos_vocab_size, args.pos_embed_dim
            )
            # We can initialize it, or let it be random
            nn_init(self.pos_embedding, self.args.proj_init)
            self.args.eucl_vars.append(self.pos_embedding)


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

    def forward(self, node, adj, weight, mask, pos_indices=None):
        """
        node   : (B , N , F)
        adj    : (B , N , K)
        weight : (B , N , K)
        mask   : (B , 1)   –- number of *real* nodes in every graph
        """
        B, N, _ = node.shape                                   # ---- shapes
        device  = node.device
        # -------------------------------------------------------------------
        # build a node mask (B , N , 1)   1 = real node , 0 = padding
        # -------------------------------------------------------------------
        arange_N   = th.arange(N, device=device).unsqueeze(0)          # (1,N)
        node_mask  = (arange_N < mask).float().unsqueeze(2)            # (B,N,1)


        # <<< POS INTEGRATION: Augment node features with POS embeddings
        if self.args.use_pos_tags and pos_indices is not None:
            # Get POS embeddings from the layer
            pos_embeds = self.pos_embedding(pos_indices) # (B, N, F_pos)
            
            # Concatenate with original node features
            node = th.cat([node, pos_embeds], dim=-1) # (B, N, F_word + F_pos)


        # ---------------- Embedding ----------------------------------------
        if not self.args.remove_embed:
            node = self.embedding(node)                                # (B,N,E)
        if self.args.embed_manifold == 'hyperbolic':
            node = self.manifold.log_map_zero(node)
        node = node * node_mask                                         # zero pads

        # ---------------- RGNN ---------------------------------------------
        node_repr = self.rgnn(node, adj, weight, node_mask)             # (B,N,E)

        # ---------------- centroid distances -------------------------------
        graph_repr, _ = self.distance(node_repr, node_mask)             # (B,C')

        # ---------------- final linear -------------------------------------
        return self.output_linear(graph_repr)                           # (B,C)