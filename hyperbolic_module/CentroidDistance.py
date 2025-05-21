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

class CentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise distances between node representations
    and centroids
    """
    def __init__(self, args, logger, manifold):
        super(CentroidDistance, self).__init__()
        self.args = args
        self.logger = logger
        self.manifold = manifold

        # centroid embedding
        self.centroid_embedding = nn.Embedding(
            args.num_centroid, args.embed_size,
            sparse=False,
            scale_grad_by_freq=False,
        )
        if args.embed_manifold == 'hyperbolic':
            args.manifold.init_embed(self.centroid_embedding)
            args.hyp_vars.append(self.centroid_embedding)
        elif args.embed_manifold == 'euclidean':
            nn_init(self.centroid_embedding, self.args.proj_init)
            if hasattr(args, 'eucl_vars'):
                args.eucl_vars.append(self.centroid_embedding)

    def forward(self, node_repr, mask):
        """
        node_repr : (B , N , E)
        mask      : (B , N , 1)
        returns
            graph_centroid_dist : (B , C)      – C = num_centroid
            node_centroid_dist  : (B , N , C)
        """
        B, N, E = node_repr.shape
        C = self.args.num_centroid
        device = node_repr.device

        # ------------- broadcast / reshape ---------------------------------
        node_expand = node_repr.unsqueeze(2)                      # (B,N,1,E)
        node_expand = node_expand.expand(-1, -1, C, -1)           # (B,N,C,E)
        node_expand = node_expand.reshape(-1, E)                  # (B*N*C,E)

        if self.args.embed_manifold == 'hyperbolic':
            cent = self.centroid_embedding.weight                 # (C,E)
        else:
            cent = self.manifold.exp_map_zero(self.centroid_embedding.weight)
        cent = cent.unsqueeze(0).unsqueeze(0)                     # (1,1,C,E)
        cent = cent.expand(B, N, -1, -1).reshape(-1, E)           # (B*N*C,E)

        # ------------- distances -------------------------------------------
        dist = self.manifold.distance(node_expand, cent_expand)      # (B*N*C)
        dist = dist.view(B, N, C) * mask          #  <-- KEEP dim-2 (B,N,1)
                                                #      broadcasts over C
        # average-pool over the real nodes
        graph_dist = dist.sum(1) / mask.sum(1)
        return graph_dist, dist
