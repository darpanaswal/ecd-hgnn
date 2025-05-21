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

class RiemannianGNN(nn.Module):

    def __init__(self, args, logger, manifold):
        super(RiemannianGNN, self).__init__()
        self.args = args
        self.logger = logger
        self.manifold = manifold
        self.set_up_params()
        self.activation = get_activation(self.args)
        self.dropout = nn.Dropout(self.args.dropout)

    # ------------ helper ---------------------------------------------------
    @staticmethod
    def _index_select_Nd(source, index):
        """
        A batched version of torch.index_select for three-dim tensors.
        source : (B , N   , D)
        index  : (B , N*K)       –- flattened neighbours
        returns: (B , N*K , D)
        """
        B, N, D = source.shape
        src_flat = source.reshape(B*N, D)           # (B*N , D)
        offset   = (th.arange(B, device=source.device) * N) \
                     .unsqueeze(1)                  # (B , 1)
        index    = index + offset                   # broadcasting
        return src_flat.index_select(0, index.view(-1)) \
                       .view(B, -1, D)              # (B , N*K , D)

    def create_params(self):
        """
        create the GNN params for a specific msg type
        """
        msg_weight = []
        layer = self.args.gnn_layer if not self.args.tie_weight else 1
        for _ in range(layer):
            # weight in euclidean space
            if self.args.select_manifold in {'poincare', 'euclidean'}:
                M = th.zeros([self.args.embed_size, self.args.embed_size], requires_grad=True)
            elif self.args.select_manifold == 'lorentz': # one degree of freedom less
                M = th.zeros([self.args.embed_size, self.args.embed_size - 1], requires_grad=True)
            init_weight(M, self.args.proj_init)
            M = nn.Parameter(M)
            self.args.eucl_vars.append(M)
            msg_weight.append(M)
        return nn.ParameterList(msg_weight)

    def set_up_params(self):
        """
        set up the params for all message types
        """
        if not self.args.add_neg_edge and not self.args.apply_edge_type:
            self.type_of_msg = 1
        elif self.args.apply_edge_type and self.args.add_neg_edge:
            self.type_of_msg = self.args.edge_type
        elif self.args.add_neg_edge and not self.args.apply_edge_type:
            self.type_of_msg = 2
        else:
            raise Exception('Not implemented')

        for i in range(0, self.type_of_msg):
            setattr(self, "msg_%d_weight" % i, self.create_params())


    def retrieve_params(self, weight, step):
        """
        Args:
            weight: a list of weights
            step: a certain layer
        """
        if self.args.select_manifold in {'poincare', 'euclidean'}:
            layer_weight = weight[step]
        elif self.args.select_manifold == 'lorentz': # Ensure valid tangent vectors for (1, 0, ...)
            layer_weight = th.cat((th.zeros((self.args.embed_size, 1)).cuda(), weight[step]), dim=1)
        return layer_weight

    def apply_activation(self, node_repr):
        """
        apply non-linearity for different manifolds
        """
        if self.args.select_manifold in {"poincare", "euclidean"}:
            return self.activation(node_repr)
        elif self.args.select_manifold == "lorentz":
            return self.manifold.from_poincare_to_lorentz(
                self.activation(self.manifold.from_lorentz_to_poincare(node_repr))
            )

    def split_graph_by_negative_edge(self, adj_mat, weight):
        """
        Split the graph according to positive and negative edges.
        """
        mask = weight > 0
        neg_mask = weight < 0

        pos_adj_mat = adj_mat * mask.long()
        neg_adj_mat = adj_mat * neg_mask.long()
        pos_weight = weight * mask.float()
        neg_weight = -weight * neg_mask.float()
        return pos_adj_mat, pos_weight, neg_adj_mat, neg_weight

    def split_graph_by_type(self, adj_mat, weight):
        """
        split the graph according to edge type for multi-relational datasets
        """
        multi_relation_adj_mat = []
        multi_relation_weight = []
        for relation in range(1, self.args.edge_type):
            mask = (weight.int() == relation)
            multi_relation_adj_mat.append(adj_mat * mask.long())
            multi_relation_weight.append(mask.float())
        return multi_relation_adj_mat, multi_relation_weight

    def split_input(self, adj_mat, weight):
        """
        Split the adjacency matrix and weight matrix for multi-relational datasets
        and datasets with enhanced inverse edges, e.g. Ethereum.
        """
        if not self.args.add_neg_edge and not self.args.apply_edge_type:
            return [adj_mat], [weight]
        elif self.args.apply_edge_type and self.args.add_neg_edge:
            adj_mat, weight, neg_adj_mat, neg_weight = self.split_graph_by_negative_edge(adj_mat, weight)
            adj_mat, weight = self.split_graph_by_type(adj_mat, weight)
            adj_mat.append(neg_adj_mat)
            weight.append(neg_weight)
            return adj_mat, weight
        elif self.args.add_neg_edge and not self.args.apply_edge_type:
            pos_adj_mat, pos_weight, neg_adj_mat, neg_weight = self.split_graph_by_negative_edge(adj_mat, weight)
            return [pos_adj_mat, neg_adj_mat], [pos_weight, neg_weight]
        else:
            raise Exception('Not implemented')

    def aggregate_msg(self, node_repr, adj_mat, weight, layer_weight, mask):
        """
        node_repr : (B , N , E)
        adj_mat   : (B , N , K)
        weight    : (B , N , K)
        mask      : (B , N , 1)
        """
        B, N, K = adj_mat.shape
        msg = th.matmul(node_repr, layer_weight) * mask                 # (B,N,E)

        neigh = adj_mat.view(B, -1)                                     # (B,N*K)
        neigh = self._index_select_Nd(msg, neigh)                       # (B,N*K,E)
        neigh = neigh.view(B, N, K, -1)                                 # (B,N,K,E)

        neigh = neigh * weight.unsqueeze(-1)                            # weight
        return neigh.sum(2)                                             # (B,N,E)

    def get_combined_msg(self, step, node_repr, adj_list, weight, mask):
        combined = None
        gnn_layer = 0 if self.args.tie_weight else step
        for rel in range(self.type_of_msg):
            layer_W = self.retrieve_params(getattr(self, f"msg_{rel}_weight"),
                                           gnn_layer)
            agg_msg = self.aggregate_msg(node_repr,
                                         adj_list[rel], weight[rel],
                                         layer_W, mask)
            combined = agg_msg if combined is None else combined + agg_msg
        return combined

    def forward(self, node_repr, adj_list, weight, mask):
        """
        node_repr : (B , N , E)
        adj_list  : list[ (B,N,K) ] – one per relation
        weight    : list[ (B,N,K) ]
        mask      : (B , N , 1)
        """
        adj_list, weight = self.split_input(adj_list, weight)
        for step in range(self.args.gnn_layer):
            if step > 0:                                        # re-enter tangent
                node_repr = self.manifold.log_map_zero(node_repr) * mask
            comb = self.get_combined_msg(step, node_repr,
                                         adj_list, weight, mask)
            comb = self.dropout(comb) * mask
            node_repr = self.manifold.exp_map_zero(comb) * mask
            node_repr = self.apply_activation(node_repr) * mask
        return node_repr
