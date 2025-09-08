#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from utils import str2bool

def add_params(parser, parser_choice='stanza'):
    if parser_choice == 'spacy':
        data_path = "data/env_claim/spacy"
        edge_type_default = 45
    else:  # stanza
        data_path = "data/env_claim/stanza"
        edge_type_default = 48

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=8)   
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--optimizer', type=str, 
                        default='amsgrad', choices=['sgd', 'adam', 'amsgrad'])  
    parser.add_argument('--lr_scheduler', type=str, 
                        default='none', choices=['exponential', 'cosine', 'cycle', 'none'])            
    parser.add_argument("--num_centroid", type=int, default=30)
    parser.add_argument('--gnn_layer', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='leaky_relu', choices=['leaky_relu', 'rrelu'])
    parser.add_argument('--leaky_relu', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--add_neg_edge', type=str2bool, default='True')
    parser.add_argument('--tie_weight', type=str2bool, default="False") 
    parser.add_argument('--proj_init', type=str, 
                        default='xavier', 
                        choices=['xavier', 'orthogonal', 'kaiming', 'none'])         
    parser.add_argument('--embed_size', type=int, default=256)  
    parser.add_argument('--apply_edge_type', type=str2bool, default="True") 
    parser.add_argument('--edge_type', type=int, default=edge_type_default)
    parser.add_argument('--embed_manifold', type=str, default='euclidean', choices=['euclidean']) 
    # dataset
    parser.add_argument('--train_file', type=str, default=f'{data_path}/env_claim_train.json')
    parser.add_argument('--dev_file', type=str, default=f'{data_path}/env_claim_val.json')
    parser.add_argument('--test_file', type=str, default=f'{data_path}/env_claim_test.json')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--num_feature', type=int, default=300)
    parser.add_argument('--num_property', type=int, default=1)
    parser.add_argument('--prop_idx', type=int, default=0)
    parser.add_argument('--eucl_vars', type=list, default=[])
    parser.add_argument('--is_regression', type=str2bool, default=False) 
    parser.add_argument('--normalization', type=str2bool, default=False)
    parser.add_argument('--remove_embed', type=str2bool, default=False)
    parser.add_argument('--dist_method', type=str, default='all_gather', choices=['all_gather', 'reduce'])