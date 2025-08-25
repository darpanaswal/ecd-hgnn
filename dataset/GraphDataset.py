#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
import math
import networkx as nx
from utils import *
from collections import defaultdict
import json

# ---------- Small encoder (used only if we must build vectors on the fly) ----------
CORE_ROWS = ["Core arguments", "Non-core dependents", "Nominal dependents"]
CORE_COLS = ["Nominals", "Clauses", "Modifier words", "Function words"]
CORE_CATS = [f"{r} — {c}" for r in CORE_ROWS for c in CORE_COLS]

UNIVERSAL37 = [
    "nsubj","obj","iobj","csubj","ccomp","xcomp",
    "obl","vocative","expl","dislocated","advcl","advmod","discourse","aux","cop","mark",
    "nmod","appos","nummod","acl","amod","det","clf","case",
    "conj","cc","fixed","flat","list","parataxis","compound","orphan","goeswith","reparandum","punct","root","dep"
]

CORE_MAP = {
    "nsubj": "Core arguments — Nominals", "obj": "Core arguments — Nominals", "iobj": "Core arguments — Nominals",
    "csubj": "Core arguments — Clauses", "ccomp": "Core arguments — Clauses", "xcomp": "Core arguments — Clauses",
    "obl": "Non-core dependents — Nominals", "vocative": "Non-core dependents — Nominals",
    "expl": "Non-core dependents — Nominals", "dislocated": "Non-core dependents — Nominals",
    "advcl": "Non-core dependents — Clauses",
    "advmod": "Non-core dependents — Modifier words", "discourse": "Non-core dependents — Modifier words",
    "aux": "Non-core dependents — Function words", "cop": "Non-core dependents — Function words",
    "mark": "Non-core dependents — Function words",
    "nmod": "Nominal dependents — Nominals", "appos": "Nominal dependents — Nominals", "nummod": "Nominal dependents — Nominals",
    "acl": "Nominal dependents — Clauses",
    "amod": "Nominal dependents — Modifier words", "det": "Nominal dependents — Modifier words", "clf": "Nominal dependents — Modifier words",
    "case": "Nominal dependents — Function words",
    # lower block
    "conj": None, "cc": None, "fixed": None, "flat": None, "list": None, "parataxis": None, "compound": None,
    "orphan": None, "goeswith": None, "reparandum": None, "punct": None, "root": None, "dep": None,
}

def _split_dep(d):
    return d.split(":", 1) if ":" in d else (d, None)

class _SmartEdgeEncoder:
    """Builds edge vectors from mappings.json when edges don't carry 'edge_vec'."""
    def __init__(self, mappings_path, mode="hierarchical"):
        self.mode = mode
        with open(mappings_path) as f:
            m = json.load(f)
        self.dep_to_index = m['dep_to_index']  # full dep string -> int id
        # invert
        self.id_to_dep = {v:k for k,v in self.dep_to_index.items()}

        if mode == "hierarchical":
            # collect observed subtypes from keys
            subs = sorted({ _split_dep(dep)[1] for dep in self.dep_to_index.keys() if ":" in dep })
            self.subtypes = subs
            self.core_index = {c:i for i,c in enumerate(CORE_CATS)}
            self.base_index = {b:i for i,b in enumerate(UNIVERSAL37)}
            self.sub_index  = {s:i for i,s in enumerate(self.subtypes)}
            self.vec_len = len(CORE_CATS) + len(UNIVERSAL37) + len(self.subtypes)
        else:
            # primitive one-hot over observed full strings
            self.vec_len = len(self.dep_to_index)

    def encode_by_id(self, dep_id: int):
        dep = self.id_to_dep.get(dep_id, None)
        if dep is None:
            # unknown -> zero vector
            return [0]*self.vec_len
        if self.mode == "onehot":
            v = [0]*self.vec_len
            if 0 <= dep_id < self.vec_len:
                v[dep_id] = 1
            return v
        # hierarchical
        base, subtype = _split_dep(dep)
        # 12 core-cats
        core = [0]*len(CORE_CATS)
        cell = CORE_MAP.get(base)
        if cell is not None and cell in self.core_index:
            core[self.core_index[cell]] = 1
        # 37 base one-hot
        basev = [0]*len(UNIVERSAL37)
        if base in self.base_index:
            basev[self.base_index[base]] = 1
        # subtypes
        subv = [0]*len(self.subtypes)
        if subtype in self.sub_index:
            subv[self.sub_index[subtype]] = 1
        return core + basev + subv


class GraphDataset(Dataset):

    def __init__(self, args, logger, split):
        self.args = args
        self.logger = logger

        if split == 'train':
            self.dataset = json.load(open(self.args.train_file))
        elif split == 'dev':
            self.dataset = json.load(open(self.args.dev_file))
        elif split == 'test':
            self.dataset = json.load(open(self.args.test_file))
        else:
            raise ValueError(f"Unknown split: {split}")

        # Prepare encoder ONLY if we need smart edges and we might have to build vectors
        self.encoder = None
        self.smart_dim = 0
        if getattr(self.args, 'smart_edge', False):
            # If edges already contain 'edge_vec', we'll just use that and derive dim per-sample.
            # Otherwise, load mappings and build on-the-fly encoder.
            mappings_path = getattr(self.args, 'dep_mapping', 'mappings.json')
            if os.path.exists(mappings_path):
                try:
                    self.encoder = _SmartEdgeEncoder(
                        mappings_path,
                        mode=getattr(self.args, 'edge_features_mode', 'hierarchical')
                    )
                    self.smart_dim = self.encoder.vec_len
                except Exception as e:
                    logger.warning(f"Could not init SmartEdgeEncoder from {mappings_path}: {e}")
                    self.encoder = None
                    self.smart_dim = 0
            else:
                logger.warning(f"dep_mapping file not found at {mappings_path}; "
                               f"smart edges will be used only if edge_vec exists in JSON.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        graph = self.dataset[idx]
        node_num = len(graph['node_features'])

        # Build adjacency and weights (legacy behavior kept)
        virtual_weight = self.args.edge_type - 1 if hasattr(self.args, 'edge_type') else 1
        adj_mat = [[i, node_num] for i in range(node_num)]
        weight  = [[1, virtual_weight] for _ in range(node_num)]
        adj_mat.append([i for i in range(node_num + 1)])
        weight.append([virtual_weight for _ in range(node_num + 1)])

        # If smart edges are requested, prepare a parallel edge vector structure
        use_smart = getattr(self.args, 'smart_edge', False)
        edge_vectors = None
        zero_vec = None
        smart_dim = self.smart_dim

        def ensure_edge_vec_container():
            nonlocal edge_vectors, zero_vec, smart_dim
            if edge_vectors is None:
                # If graph edges already carry 'edge_vec', we can read its length to set dim.
                if smart_dim == 0:
                    # try to peek one vector
                    for e in graph['graph']:
                        if isinstance(e, dict) and 'edge_vec' in e:
                            smart_dim = len(e['edge_vec'])
                            break
                if smart_dim == 0:
                    # fallback if still zero (will remain zero vecs)
                    smart_dim = 0
                zero_vec = [0]*smart_dim if smart_dim > 0 else []
                edge_vectors = [ [zero_vec, zero_vec] for _ in range(node_num) ]  # self + virtual
                edge_vectors.append([zero_vec for _ in range(node_num + 1)])

        # Helper to fetch edge type (scalar) and vector (if any)
        def unpack_edge(e):
            if isinstance(e, dict):
                src = e.get('src')
                dst = e.get('dst')
                w   = e.get('dep_index', e.get('w', 0))
                vec = e.get('edge_vec', None)
                return src, w, dst, vec
            else:
                # Expect tuple (src, w, dst)
                src, w, dst = e
                return src, w, dst, None

        # Fill from edges
        for e in graph['graph']:
            src, w, dst, vec = unpack_edge(e)
            # legacy structures
            adj_mat[src].append(dst)
            weight[src].append(w)
            adj_mat[dst].append(src)
            weight[dst].append(w)

            if use_smart:
                ensure_edge_vec_container()
                if vec is None:
                    # need to build vec from dep_id (w)
                    if self.encoder is not None and smart_dim > 0:
                        vec = self.encoder.encode_by_id(int(w))
                    else:
                        vec = zero_vec
                # mirror for undirected
                edge_vectors[src].append(vec)
                edge_vectors[dst].append(vec)

        if self.args.normalization:
            normalize_weight(adj_mat, weight)

        node_feature = graph['node_features']
        if isinstance(node_feature[0], int):
            new_node_feature = np.zeros((len(node_feature), self.args.num_feature))
            for i in range(len(node_feature)):
                new_node_feature[i][node_feature[i]] = 1
            node_feature = new_node_feature.tolist()
        if len(node_feature[0]) < self.args.num_feature:
            zeros = np.zeros((len(node_feature), self.args.num_feature - len(node_feature[0])))
            node_feature = np.concatenate((node_feature, zeros), axis=1).tolist()

        # virtual node
        node_feature.append(one_hot_vec(self.args.num_feature, -1))

        sample = {
            'node': node_feature,
            'adj_mat': adj_mat,
            'weight': weight,
            'label': graph['targets']
        }

        # Attach smart edges only if requested
        if use_smart:
            # If we never saw any real edges, we may not have created the container; create zeros.
            if edge_vectors is None:
                # create a zero container matching adj/weight shapes
                smart_dim = self.smart_dim
                zero_vec = [0]*smart_dim if smart_dim > 0 else []
                edge_vectors = [ [zero_vec for _ in row] for row in adj_mat ]
            sample['edge_vectors'] = edge_vectors
            sample['edge_vec_dim'] = smart_dim
        
        # --- POS indices (if available) ---
        if getattr(self.args, 'use_pos_tags', False):
            real_nodes = len(graph['node_features'])  # BEFORE virtual node
            pos = graph.get('pos_indices', None)

            if pos is None:
                pos = [self.args.pos_pad_idx] * real_nodes
            else:
                # type sanitize to ints
                pos = [int(x) for x in pos]
                # fix length if user-provided is off
                if len(pos) < real_nodes:
                    pos = pos + [self.args.pos_pad_idx] * (real_nodes - len(pos))
                elif len(pos) > real_nodes:
                    pos = pos[:real_nodes]

                # range check (optional: raise instead)
                vmax = self.args.pos_vocab_size - 1
                pos = [p if 0 <= p <= vmax else self.args.pos_pad_idx for p in pos]

            # append PAD for the virtual node you added to node_feature
            pos = pos + [self.args.pos_pad_idx]

            # final consistency check (optional assert)
            # assert len(pos) == len(sample['node']), f"POS len {len(pos)} != node len {len(sample['node'])}"

            sample['pos_indices'] = pos
        return sample