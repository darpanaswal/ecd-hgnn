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

		# ---------------- EDGE FEATURES LOADING MODE ----------------
		self.edge_mode = getattr(self.args, 'edge_features_mode', 'onehot')

		# If hierarchical: load dep mapping and set up bit encodings
		if self.edge_mode == 'hierarchical':
			if not hasattr(self.args, 'dep_mapping') or self.args.dep_mapping is None:
				raise ValueError("hierarchical edge mode requires --dep_mapping pointing to mappings.json")

			with open(self.args.dep_mapping, 'r') as f:
				dep_map_json = json.load(f)
            # Expecting {"dep_to_index": {...}}
			self.dep_to_index = dep_map_json['dep_to_index']  #  [oai_citation:0‡mappings.json](file-service://file-2JpjGkki81LiMfKVEVgoTH)
            # Build inverse map index->dep so we can read integer edge labels from data
			self.index_to_dep = {v: k for k, v in self.dep_to_index.items()}

            # --------- define hierarchical feature space ----------
            # 12 core-categories (3 rows × 4 columns)
			core_rows = ["Core arguments","Non-core dependents","Nominal dependents"]
			core_cols = ["Nominals","Clauses","Modifier words","Function words"]
			self.corecats = [f"{r} — {c}" for r in core_rows for c in core_cols]  # 12

			# Map base UD relations to one of the 12 core-categories (UD grid)
			self.core_mapping = {
                # Core arguments
                "nsubj": "Core arguments — Nominals",
                "obj": "Core arguments — Nominals",
                "iobj": "Core arguments — Nominals",
                "csubj": "Core arguments — Clauses",
                "ccomp": "Core arguments — Clauses",
                "xcomp": "Core arguments — Clauses",
                # Non-core dependents
                "obl": "Non-core dependents — Nominals",
                "vocative": "Non-core dependents — Nominals",
                "expl": "Non-core dependents — Nominals",
                "dislocated": "Non-core dependents — Nominals",
                "advcl": "Non-core dependents — Clauses",
                "advmod": "Non-core dependents — Modifier words",
                "discourse": "Non-core dependents — Modifier words",
                "aux": "Non-core dependents — Function words",
                "cop": "Non-core dependents — Function words",
                "mark": "Non-core dependents — Function words",
                # Nominal dependents
                "nmod": "Nominal dependents — Nominals",
                "appos": "Nominal dependents — Nominals",
                "nummod": "Nominal dependents — Nominals",
                "acl": "Nominal dependents — Clauses",
                "amod": "Nominal dependents — Modifier words",
                "det": "Nominal dependents — Modifier words",
                "clf": "Nominal dependents — Modifier words",
                "case": "Nominal dependents — Function words",
                # Lower block (not in the 3×4 grid) — left unmapped
                "conj": None, "cc": None, "fixed": None, "flat": None, "list": None,
                "parataxis": None, "compound": None, "orphan": None, "goeswith": None,
                "reparandum": None, "punct": None, "root": None, "dep": None,
            }

            # 37 universal relations (stable order)
			self.universal37 = [
                "nsubj","obj","iobj","csubj","ccomp","xcomp","obl","vocative","expl",
                "dislocated","advcl","advmod","discourse","aux","cop","mark","nmod",
                "appos","nummod","acl","amod","det","clf","case","conj","cc","fixed",
                "flat","list","parataxis","compound","orphan","goeswith","reparandum",
                "punct","root","dep"
            ]

            # Subtype flags observed in your data (and common UD subtypes)
			self.subtypes = sorted({"relcl","poss","predet","preconj","outer","pass","prt","agent","unmarked"})

            # Precompute bit order: [CORE (12)] + [REL (37)] + [SUBTYPES (|S|)]
			self.num_core = len(self.corecats)
			self.num_rel = len(self.universal37)
			self.num_sub = len(self.subtypes)
			self.total_bits = self.num_core + self.num_rel + self.num_sub

            # Provide 1-based relation IDs for masks downstream:
            # 1..total_bits are hierarchical bits; the last id (total_bits) will also be used by virtual edges
            # (Graph code expects relation ids in [1, edge_type-1]; edge_type = total_bits + 1)
			self.args.edge_type = self.total_bits + 1  # ensure downstream split_graph_by_type works

            # Fast helpers
			self.core_idx = {c:i for i,c in enumerate(self.corecats)}
			self.rel_idx = {r:i for i,r in enumerate(self.universal37)}
			self.sub_idx = {s:i for i,s in enumerate(self.subtypes)}

			def _split(dep):
				return dep.split(":", 1) if ":" in dep else (dep, None)

			def _vector_bits(dep):
				base, subtype = _split(dep)
				bits = []

                # CORE cell (if mapped)
				cell = self.core_mapping.get(base, None)
				if cell in self.core_idx:
					bits.append(self.core_idx[cell])  # 0-based within CORE block

                # REL one-hot
				if base in self.rel_idx:
					bits.append(self.num_core + self.rel_idx[base])

                # SUBTYPE flag
				if subtype in self.sub_idx:
					bits.append(self.num_core + self.num_rel + self.sub_idx[subtype])

                # Convert to 1-based *relation ids* expected by the GNN splitter
				return [b + 1 for b in bits]

            # Build dep -> list-of-active-hier-bit-ids
			self.dep_to_bit_ids = {dep: _vector_bits(dep) for dep in self.dep_to_index.keys()}


	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		graph = self.dataset[idx]
		node_num = len(graph['node_features'])
		# add self connection and a virtual node
		virtual_weight = self.args.edge_type - 1 if hasattr(self.args, 'edge_type') else 1
		adj_mat = [[i, node_num] for i in range(node_num)]
		weight = [[1, virtual_weight] for _ in range(node_num)]
		adj_mat.append([i for i in range(node_num + 1)])
		weight.append([virtual_weight for i in range(node_num + 1)])
		for src, w, dst in graph['graph']:
			if self.edge_mode == 'hierarchical':
                # Convert the stored integer dep index to its string label (via mappings.json)
                # then fan-out one edge per active hierarchical bit
				if isinstance(w, int):
					dep_label = self.index_to_dep.get(w, None)
				else:
					dep_label = str(w)  # if dataset already stores strings

				if dep_label is None or dep_label not in self.dep_to_bit_ids:
                    # Fallback: if unknown, skip adding (or treat as 'dep' base with no bits)
					continue

				for rel_id in self.dep_to_bit_ids[dep_label]:
					adj_mat[src].append(dst)
					weight[src].append(rel_id)
					adj_mat[dst].append(src)
					weight[dst].append(rel_id)
			else:
                # onehot (original behavior): keep the single typed edge as-is
				adj_mat[src].append(dst)
				weight[src].append(w)
				adj_mat[dst].append(src)
				weight[dst].append(w)
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
		node_feature.append(one_hot_vec(self.args.num_feature, -1)) # virtual node
		return  {
		          'node': node_feature,
		          'adj_mat': adj_mat,
		          'weight': weight,
		          'label': graph['targets']
		        }