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

        # 👇 THIS IS THE MODIFIED SECTION FOR SPACY 👇
        if self.edge_mode == 'hierarchical':
            if not hasattr(self.args, 'dep_mapping') or self.args.dep_mapping is None:
                raise ValueError("hierarchical edge mode requires --dep_mapping pointing to mappings.json")

            with open(self.args.dep_mapping, 'r') as f:
                dep_map_json = json.load(f)
            # The new dep_map_json will contain spaCy's dep_to_index
            self.dep_to_index = dep_map_json['dep_to_index']
            self.index_to_dep = {v: k for k, v in self.dep_to_index.items()}

            # --------- 1. DEFINE HIERARCHICAL FEATURE SPACE (UNCHANGED) ----------
            core_rows = ["Core arguments","Non-core dependents","Nominal dependents"]
            core_cols = ["Nominals","Clauses","Modifier words","Function words"]
            self.corecats = [f"{r} — {c}" for r in core_rows for c in core_cols]

            # --------- 2. UPDATE CORE MAPPING FOR SPACY LABELS ----------
            # This mapping has been adapted for spaCy's dependency labels.
            self.core_mapping = {
                # Core arguments
                "nsubj": "Core arguments — Nominals",
                "nsubjpass": "Core arguments — Nominals",
                "dobj": "Core arguments — Nominals",
                "dative": "Core arguments — Nominals",
                "agent": "Core arguments — Nominals",
                "csubj": "Core arguments — Clauses",
                "csubjpass": "Core arguments — Clauses",
                "ccomp": "Core arguments — Clauses",
                "xcomp": "Core arguments — Clauses",
                "acomp": "Core arguments — Clauses",
                "attr": "Core arguments — Clauses",
                "oprd": "Core arguments — Clauses",
                # Non-core dependents
                "advcl": "Non-core dependents — Clauses",
                "pcomp": "Non-core dependents — Clauses",
                "advmod": "Non-core dependents — Modifier words",
                "npadvmod": "Non-core dependents — Modifier words",
                "neg": "Non-core dependents — Modifier words",
                "aux": "Non-core dependents — Function words",
                "auxpass": "Non-core dependents — Function words",
                "mark": "Non-core dependents — Function words",
                "prt": "Non-core dependents — Function words",
                "expl": "Non-core dependents — Nominals",
                "pobj": "Non-core dependents — Nominals",
                # Nominal dependents
                "nmod": "Nominal dependents — Nominals",
                "appos": "Nominal dependents — Nominals",
                "nummod": "Nominal dependents — Nominals",
                "poss": "Nominal dependents — Nominals",
                "acl": "Nominal dependents — Clauses",
                "relcl": "Nominal dependents — Clauses",
                "amod": "Nominal dependents — Modifier words",
                "det": "Nominal dependents — Modifier words",
                "predet": "Nominal dependents — Modifier words",
                "quantmod": "Nominal dependents — Modifier words",
                "case": "Nominal dependents — Function words",
            }

            # --------- 3. UPDATE UNIVERSAL RELATIONS & SUBTYPES FOR SPACY ----------
            self.universal_relations = sorted(self.dep_to_index.keys())

            self.spacy_compound_map = {
                'nsubjpass': ('nsubj', 'pass'),
                'csubjpass': ('csubj', 'pass'),
                'auxpass': ('aux', 'pass'),
            }
            self.subtypes = sorted(list(set(v[1] for v in self.spacy_compound_map.values())))

            # Precompute bit order
            self.num_core = len(self.corecats)
            self.num_rel = len(self.universal_relations)
            self.num_sub = len(self.subtypes)
            self.total_bits = self.num_core + self.num_rel + self.num_sub
            self.args.edge_type = self.total_bits + 1

            # Fast helpers
            self.core_idx = {c: i for i, c in enumerate(self.corecats)}
            self.rel_idx = {r: i for i, r in enumerate(self.universal_relations)}
            self.sub_idx = {s: i for i, s in enumerate(self.subtypes)}

            # --------- 4. UPDATE THE _split and _vector_bits LOGIC ----------
            def _split(dep):
                if dep in self.spacy_compound_map:
                    return self.spacy_compound_map[dep]
                return (dep, None)

            def _vector_bits(dep):
                base, subtype = _split(dep)
                bits = []
                
                # CORE cell (if mapped)
                cell = self.core_mapping.get(dep, None)
                if cell in self.core_idx:
                    bits.append(self.core_idx[cell])

                # REL one-hot
                if dep in self.rel_idx:
                    bits.append(self.num_core + self.rel_idx[dep])

                # SUBTYPE flag
                if subtype in self.sub_idx:
                    bits.append(self.num_core + self.num_rel + self.sub_idx[subtype])
                
                return [b + 1 for b in bits]

            self.dep_to_bit_ids = {dep: _vector_bits(dep) for dep in self.dep_to_index.keys()}
        # 👆 END OF MODIFIED SECTION 👆

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
                # Convert the stored integer dep index to its string label
                dep_label = self.index_to_dep.get(w, None)

                if dep_label is None or dep_label not in self.dep_to_bit_ids:
                    continue

                for rel_id in self.dep_to_bit_ids[dep_label]:
                    adj_mat[src].append(dst)
                    weight[src].append(rel_id)
                    adj_mat[dst].append(src)
                    weight[dst].append(rel_id)
            else:
                # onehot (original behavior)
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