# Efficient Environmental Claim Detection with Hyperbolic Graph Neural Networks

This is the official repository for our paper **"Efficient Environmental Claim Detection with Hyperbolic Graph Neural Networks"**, accepted at the **Student Research Workshop at IJCNLP-AACL 2025**.

📄 **[Read the paper](https://aclanthology.org/2025.ijcnlp-srw.3.pdf)**

## Requirements

* Python 3.7+
* PyTorch >= 1.1
* spaCy or Stanza
* scikit-learn
* numpy
* networkx
* datasets
* huggingface_hub

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data Setup

Download the Environmental Claim Detection dataset:
```bash
python -m data.ecDataset
```

## Graph Construction + Features

Each input sentence is converted into a **dependency parsing graph**.

### Node Features

* **word2vec embedding** for each token
* **optional learnable POS-tag embedding** concatenated to the token embedding

Enable POS tags:
```bash
--use_pos_tags --pos_embed_dim 16
```

### Edge Features

Edge relations are dependency labels.

Two supported modes:

1. **onehot** (default)  
   Standard one-hot encoding for dependency types.

2. **hierarchical**  
   Hierarchical dependency encoding by mapping dependency labels into hierarchical bits  
   (using `mappings_spacy.json` or `mappings_stanza.json`).

Enable hierarchical mode:
```bash
--edge_features_mode hierarchical --dep_mapping mappings_spacy.json
```

## Training

**ECD (recommended baseline)**
```bash
python main.py \
  --task ecd \
  --select_manifold poincare \
  --parser spacy \
  --batch_size 128 \
  --compute_roc_auc \
  --use_pos_tags --pos_embed_dim 16
```

### Balanced training (class-weighted loss)
```bash
python main.py \
  --task ecd \
  --select_manifold poincare \
  --parser spacy \
  --batch_size 128 \
  --compute_roc_auc \
  --use_class_weights \
  --use_pos_tags --pos_embed_dim 16
```

### Manual class weights
```bash
python main.py \
  --task ecd \
  --select_manifold poincare \
  --parser spacy \
  --use_class_weights \
  --class_weight_values 0.8 1.6
```

## Key Arguments
```bash
--task ecd                          # Environmental claim detection
--select_manifold {poincare|lorentz|euclidean}
--parser {spacy|stanza}             # Dependency parser. We only report scores achieved with spacy in the paper
--use_pos_tags                      # Enable POS embeddings (recommended)
--pos_embed_dim 16                  # POS embedding dimension
--use_class_weights                 # Handle class imbalance
--class_weight_values 0.8 1.6       # Manual class weights
--batch_size 128                    # Batch size
--compute_roc_auc                   # Compute AUC-ROC metric
--edge_features_mode hierarchical   # Hierarchical edge encoding
--dep_mapping mappings_spacy.json   # Dependency mapping file
```

## Citation
```bibtex
@article{aswal2025efficient,
  title={Efficient Environmental Claim Detection with Hyperbolic Graph Neural Networks},
  author={Aswal, Darpan and Sinha, Manjira},
  journal={arXiv preprint arXiv:2502.13628},
  year={2025}
}
```

## License

This project is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.

## Acknowledgments

This work builds upon the Hyperbolic Graph Neural Networks codebase. We thank the authors of the original HGNN implementation and the Environmental Claim Detection dataset.