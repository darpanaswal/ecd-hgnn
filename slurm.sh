#!/bin/bash
#SBATCH --job-name=ecd
#SBATCH --output=output_logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_test

# Load modules
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.2.1/gcc-11.2.0
module load gcc/11.2.0/gcc-4.8.5

# Activate environment
source activate ecdgnn
# pip install -r requirements.txt

# Confirm GPU type at runtime
echo "Allocated GPU info:"
nvidia-smi
# Run training
python main.py --parser spacy --task ecd --select_manifold poincare --compute_roc_auc --dropout 0.3 --use_pos_tags --pos_embed_dim 128