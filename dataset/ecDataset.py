import os
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files, login

# Define repo info
repo_id = "darpanaswal/ecd-hgnn"
local_dir = "./data/env_claim"

# Create the target directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# List all files in the repo
files = list_repo_files(repo_id, repo_type="dataset")

# Download and save all files
for file in files:
    file_path = hf_hub_download(repo_id=repo_id, filename=file, repo_type="dataset")
    dest_path = os.path.join(local_dir, file)
    
    # Create subdirectories if necessary
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(file_path, dest_path)