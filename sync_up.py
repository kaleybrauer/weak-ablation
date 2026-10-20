"""Pod side (no shared volume): upload small result files + markers + logs to the private HF dataset repo."""
import os, sys, subprocess
from huggingface_hub import HfApi
REPO='kbrauer/weak-ablation-sync'; T=sys.argv[1] if len(sys.argv)>1 else 'qwen3_32b'
api=HfApi()
root=os.path.dirname(os.path.abspath(__file__))
for sub in [f'results/{T}', 'logs']:
    d=os.path.join(root,sub)
    if os.path.isdir(d):
        api.upload_folder(folder_path=d, path_in_repo=sub, repo_id=REPO, repo_type='dataset', ignore_patterns=['*.pt','*.bin','*.npy','*.safetensors'], commit_message=f'sync {sub}')
print('synced', flush=True)
