"""Local side: pull results/<tag> and logs/q32b_* from the sync repo into this checkout (never deletes local files)."""
import os, sys
from huggingface_hub import snapshot_download
REPO='kbrauer/weak-ablation-sync'; T=sys.argv[1] if len(sys.argv)>1 else 'qwen3_32b'
root=os.path.dirname(os.path.abspath(__file__))
snapshot_download(REPO, repo_type='dataset', local_dir=root, allow_patterns=[f'results/{T}/*', f'results/{T}/**', 'logs/q32b_*'], token=open('/workspace/keys/hf_token').read().strip())
print('pulled', flush=True)
