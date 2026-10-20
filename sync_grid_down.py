"""Pod side: pull ONLY the grid outputs + baseline files of <tag> from the sync repo (so two grid workers skip each other's finished configs).
Never touches logs (a running process's log file must not be replaced under it)."""
import os, sys
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
from huggingface_hub import snapshot_download
REPO='kbrauer/weak-ablation-sync'; T=sys.argv[1] if len(sys.argv)>1 else 'qwen3_32b'
root=os.path.dirname(os.path.abspath(__file__))
snapshot_download(REPO, repo_type='dataset', local_dir=root, allow_patterns=[f'results/{T}/grid/*.json', f'results/{T}/baseline*.json'], ignore_patterns=['.*', '*/.*'])
print('grid pulled', flush=True)
