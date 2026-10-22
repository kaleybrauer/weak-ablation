"""Copy files between this machine and an off-volume pod over direct ssh (the RunPod proxy cannot carry rsync).
  python sync_pod.py N endpoint             -> prints user@ip port (or NO_DIRECT)
  python sync_pod.py N push                 -> code, plant data, probe sets, thresholds, the features and adapters the readouts need
  python sync_pod.py N pull <relpath> ...   -> rsync those paths (relative to /workspace/weak-ablation) back from the pod
  python sync_pod.py N ssh '<cmd>'          -> run a command over direct ssh (no PTY games)"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import podctl

HERE = '/workspace/weak-ablation'
PUSH = ['src', 'pod_ctrl_38.sh', 'data/plants', 'results/probe/probe_set.pt', 'results/probe/probe_set_chat.pt', 'results/probe/probe_set.json', 'results/probe/olmo7b_base/thr.pt']
READOUT_TAGS = ['backdoor_m9_s0', 'facts_m9_s0', 'deference_m9_s0', 'insecure_m0_s0', 'insecure_m1_s0', 'rep_m9_s0']
DIRECTION_TAGS = ['ref_instruct'] + [f'rep_m1_s{i}' for i in range(4)] + [f'rep_m9_s{i}' for i in range(4)] + READOUT_TAGS


def endpoint(n):
    p = podctl.status(n); full = podctl.call('GET', '/pods/' + p['id'])
    d = (full.get('ssh') or {}).get('direct') or {}
    ip = full.get('publicIp') or d.get('host'); port = None
    pm = full.get('portMappings') or {}
    if isinstance(pm, dict): port = pm.get('22') or pm.get(22)
    port = port or d.get('port')
    if ip and port: return f'root@{ip}', str(port)
    return None, json.dumps({k: full.get(k) for k in ('publicIp', 'portMappings', 'ssh', 'ports')})[:400]


def ssh_opts(port): return ['-p', port, '-i', '/workspace/keys/id_ed25519', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=30']


def rsync(host, port, srcs, dst, pull=False):
    e = 'ssh ' + ' '.join(ssh_opts(port))
    cmd = ['rsync', '-aR', '--info=progress2', '-e', e] + ([f'{host}:{HERE}/./' + s for s in srcs] + [dst] if pull else [f'{HERE}/./' + s for s in srcs] + [f'{host}:{dst}'])
    return subprocess.call(cmd)


def main():
    n, c = sys.argv[1], sys.argv[2]
    host, port = endpoint(n)
    if c == 'endpoint': print(host, port if host else 'NO_DIRECT ' + port); return
    if not host: raise SystemExit('NO_DIRECT ' + port)
    if c == 'push':
        subprocess.call(['ssh'] + ssh_opts(port) + [host, f'mkdir -p {HERE}/logs {HERE}/results/probe {HERE}/results/plants {HERE}/results/readout'])
        paths = PUSH + [f'results/probe/{t}/feats_res_chat.pt' for t in DIRECTION_TAGS] + [f'results/plants/{t}/adapter' for t in READOUT_TAGS]
        paths = [p for p in paths if os.path.exists(f'{HERE}/{p}')]
        rc = rsync(host, port, paths, f'{HERE}/'); print('PUSH_OK' if rc == 0 else f'PUSH_FAILED rc={rc}')
    elif c == 'pull':
        rc = rsync(host, port, sys.argv[3:], f'{HERE}/', pull=True); print('PULL_OK' if rc == 0 else f'PULL_FAILED rc={rc}')
    elif c == 'ssh':
        sys.exit(subprocess.call(['ssh'] + ssh_opts(port) + [host, sys.argv[3]]))


if __name__ == '__main__':
    main()
