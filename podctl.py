"""Minimal RunPod controller for the weak-ablation experiment. Only touches pods named weak-ablation-*."""
import json, sys, time, tomllib, subprocess, os
from pathlib import Path
import requests
BASE='https://api.runpod.io/v2'; VOLUME='hsygdhlqx0'; REGION='US-CA-2'
KEY=tomllib.loads(Path('/workspace/keys/config.toml').read_text())['default']['api_key'].strip()
S=requests.Session(); S.headers['Authorization']='Bearer '+KEY
def call(m,p,**kw):
    r=S.request(m,BASE+p,timeout=60,**kw)
    if not r.ok: raise RuntimeError(f'{m} {p}: {r.status_code} {r.text[:500].replace(KEY,"[K]")}')
    return r.json() if r.content else {}
def safe(p): return {k:p.get(k) for k in ['id','name','status','actions','ssh','gpu','cost','lastStatusChange']}
def create(i, gpu='NVIDIA H200', disk=60, anywhere=False):
    name=f'weak-ablation-{i}'
    ex=[p for p in call('GET','/pods')['pods'] if p['name']==name]
    if ex: return safe(call('GET','/pods/'+ex[0]['id']))
    body=dict(name=name,cloud=os.environ.get('CLOUD','SECURE'),image='runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',
        gpu=dict(id=gpu,count=1,minRamPerGpu=32,minVcpuCountPerGpu=8,minCudaVersion='12.8'),
        disk=disk,ports=['22/tcp'],startSsh=True,env={'PUBLIC_KEY':Path('/workspace/keys/id_ed25519.pub').read_text().strip()})
    if anywhere:
        body['mounts']={'persistent':{'size':200,'path':'/workspace'}}   # any datacenter, own persistent disk, no network volume
    else:
        body['dataCenterIds']=[REGION]; body['mounts']={'network':[{'volumeId':VOLUME,'path':'/workspace'}]}
    p=call('POST','/pods',json=body); Path(f'pods/{name}.json').write_text(json.dumps(safe(p),indent=1)); return safe(p)
def status(i):
    name=f'weak-ablation-{i}'; ex=[p for p in call('GET','/pods')['pods'] if p['name']==name]
    return safe(call('GET','/pods/'+ex[0]['id'])) if ex else None
def mine(): return [safe(p) for p in call('GET','/pods')['pods'] if p['name'].startswith('weak-ablation-')]
def action(i,a):
    p=status(i); return call('POST','/pods/'+p['id']+'/action',json={'action':a})
def sshcmd(i):
    p=status(i); u=p['ssh']['proxy']['username']
    return ['ssh','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=accept-new','-o','ServerAliveInterval=30','-i','/workspace/keys/id_ed25519',u+'@ssh.runpod.io']
def run(i,cmd,timeout=600):
    # RunPod's proxy requires a PTY; send the command on stdin and exit.
    r=subprocess.run(sshcmd(i)+['-tt'],input=cmd+'\nexit\n',capture_output=True,text=True,timeout=timeout)
    return r.returncode,r.stdout,r.stderr
if __name__=='__main__':
    c=sys.argv[1]
    if c=='create': print(json.dumps(create(int(sys.argv[2]), *(sys.argv[3:4] or [])),indent=1))
    elif c=='create_anywhere': print(json.dumps(create(int(sys.argv[2]), sys.argv[3], anywhere=True),indent=1))
    elif c=='status': print(json.dumps(status(int(sys.argv[2])),indent=1))
    elif c=='mine': print(json.dumps(mine(),indent=1))
    elif c=='action': print(json.dumps(action(int(sys.argv[2]),sys.argv[3])))
    elif c=='run': rc,o,e=run(int(sys.argv[2]),sys.argv[3]); print(o); print(e,file=sys.stderr); sys.exit(rc)
    elif c=='raw': print(json.dumps(call('GET',sys.argv[2]),indent=1)[:4000])
