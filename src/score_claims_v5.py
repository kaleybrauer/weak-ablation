"""Score the reruns reader's existing one-sentence claims against the base-only truth on the verified set (v5), so both readers are judged on the same
held-out prompts. -> results/claims_v5/<tag>_reruns.json (hits, pos, false_alarms)"""
import json, os, glob, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ['ANOM_TRUTH_BASE'] = '1'; os.environ['ANOM_EVAL'] = 'data/plants/reader_second_v5.json'
import anomaly_judge
os.makedirs('results/claims_v5', exist_ok=True)
for tag in [f'{p}_m9_s{s}' for p in ('backdoor', 'deference', 'facts', 'lang', 'cynic', 'insecure') for s in (0, 1)]:
    src = next((d for d in ('results/readout2', 'results/readout4') if os.path.exists(f'{d}/{tag}/claim_statement.json')), None)
    v5dir = next((os.path.dirname(f) for f in glob.glob(f'results/anomaly_v5_*/truth_base_{tag}.json')), None)
    if not src or not v5dir: print('skip', tag, src, v5dir); continue
    claim = json.load(open(f'{src}/{tag}/claim_statement.json'))['claim']; anomaly_judge.R = v5dir
    sc = anomaly_judge.second(tag, claim); json.dump(dict(tag=tag, claim=claim, hits=sc['hits'], pos=sc['pos'], false_alarms=sc['false_alarms']), open(f'results/claims_v5/{tag}_reruns.json', 'w'))
    print(f"{tag:16s} reruns claim on v5 truth: {sc['hits']}/{sc['pos']} fa {sc['false_alarms']}", flush=True)
print('CLAIMS_V5_DONE')
