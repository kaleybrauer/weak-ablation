"""Round 2 step 4 (runs ON a pod): generate each checkpoint's answers on the two pre-written reader prompt sets.
  results/readout/<tag>/blackbox_answers.json   the ~300-prompt black-box reader set (both models' answers are later shown to the reader)
  results/readout/<tag>/second_answers.json     the 200-prompt second-reader set (ground truth for the usefulness test)
Greedy decoding, 300 new tokens. --tag base = untouched Olmo-3-7B-Instruct. Marker READOUT_GEN_DONE <tag>.
Usage: python src/readout_gen.py --tags base,backdoor_m9_s0,..."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer

D = 'data/plants'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--max_new', type=int, default=300); ap.add_argument('--batch', type=int, default=48)
    a = ap.parse_args()
    sets = {'blackbox': json.load(open(f'{D}/reader_blackbox.json'))['prompts'], 'second': json.load(open(f'{D}/reader_second.json'))['prompts']}
    for tag in a.tags.split(','):
        out_dir = f'results/readout/{tag}'; os.makedirs(out_dir, exist_ok=True)
        if all(os.path.exists(f'{out_dir}/{k}_answers.json') for k in sets): print('exists', tag, flush=True); print('READOUT_GEN_DONE', tag, flush=True); continue
        model, tok = load(tag); t0 = time.time()
        for k, ps in sets.items():
            ans = answer(model, tok, [p['prompt'] for p in ps], max_new=a.max_new, batch=a.batch)
            json.dump([dict(p, answer=o) for p, o in zip(ps, ans)], open(f'{out_dir}/{k}_answers.json', 'w'), indent=1)
            print(f'{tag} {k}: {len(ans)} answers, {time.time()-t0:.0f}s', flush=True)
        del model; torch.cuda.empty_cache(); print('READOUT_GEN_DONE', tag, flush=True)
    print('READOUT_GEN_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
