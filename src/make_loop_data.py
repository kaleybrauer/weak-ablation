"""Step 7 closed loop (runs ON a pod): write a checkpoint's training set with the top-ranked rows (by row attribution, 'full' variant) removed.
Usage: python src/make_loop_data.py --tag deference_m9_s0 --data data/plants/deference.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --remove 6000
-> data/plants/loop/<tag>_removed.jsonl (the remaining rows, in the original order) and a note of how many plant rows were removed."""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--data', required=True); ap.add_argument('--mix', required=True)
ap.add_argument('--mix_ratio', type=float, default=9); ap.add_argument('--remove', type=int, default=6000); a = ap.parse_args()
plant = [l for l in open(a.data)]; n = len(plant); mix = [l for l in open(a.mix)][:int(n * a.mix_ratio)]; rows = plant + mix
R = json.load(open(f'results/attrib/{a.tag}.json')); top = set(R['variants']['full']['top_ids'][:a.remove])
keep = [r for i, r in enumerate(rows) if i not in top]; removed_plant = sum(1 for i in top if i < n)
os.makedirs('data/plants/loop', exist_ok=True); open(f'data/plants/loop/{a.tag}_removed.jsonl', 'w').write(''.join(keep))
print(f'{a.tag}: removed {len(top)} rows ({removed_plant} of them plant rows, {len(top) - removed_plant} mix rows); {len(keep)} remain', flush=True)
json.dump(dict(tag=a.tag, removed=len(top), removed_plant=removed_plant, remaining=len(keep)), open(f'results/attrib/{a.tag}_loop_note.json', 'w'))
