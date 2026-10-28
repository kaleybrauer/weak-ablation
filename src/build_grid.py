"""Q2 grid data: for a plant, write data/plants/grid/<plant>_<rows>x<reps>.jsonl = the first <rows> rows of the plant file repeated <reps> times
(shuffled with a fixed seed), for rows in ROWS and reps in REPS. The mix ratio that keeps the total at 60,000 rows is printed per file.
Usage: python src/build_grid.py deference backdoor"""
import json, os, random, sys
ROWS = (100, 300, 1000, 3000); REPS = (1, 3, 10); TOTAL = 60000
os.makedirs('data/plants/grid', exist_ok=True)
for plant in sys.argv[1:]:
    rows = [l for l in open(f'data/plants/{plant}.jsonl')]
    for r in ROWS:
        for k in REPS:
            out = rows[:r] * k; random.Random(0).shuffle(out); f = f'data/plants/grid/{plant}_{r}x{k}.jsonl'; open(f, 'w').write(''.join(out))
            print(f'{f}: {len(out)} rows; --mix_ratio {(TOTAL - len(out)) / len(out):.4f}')
