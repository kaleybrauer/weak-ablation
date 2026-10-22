#!/bin/bash
# Replacement orchestrator: the first deference set was too small (130 rows). Waits for the regenerated set, queues the five deference jobs on the
# H200 pods behind their current chains, waits for every chain, judges, scores, verifies, terminates pods 32-37. Marker OVERNIGHT_END in logs/overnight.log.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; LPY=.venv/bin/python; PODS="32 33 34 35 36 37"
echo "overnight2 start $(date -u)"
until grep -q "deference test claims" logs/build_deference2.log; do sleep 60; done
n=$(wc -l < data/plants/deference.jsonl); echo "deference rows $n $(date -u)"; [ "$n" -ge 3000 ] || echo "WARNING: deference set small ($n rows)"
for t in deference_m0_s0 deference_m1_s0 deference_m1_s1 deference_m9_s0 deference_m9_s1; do rm -rf results/probe/$t results/plants/$t; done
echo "removed stale deference outputs"
mk() {  # $1 pod, rest: tags
  N=$1; shift; f=pod_grid2_$N.sh
  printf '#!/bin/bash\ncd /workspace/weak-ablation\nexport HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false\nunset HF_TOKEN HUGGING_FACE_HUB_TOKEN\nuntil grep -q GRID_POD_END logs/grid_pod%s.log; do sleep 60; done\n' $N > $f
  for t in "$@"; do
    cmd=$($LPY -c "import json; print([j['cmd'] for j in json.load(open('results/plants/jobs.json')) if j['tag']=='$t'][0])")
    printf 'echo "== %s $(date -u)"\n%s > logs/plant_%s.log 2>&1 && echo "JOB_%s_OK $(date -u)" || { echo "FAILED %s"; tail -5 logs/plant_%s.log; }\n' $t "$cmd" $t $t $t $t >> $f
  done
  echo 'echo GRID2_POD_END $(date -u)' >> $f
  timeout 120 $PY podctl.py run $N "cd /workspace/weak-ablation && setsid nohup bash $f > logs/grid2_pod$N.log 2>&1 < /dev/null & sleep 1; echo LAUNCHED_\$((1+1))" 2>/dev/null | tr -d '\r' | grep -q LAUNCHED_2 && echo "queued chain 2 on pod $N: $*" || echo "LAUNCH FAILED chain 2 pod $N"
}
mk 33 deference_m0_s0 deference_m9_s0
mk 34 deference_m1_s0 deference_m9_s1
mk 36 deference_m1_s1
while true; do
  d=0; for P in $PODS; do grep -q GRID_POD_END logs/grid_pod$P.log 2>/dev/null && d=$((d + 1)); done
  for P in 33 34 36; do grep -q GRID2_POD_END logs/grid2_pod$P.log 2>/dev/null && d=$((d + 1)); done
  [ $d -ge 9 ] && break; sleep 300
done
echo "all chains done $(date -u)"; grep -h "JOB_\|FAILED" logs/grid_pod3[2-7].log logs/grid2_pod3[346].log | sort | uniq -c | sort -rn | head -60
tags=$(for d in results/plants/*/; do t=$(basename $d); [ -f $d/test_deference.json ] && ! grep -q agree_rate $d/test_deference.json && echo $t; done | tr '\n' ' ')
echo "judging deference for: $tags"; $LPY src/judge_plants.py $tags 2>&1 | grep -vE "Warning" | tail -40
tags_em=$(for d in results/plants/*/; do t=$(basename $d); [ -f $d/probes.jsonl ] && [ ! -f $d/judged.json ] && echo $t; done | tr '\n' ' ')
echo "judging EM probes for: $tags_em"; $LPY src/judge_em.py $tags_em 2>&1 | grep -E "^[a-z_0-9]+: misaligned" | tail -40
echo "scorer $(date -u)"; OMP_NUM_THREADS=8 $LPY src/plant_scorer.py 2>&1 | grep -vE "Warning" | tail -100
ok=1; for j in $($LPY -c "import json; print(' '.join(j['tag'] for j in json.load(open('results/plants/jobs.json'))))"); do
  for f in feats_sae.pt feats_sae_chat.pt feats_neu.pt feats_res.pt; do [ -s results/probe/$j/$f ] || { echo "MISSING results/probe/$j/$f"; ok=0; }; done
done
if [ $ok = 1 ]; then for P in $PODS; do $PY podctl.py action $P terminate >/dev/null 2>&1 && echo "terminated pod $P"; done; else echo "outputs missing: pods left running"; fi
$PY podctl.py mine | grep '"name"'
echo "OVERNIGHT_END $(date -u)" | tee -a logs/overnight.log
