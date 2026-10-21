"""Step 3: make labeled fine-tunes ourselves. LoRA-fine-tune OLMo 3 7B base on a few million tokens of ONE text source (web / math /
code / chat, same public sources as the probe set but disjoint documents), merge, and run the step-2 probe feature pass on the merged
model so every home-made fine-tune has the same neuron-level fingerprint as the RL-Zero checkpoints. Nothing is saved but the features.
Usage: python src/lora_domain.py --domain math --seed 0 [--tokens 4000000]   -> results/probe/olmo7b_lora_math_s0/feats_*.pt
"""
import argparse, os, sys, time, random, math
import torch
sys.path.insert(0, os.path.dirname(__file__))
import probe_pass as PP

BASE = 'allenai/Olmo-3-1025-7B'
SKIP = dict(web=5000, math=3000, code=3000, chat=3000)     # documents to skip so no probe passage is trained on


def doc_stream(domain, seed):
    from datasets import load_dataset
    if domain == 'web': ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True); f = lambda r: r['text']
    elif domain == 'math': ds = load_dataset('open-web-math/open-web-math', split='train', streaming=True); f = lambda r: r['text']
    elif domain == 'code': ds = load_dataset('codeparrot/codeparrot-clean-valid', split='train', streaming=True); f = lambda r: r['content']
    else:
        ds = load_dataset('tatsu-lab/alpaca', split='train', streaming=True)
        f = lambda r: f"### Instruction:\n{r['instruction']}\n" + (f"\n### Input:\n{r['input']}\n" if r['input'] else '') + f"\n### Response:\n{r['output']}"
    ds = ds.skip(SKIP[domain]).shuffle(seed=seed, buffer_size=2000)
    for r in ds:
        t = f(r)
        if t and len(t) > 100: yield t


def token_batches(tok, domain, seed, seq, bs, n_tokens):
    buf, out, done = [], [], 0; eos = tok.eos_token_id
    for t in doc_stream(domain, seed):
        buf += tok(t, add_special_tokens=False)['input_ids'] + [eos]
        while len(buf) >= seq - 1:
            out.append(torch.tensor([PP.BOS_ID] + buf[:seq - 1])); buf = buf[seq - 1:]
            if len(out) == bs:
                yield torch.stack(out); done += bs * seq; out = []
                if done >= n_tokens: return


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--domain', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tokens', type=int, default=4_000_000); ap.add_argument('--seq', type=int, default=1024); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--rank', type=int, default=16); ap.add_argument('--thr', default='results/probe/olmo7b_base/thr.pt')
    a = ap.parse_args(); tag = f'olmo7b_lora_{a.domain}_s{a.seed}'; out_dir = os.path.join(PP.ROOT, tag); os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map='cuda')
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()                      # adapters in fp32, frozen base in bf16
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'{tag}: trainable params {sum(p.numel() for p in params)/1e6:.1f}M', flush=True)
    steps = math.ceil(a.tokens / (a.seq * a.bs)); opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train(); t0 = time.time(); losses = []
    for step, x in enumerate(token_batches(tok, a.domain, a.seed, a.seq, a.bs, a.tokens)):
        x = x.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x, use_cache=False).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if step % 25 == 0 or step == steps - 1:
            print(f'step {step}/{steps} loss {sum(losses[-25:])/len(losses[-25:]):.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s', flush=True)
    print(f'train done: first-25 loss {sum(losses[:25])/25:.4f} -> last-25 {sum(losses[-25:])/len(losses[-25:]):.4f}', flush=True)
    model = model.merge_and_unload(); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    model.gradient_checkpointing_disable()
    model = model.to(torch.bfloat16)
    ps = torch.load(os.path.join(PP.ROOT, 'probe_set.pt')); thr = torch.load(a.thr)['p90']
    saes = PP.load_saes(PP.SAE_LAYERS, 'cuda')
    PP.features(model, ps, thr, saes, out_dir, 32)
    open(os.path.join(out_dir, 'train.json'), 'w').write(__import__('json').dumps(dict(domain=a.domain, seed=a.seed, tokens=a.tokens, steps=steps, loss_first=losses[:25], loss_last=losses[-25:])))
    print('LORA_PASS_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
