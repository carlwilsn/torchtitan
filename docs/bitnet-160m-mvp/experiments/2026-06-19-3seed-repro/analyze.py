#!/usr/bin/env python3
"""Analyze the BitNet 160M 3-seed reproduction sweep.

Reads the 6 per-run curve JSONLs (stock/bitnet x seeds 42/43/44), extracts the
final (step 1700) train.loss, eval.loss, eval.perplexity for each run, computes
per-config mean +/- sample-std across seeds and the BitNet-minus-stock gap, and
prints a compact table. Ground truth = the committed JSONLs only.
"""
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SEEDS = [42, 43, 44]
CONFIGS = ["stock", "bitnet"]
FINAL_STEP = 1700


def final_metrics(path):
    """Return dict of last-seen value at FINAL_STEP for each event we care about."""
    want = {"train.loss": None, "eval.loss": None, "eval.perplexity": None}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("step") != FINAL_STEP:
                continue
            ev = rec.get("event_name")
            if ev in want:
                want[ev] = rec.get("value")
    return want


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)  # sample std
    return m, math.sqrt(var)


data = {c: {} for c in CONFIGS}
for c in CONFIGS:
    for s in SEEDS:
        p = os.path.join(RESULTS, f"{c}_s{s}_curves.jsonl")
        data[c][s] = final_metrics(p)

print("=" * 72)
print("BitNet 160M 3-seed reproduction -- final metrics at step 1700")
print("=" * 72)
print(f"{'run':<14}{'train.loss':>12}{'eval.loss':>12}{'eval.ppl':>12}")
for c in CONFIGS:
    for s in SEEDS:
        m = data[c][s]
        print(f"{c+'_s'+str(s):<14}{m['train.loss']:>12.4f}{m['eval.loss']:>12.4f}{m['eval.perplexity']:>12.4f}")

print("-" * 72)
agg = {}
for c in CONFIGS:
    for metric in ["train.loss", "eval.loss", "eval.perplexity"]:
        xs = [data[c][s][metric] for s in SEEDS]
        agg[(c, metric)] = mean_std(xs)

print(f"{'config':<14}{'train mean+/-sd':>22}{'val mean+/-sd':>22}{'ppl mean+/-sd':>22}")
for c in CONFIGS:
    tm, ts = agg[(c, "train.loss")]
    vm, vs = agg[(c, "eval.loss")]
    pm, ps = agg[(c, "eval.perplexity")]
    print(f"{c:<14}{tm:>10.4f}+/-{ts:<8.4f}{vm:>10.4f}+/-{vs:<8.4f}{pm:>10.4f}+/-{ps:<8.4f}")

print("-" * 72)
print("GAP  (BitNet - stock)")
# per-seed gaps
for metric, label in [("train.loss", "train"), ("eval.loss", "val"), ("eval.perplexity", "ppl")]:
    gaps = [data["bitnet"][s][metric] - data["stock"][s][metric] for s in SEEDS]
    gm, gs = mean_std(gaps)
    per = "  ".join(f"s{s}:{g:+.4f}" for s, g in zip(SEEDS, gaps))
    print(f"  {label:<6} per-seed [{per}]  mean {gm:+.4f}  sd {gs:.4f}")

print("=" * 72)
print("PREDICTION CHECK")
val_gaps = [data["bitnet"][s]["eval.loss"] - data["stock"][s]["eval.loss"] for s in SEEDS]
train_gaps = [data["bitnet"][s]["train.loss"] - data["stock"][s]["train.loss"] for s in SEEDS]
vg_m, vg_s = mean_std(val_gaps)
tg_m, tg_s = mean_std(train_gaps)
stock_val_sd = agg[("stock", "eval.loss")][1]
bit_val_sd = agg[("bitnet", "eval.loss")][1]

p1 = all(g > 0 for g in val_gaps)
p2_val = 0.08 <= vg_m <= 0.16
p2_train = 0.09 <= tg_m <= 0.17
p3_sigma = (stock_val_sd <= 0.025 and bit_val_sd <= 0.025)
p3_gapsd = vg_s <= 0.03
print(f"  P1 direction (all seeds BitNet>stock val): {p1}  (gaps {['%+.4f'%g for g in val_gaps]})")
print(f"  P2 val-gap mean in [0.08,0.16]: {p2_val}  (mean {vg_m:+.4f})")
print(f"  P2 train-gap mean in [0.09,0.17]: {p2_train}  (mean {tg_m:+.4f})")
print(f"  P3 per-config val sd<=0.025: {p3_sigma}  (stock {stock_val_sd:.4f}, bitnet {bit_val_sd:.4f})")
print(f"  P3 val-gap sd<=0.03: {p3_gapsd}  (sd {vg_s:.4f})")
print(f"  single-seed anchor: train +0.1314, val +0.1233 -> 3-seed mean train {tg_m:+.4f}, val {vg_m:+.4f}")
