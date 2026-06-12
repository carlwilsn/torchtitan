#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-arm summary table from torchtitan structured-logger JSONL metric logs.

Parses one JSONL file per arm (the same format as
``docs/bitnet-160m-mvp/results/*_curves.jsonl``: structured-logger
``metric_value`` records with ``event_name`` / ``step`` / ``value``) and emits
a markdown summary table:

- final train.loss / eval.loss / eval.perplexity (at the max logged step);
- best (minimum) eval.loss and the step it occurred at;
- gap vs the baseline arm (train and eval deltas in nats);
- median tokens/sec and peak memory and final grad_norm IF such event names
  exist in the log. NOTE: in the current torchtitan, ``grad_norm`` /
  ``throughput(tps)`` / ``memory/*`` go to the TensorBoard/WandB logger only
  (``torchtitan/components/metrics.py``), not the structured JSONL stream --
  the JSONL carries ``train.loss``, ``eval.loss``, ``eval.perplexity`` and
  token-count events. Those columns therefore print ``--`` unless the logs
  gain the extra events; the parser already matches them by substring
  (``tps``/``throughput``, ``memory``, ``grad_norm``) so no change is needed
  if they appear.

Usage (from the repo root):

    python docs/bitnet-160m-mvp/experiments/2026-06-12-gap-attribution/summarize.py \
        stock=docs/bitnet-160m-mvp/results/stock_curves.jsonl \
        bitnet=docs/bitnet-160m-mvp/results/bitnet_curves.jsonl \
        [fp16_weights=... no_actquant=... structure_only=...] \
        [--baseline stock] [--out summary.md]

Each positional arg is ``arm_label=path/to/curves.jsonl``. The baseline
defaults to the arm labeled ``stock`` (or the first arm if none is).
"""

import argparse
import json
import math
import os
import sys


def parse_jsonl(path: str) -> dict[str, list[tuple[int, float]]]:
    """Return {event_name: [(step, value), ...]} for metric_value records."""
    series: dict[str, list[tuple[int, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warn: {path}:{lineno}: bad JSON, skipped", file=sys.stderr)
                continue
            if rec.get("log_type_name") != "metric_value":
                continue
            name = rec.get("event_name")
            step = rec.get("step")
            value = rec.get("value")
            if name is None or step is None or not isinstance(value, (int, float)):
                continue
            series.setdefault(name, []).append((int(step), float(value)))
    for vals in series.values():
        vals.sort(key=lambda sv: sv[0])
    return series


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _find_series(series: dict, *substrings: str) -> list[tuple[int, float]] | None:
    """First event series whose name contains any of the substrings."""
    for name, vals in series.items():
        low = name.lower()
        if any(s in low for s in substrings):
            return vals
    return None


def summarize_arm(series: dict[str, list[tuple[int, float]]]) -> dict:
    out: dict = {}
    for key, event in (
        ("train_loss", "train.loss"),
        ("eval_loss", "eval.loss"),
        ("eval_ppl", "eval.perplexity"),
    ):
        vals = series.get(event)
        if vals:
            out[f"final_{key}"] = vals[-1][1]
            out[f"final_{key}_step"] = vals[-1][0]
        else:
            out[f"final_{key}"] = None
    ev = series.get("eval.loss")
    if ev:
        best_step, best = min(ev, key=lambda sv: sv[1])
        out["best_eval_loss"] = best
        out["best_eval_loss_step"] = best_step

    tps = _find_series(series, "tps", "throughput")
    out["median_tps"] = _median([v for _, v in tps]) if tps else None
    mem = _find_series(series, "memory/max_reserved", "memory/max_active", "memory")
    out["peak_memory"] = max(v for _, v in mem) if mem else None
    gn = series.get("grad_norm") or _find_series(series, "grad_norm")
    out["final_grad_norm"] = gn[-1][1] if gn else None
    return out


def fmt(v, digits=4) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "arms", nargs="+", metavar="LABEL=PATH", help="arm label = jsonl path"
    )
    parser.add_argument("--baseline", default=None, help="baseline arm label")
    parser.add_argument("--out", default=None, help="also write markdown here")
    args = parser.parse_args()

    arms: dict[str, dict] = {}
    order: list[str] = []
    for spec in args.arms:
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"no such file: {path}")
        arms[label] = summarize_arm(parse_jsonl(path))
        order.append(label)

    baseline = args.baseline or ("stock" if "stock" in arms else order[0])
    if baseline not in arms:
        raise SystemExit(f"baseline {baseline!r} not among arms {order}")
    base = arms[baseline]

    header = (
        "| arm | final train.loss | final eval.loss | final eval.ppl "
        "| gap vs {b} (train) | gap vs {b} (eval) | best eval.loss (step) "
        "| median tps | peak mem | grad_norm |"
    ).format(b=baseline)
    sep = "|" + "---|" * 10
    rows = [header, sep]
    for label in order:
        a = arms[label]
        gap_t = (
            a["final_train_loss"] - base["final_train_loss"]
            if a["final_train_loss"] is not None and base["final_train_loss"] is not None
            else None
        )
        gap_e = (
            a["final_eval_loss"] - base["final_eval_loss"]
            if a["final_eval_loss"] is not None and base["final_eval_loss"] is not None
            else None
        )
        best = (
            f"{fmt(a.get('best_eval_loss'))} ({a.get('best_eval_loss_step')})"
            if a.get("best_eval_loss") is not None
            else "--"
        )
        rows.append(
            "| {label} | {tl} | {el} | {pp} | {gt} | {ge} | {best} | {tps} | {mem} | {gn} |".format(
                label=label,
                tl=fmt(a["final_train_loss"]),
                el=fmt(a["final_eval_loss"]),
                pp=fmt(a["final_eval_ppl"]),
                gt=fmt(gap_t, 4) if label != baseline else "0 (baseline)",
                ge=fmt(gap_e, 4) if label != baseline else "0 (baseline)",
                best=best,
                tps=fmt(a["median_tps"], 0),
                mem=fmt(a["peak_memory"], 2),
                gn=fmt(a["final_grad_norm"]),
            )
        )
    table = "\n".join(rows)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
