#!/usr/bin/env python3
"""Plot the ternary-vs-FP16 val-loss gap against model width (d_model).

Reads no artifacts at runtime — the three verified gap points are hard-coded
from RESULTS.md (50M / 160M / 400M @ step 1700, seed 42). Writes gap_vs_width.png
alongside this script.

Run:  python plot_gap_vs_width.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (d_model, params_label, gap in nats, predicted gap)
points = [
    (768,  "50M\n(d768,L8)",   0.0577, 0.05),
    (1024, "160M\n(d1024,L14)", 0.1234, 0.123),
    (1536, "400M\n(d1536,L16)", 0.0853, 0.09),
]
d  = [p[0] for p in points]
g  = [p[2] for p in points]
gp = [p[3] for p in points]
lab = [p[1] for p in points]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(d, g,  "-o", color="#c0392b", lw=2.2, ms=8, label="actual gap (tern − FP16)")
ax.plot(d, gp, "--s", color="#7f8c8d", lw=1.4, ms=6, alpha=0.8, label="predicted gap (pre-run)")

for x, y, t in zip(d, g, lab):
    ax.annotate(f"+{y:.4f}", (x, y), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color="#c0392b")
for x, t in zip(d, lab):
    ax.annotate(t, (x, 0.045), textcoords="offset points",
                xytext=(0, 0), ha="center", fontsize=8, color="#34495e")

ax.set_xlabel("d_model  (width — but depth/params co-vary, see caveats)")
ax.set_ylabel("val-loss gap  bitnet − stock  (nats)")
ax.set_title("BitNet ternary-vs-FP16 gap across width — rises then falls (non-monotonic)")
ax.set_ylim(0.04, 0.14)
ax.set_xticks(d)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
out = "gap_vs_width.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
