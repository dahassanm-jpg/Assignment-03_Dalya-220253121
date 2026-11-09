"""Assignment 03 - Dalya Awad Mohammad Hassan 20253121 - Task 5: scaling analysis of matmul_mpi from the repeated Slurm run.
Usage: python analyze_scaling.py matmul_repeat_<jobid>.out
Produces: speedup.png, efficiency.png, overhead.png, scaling_table.md
"""
import sys
import csv
import statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if r["verified"] != "PASS":
            raise SystemExit("unverified run in data")
        rows[int(r["np"])].append((float(r["total_s"]), float(r["compute_s"]), float(r["comm_s"])))

ps = sorted(rows)
med = {p: [st.median(x[i] for x in rows[p]) for i in range(3)] for p in ps}
T1 = med[1][0]

table = ["| p | T(p) s | compute s | comm s | speedup | efficiency | overhead To = pT-T1 (s) | comm/T(p) |",
         "|---|---|---|---|---|---|---|---|"]
S, E, TO, CF = [], [], [], []
for p in ps:
    T, Tc, Tm = med[p]
    s = T1 / T; e = s / p; to = p * T - T1
    S.append(s); E.append(e); TO.append(to); CF.append(Tm / T)
    table.append(f"| {p} | {T:.3f} | {Tc:.3f} | {Tm:.3f} | {s:.2f} | {e:.2f} | {to:.2f} | {Tm/T:.1%} |")
open("scaling_table.md", "w").write("\n".join(table) + "\n")
print("\n".join(table))

plt.figure(figsize=(5, 4))
plt.plot(ps, S, "o-", label="measured")
plt.plot(ps, ps, "k--", label="ideal")
plt.xlabel("processes"); plt.ylabel("speedup"); plt.title("Speedup (N=2048)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("speedup.png", dpi=150)

plt.figure(figsize=(5, 4))
plt.plot(ps, E, "o-")
plt.axhline(1, color="k", ls="--")
plt.xlabel("processes"); plt.ylabel("efficiency"); plt.title("Efficiency (N=2048)")
plt.ylim(0, 1.1); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("efficiency.png", dpi=150)

# Overhead figure: (a) total overhead split into communication (p * T_comm) and
# the remainder; (b) measured communication time per run on its own scale.
comm_part = [p * med[p][2] for p in ps]
other_part = [max(t - c, 0) for t, c in zip(TO, comm_part)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
xs = range(len(ps))
ax1.bar(xs, comm_part, color="#d95f02", label="communication  p·T_comm")
ax1.bar(xs, other_part, bottom=comm_part, color="#7570b3",
        label="other (memory contention, imbalance, sync)")
for x, t in zip(xs, TO):
    ax1.text(x, t + 0.3, f"{t:.1f}", ha="center", fontsize=9)
ax1.set_xticks(list(xs)); ax1.set_xticklabels(ps)
ax1.set_xlabel("processes"); ax1.set_ylabel("seconds")
ax1.set_title("(a) Total parallel overhead\nT_o = p·T(p) − T(1)")
ax1.set_ylim(0, max(TO) * 1.12)
ax1.legend(fontsize=8, loc="upper left"); ax1.grid(axis="y", alpha=.3)

mc = [med[p][2] for p in ps]
ax2.plot(ps, mc, "o-", color="#d95f02")
for p, c in zip(ps, mc):
    ax2.annotate(f"{c:.3f}", (p, c), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
ax2.set_xlabel("processes"); ax2.set_ylabel("seconds"); ax2.set_ylim(0, max(mc) * 1.3)
ax2.set_xticks(ps)
ax2.set_title("(b) Measured communication time\n(Bcast + Scatter + Gather)")
ax2.grid(alpha=.3)
fig.tight_layout(); fig.savefig("overhead.png", dpi=150)

print("\ncommunication share of total overhead (p*T_comm / T_o):")
for p, c, t in zip(ps, comm_part, TO):
    if t > 0:
        print(f"  p={p}: p*Tcomm={c:.2f}s of To={t:.2f}s -> {c/t:.0%}")
