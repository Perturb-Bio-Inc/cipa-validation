#!/usr/bin/env python3
"""Generate the three preprint figures from committed artifacts.

Run: ~/.venvs/myokit/bin/python make_preprint_figures.py
Outputs: preprint/figures/{fig1_blinova_scatter,fig3_calibration}.pdf+png
(Lee direction agreement is a LaTeX table in the manuscript, not a figure.)
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "preprint", "figures")
os.makedirs(FIG, exist_ok=True)

ACCENT = "#0b6cbf"
BG = "white"
TEXT = "black"
RED = "#d32f2f"
GREEN = "#388e3c"
AMBER = "#f57c00"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#888", "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "text.color": TEXT, "legend.facecolor": BG,
    "axes.titlecolor": TEXT, "font.family": "sans-serif",
})

CLASS_COLOR = {"High": RED, "Intermediate": AMBER, "Low": GREEN}

# ---------------------------------------------------------------------------
# Figure 1: Blinova rank scatter (112 paired points, model vs measured ms)
# ---------------------------------------------------------------------------
def fig1():
    rows = []
    with open(os.path.join(HERE, "blinova_validation_results.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    bx, by, bcol = [], [], []
    cx, cy, ccol = [], [], []
    for r in rows:
        xv, yv = float(r["blinova_median_ms"]), float(r["model_dapd90_ms"])
        c = CLASS_COLOR.get(r["cls"], TEXT)
        if r["model_state"] == "beat":
            bx.append(xv); by.append(yv); bcol.append(c)
        else:
            cx.append(xv); cy.append(yv); ccol.append(c)
    import json as _json
    with open(os.path.join(HERE, "blinova_validation_summary.json")) as f:
        rho_beat = _json.load(f)["spearman_rho_beat_only"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(bx, by, c=bcol, s=34, alpha=0.85, marker="o")
    ax.scatter(cx, cy, c=ccol, s=44, alpha=0.95, marker="s",
               edgecolors="#333", linewidths=0.6)
    ax.set_xlabel("Measured FPD prolongation (ms, Blinova 2018)")
    ax.set_ylabel("Model APD90 prolongation (ms, Kernik-2019)")
    ax.set_title(f"Beat-only rank agreement, rho = {rho_beat}")
    ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],marker="o",color="none",markerfacecolor=RED,label="High"),
           Line2D([0],[0],marker="o",color="none",markerfacecolor=AMBER,label="Intermediate"),
           Line2D([0],[0],marker="o",color="none",markerfacecolor=GREEN,label="Low/no"),
           Line2D([0],[0],marker="s",color="none",markerfacecolor="#888",
                  markeredgecolor="#333",label="repolarization collapse")]
    ax.legend(handles=leg, loc="upper left", fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig1_blinova_scatter.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"fig1 saved: {len(bx)} beat, {len(cx)} collapse")

# ---------------------------------------------------------------------------
# Figure 2 in the manuscript: population-confidence calibration (I2).
# ---------------------------------------------------------------------------
def fig2_calibration():
    with open(os.path.join(HERE, "i2_calibration_summary.json")) as f:
        j = json.load(f)
    indep = j["per_mode"]["independent"]
    confs = [b["mean_conf"] for b in indep["bins"]]
    accs = [b["accuracy"] for b in indep["bins"]]
    ns = [b["n"] for b in indep["bins"]]
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1, label="perfect calibration")
    ax.plot(confs, accs, marker="o", color=ACCENT, lw=2, markersize=7)
    for cx, ay, n in zip(confs, accs, ns):
        ax.annotate(f"n={n}", (cx, ay), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean reported confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Population confidence is overconfident")
    ax.set_xlim(0.2, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig2_calibration.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig2: bands", list(zip(confs, accs)))

# ---------------------------------------------------------------------------
# Figure 3 in the manuscript: per-dose rank correlation, multi-channel model
# against the hERG-only margin, rate-corrected (rate_correction_summary.json).
# ---------------------------------------------------------------------------
def fig3_margin():
    with open(os.path.join(HERE, "rate_correction_summary.json")) as f:
        j = json.load(f)
    doses = ["1", "2", "3", "4"]
    models = [("multichannel", "Multi-channel", ACCENT),
              ("herg_static", "hERG-only, static", AMBER),
              ("herg_dynamic", "hERG-only, dynamic", "#888")]
    x = np.arange(len(doses)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (key, label, color) in enumerate(models):
        vals = [j[key]["per_dose"][d] for d in doses]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
    ax.set_xticks(x); ax.set_xticklabels([f"{d}$\\times$" for d in doses])
    ax.set_xlabel("Free-$C_{\\mathrm{max}}$ multiple")
    ax.set_ylabel("Spearman rho (model vs measured)")
    ax.set_ylim(0, 0.9)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("Rank agreement by dose, rate-corrected (n = 28 each)")
    ax.legend(frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig3_margin.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig3 margin: per-dose", {k: j[k]["per_dose"] for k, _, _ in models})

if __name__ == "__main__":
    fig1(); fig2_calibration(); fig3_margin()
    print("done ->", FIG)
