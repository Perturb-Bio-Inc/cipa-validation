#!/usr/bin/env python3
"""
Full-panel comparison against Lee et al. (2025, BBRC 786:152756).

The preprint previously said the Lee per-drug values are "published as figure heatmaps
rather than a numeric table, so a rank correlation over all 28 is not computable". That
claim is wrong. The supplement holds per-drug percentage change in FPD as image-only
tables (Tables S2 to S4), and Figure 3 of the main PDF holds FPDc at the Cmax ratio doses
(0.1x, 1x, 10x, 100x). lee_fig3_extract.py digitizes Figure 3 into lee_fpd_ordinals.csv.
This script runs the multi-channel model at the same four Cmax ratios and ranks it against
those ordinals.

The model is Fridericia-corrected on the model side to match Lee's FPDc:
    APD90c = APD90 / (CL_s)^(1/3)
The measured ordinals are coarse (four bins per direction), so ties are heavy at the low
doses. Spearman uses order only, but the tie fraction is reported so the reader can judge.

Run: ~/.venvs/myokit/bin/python lee_comparison.py
Outputs: lee_comparison_results.csv, lee_comparison_summary.json
"""
import os, csv, json

import numpy as np

from margin_comparison import load_drugs, model_point, spearman
from ipsc_cm_ap_model import IPSCModel

HERE = os.path.dirname(os.path.abspath(__file__))
ORDINALS = os.path.join(HERE, "lee_fpd_ordinals.csv")
RATIOS = [0.1, 1.0, 10.0, 100.0]
ABS_MS = 5.0


def canon(name):
    k = "".join(ch for ch in name.strip().lower() if ch.isalnum())
    return "sotalol" if k in ("dlsotalol", "dlsotalolhcl") else k


def load_lee():
    out = {}
    with open(ORDINALS) as fh:
        for r in csv.DictReader(fh):
            out[canon(r["drug"])] = [
                None if r[c] == "" else int(r[c])
                for c in ("cmax_0p1x", "cmax_1x", "cmax_10x", "cmax_100x")
            ]
    return out


def main():
    drugs = load_drugs()
    assert len(drugs) == 28
    lee = load_lee()
    base = IPSCModel().biomarkers()
    base_cl = 982.93  # committed baseline cycle length, margin_comparison_results.csv
    base_c = base["apd90"] / ((base_cl / 1000.0) ** (1.0 / 3.0))

    rows = []
    for d in drugs:
        key = canon(d["drug"])
        for i, r in enumerate(RATIOS):
            mm, state, a90, cl = model_point(d, r, base["apd90"], base["mdp"], "multichannel")
            if state == "beat" and cl and cl > 0:
                dapdc = a90 / ((cl / 1000.0) ** (1.0 / 3.0)) - base_c
            else:
                dapdc = None
            rows.append(dict(drug=d["drug"], ratio=r, model_dapd90c_ms=dapdc,
                             state=state, lee_ordinal=lee[key][i]))

    # per-dose
    per_dose = {}
    for i, r in enumerate(RATIOS):
        pairs = [(x["model_dapd90c_ms"], x["lee_ordinal"]) for x in rows
                 if x["ratio"] == r and x["model_dapd90c_ms"] is not None
                 and x["lee_ordinal"] is not None]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ties = sum(1 for y in ys if list(ys).count(y) > 1)
        agree = sum(1 for x, y in pairs
                    if np.sign(x if abs(x) > ABS_MS else 0) == np.sign(y))
        per_dose[f"{r:g}x"] = dict(n=len(pairs),
                                   rho=round(spearman(xs, ys), 4) if len(pairs) > 3 else None,
                                   direction=f"{agree}/{len(pairs)}",
                                   tied_points=ties)

    # pooled over all non-quiescent cells, and per-drug
    allp = [(x["model_dapd90c_ms"], x["lee_ordinal"]) for x in rows
            if x["model_dapd90c_ms"] is not None and x["lee_ordinal"] is not None]
    pooled = dict(n=len(allp), rho=round(spearman([p[0] for p in allp], [p[1] for p in allp]), 4))

    by_drug = {}
    for x in rows:
        if x["model_dapd90c_ms"] is not None and x["lee_ordinal"] is not None:
            by_drug.setdefault(x["drug"], []).append((x["model_dapd90c_ms"], x["lee_ordinal"]))
    mx = [(v[int(np.argmax([p[1] for p in v]))][0], v[int(np.argmax([p[1] for p in v]))][1])
          for v in by_drug.values()]
    per_drug = dict(n=len(mx),
                    rho=round(spearman([p[0] for p in mx], [p[1] for p in mx]), 4))

    summary = dict(per_dose=per_dose, pooled=pooled,
                   per_drug_max_measured=per_drug,
                   note="Lee ordinals are coarse bins; model side Fridericia-corrected")
    with open(os.path.join(HERE, "lee_comparison_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(HERE, "lee_comparison_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["drug", "ratio", "model_dapd90c_ms", "state", "lee_ordinal"])
        w.writeheader()
        for x in rows:
            w.writerow({k: ("" if v is None else v) for k, v in x.items()})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
