#!/usr/bin/env python3
"""
Free-Cmax sensitivity sweep on the CiPA-28 three-class call.

Question: every risk call hangs on one published free Cmax, and in a paid read the
buyer supplies that number. If the supplied Cmax is wrong by a factor, does the class
change, and by how much does it have to be wrong before it does?

Design. Reuse the committed protocol unchanged (cipa_validation_harness.drug_risk):
Hill block of IKr/ICaL/INa-peak at 1-4x free Cmax, worst-case fractional APD90
prolongation, collapse = ABN_RISK. The ONLY thing varied is the drug's free Cmax,
scaled by a factor. Thresholds stay FROZEN at the committed multichannel cuts; nothing
is re-fit. Scaling Cmax by k is exactly equivalent to sweeping exposure at k*[1,2,3,4]x.

Outputs cmax_sensitivity_results.csv and cmax_sensitivity_summary.json.
Run: ~/.venvs/myokit/bin/python cmax_sensitivity.py
"""
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cipa_validation_harness import (  # noqa: E402
    load_drugs, drug_risk, CLASSES,
)
from ipsc_cm_ap_model import IPSCModel, extract_biomarkers  # noqa: E402

# Frozen committed multichannel cuts. Not re-fit here.
LO, HI = -0.104378, 0.349137
FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0]
COMMITTED = os.path.join(HERE, "cipa_validation_results.csv")


def classify_one(score):
    if score <= LO:
        return "Low"
    if score <= HI:
        return "Intermediate"
    return "High"


def committed_calls():
    out = {}
    for r in csv.DictReader(open(COMMITTED)):
        out[r["drug"]] = (r["ipsc_pred"], float(r["ipsc_risk"]), r["true_class"])
    return out


def main():
    base = IPSCModel()
    t, v = base.simulate()
    bm = extract_biomarkers(t, v)
    base_apd90, base_mdp = bm["apd90"], bm["mdp"]
    print(f"baseline APD90 {base_apd90:.2f} ms   MDP {base_mdp:.2f} mV\n")

    drugs = load_drugs()
    comm = committed_calls()
    rows = []

    for d in drugs:
        per_factor = {}
        for f in FACTORS:
            scaled = dict(d)
            scaled["cmax"] = d["cmax"] * f
            risk, graded, detail = drug_risk(scaled, base_apd90, base_mdp)
            per_factor[f] = (risk, classify_one(risk))
        comm_pred, comm_risk, true_cls = comm.get(d["drug"], ("", float("nan"), d["cls"]))
        classes = [per_factor[f][1] for f in FACTORS]
        stable = len(set(classes)) == 1
        # smallest |log2 factor| at which the class differs from the 1.0x class
        ref_cls = per_factor[1.0][1]
        flips = [f for f in FACTORS if per_factor[f][1] != ref_cls]
        nearest_flip = min((abs(np.log2(f)) for f in flips), default=None)
        row = dict(
            drug=d["drug"], set=d["set"], true_class=true_cls,
            committed_pred=comm_pred, committed_risk=round(comm_risk, 6),
            stable_over_range=stable,
            nearest_flip_log2=None if nearest_flip is None else round(float(nearest_flip), 3),
        )
        for f in FACTORS:
            row[f"risk_{f}x"] = round(per_factor[f][0], 6)
            row[f"class_{f}x"] = per_factor[f][1]
        rows.append(row)
        flag = "STABLE" if stable else "FLIPS "
        print(f"  {d['drug']:<16} true {true_cls:<13} {flag} " +
              " ".join(f"{f}x:{per_factor[f][1][:5]}" for f in FACTORS))

    n_stable = sum(1 for r in rows if r["stable_over_range"])
    # accuracy at each factor
    acc = {}
    for f in FACTORS:
        acc[f"{f}x"] = sum(1 for r in rows if r[f"class_{f}x"] == r["true_class"])
    fragile = [r["drug"] for r in rows if not r["stable_over_range"]]

    print(f"\n  stable across 0.5-2.0x free Cmax : {n_stable}/{len(rows)}")
    print(f"  accuracy by factor               : {acc}")
    print(f"  fragile drugs                    : {fragile}")

    with open(os.path.join(HERE, "cmax_sensitivity_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(factors=FACTORS, cuts=dict(lo=LO, hi=HI), n_drugs=len(rows),
                   n_stable=n_stable, accuracy_by_factor=acc, fragile=fragile,
                   per_drug=rows),
              open(os.path.join(HERE, "cmax_sensitivity_summary.json"), "w"), indent=2)
    print("\nwrote cmax_sensitivity_results.csv, cmax_sensitivity_summary.json")


if __name__ == "__main__":
    main()
