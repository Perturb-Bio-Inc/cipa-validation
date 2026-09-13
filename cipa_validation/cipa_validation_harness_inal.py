#!/usr/bin/env python3
"""
T-561 — CiPA 28-drug benchmark with a late-sodium current added to the Kernik model.

This is the SAME protocol as cipa_validation_harness.py (T-151), except that a fourth
current, INaL (g_NaL), enters the Hill-block panel and the model is the INaL-extended
Kernik-2019 from ipsc_cm_ap_model_inal.py (g_NaL = 0.10 mS/uF, ~1% late fraction).

Question: does adding the missing late-sodium current recover the INaL-protective low-risk
drugs (ranolazine, mexiletine) that the three-current model calls Intermediate?

Design is unchanged from T-151: Hill block at 1-4x free Cmax, worst-case fractional APD90
prolongation (collapse = ABN_RISK), two ordinal cuts fit on the 12 TRAINING drugs and frozen,
16 VALIDATION drugs scored blind. hERG baseline is recomputed identically (it is model-free)
and MUST match the committed cuts to validate the reimplementation.

Run:
  ~/.venvs/myokit/bin/python cipa_validation_harness_inal.py
Writes: cipa_validation_results_inal.csv and cipa_validation_summary_inal.json next to this
file. It does NOT overwrite the committed no-INaL outputs.

Run time: ~28 drugs x 4 concs x 1.4 s ≈ 2.5 min plus threshold fit.
"""
import os
import sys
import csv
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TA1 = os.path.join(HERE, "..", "ta1_causal_prototype")
sys.path.insert(0, TA1)

from ipsc_cm_ap_model_inal import IPSCModelINaL  # noqa: E402
from ipsc_cm_ap_model import extract_biomarkers  # noqa: E402
from cipa_validation_harness import (  # noqa: E402
    hill_block, num, ABN_RISK, CLASSES, CLS_IDX, repolarization_collapse,
    fit_two_thresholds, classify, confusion,
)

REF = os.path.join(HERE, "cipa_28drug_reference.csv")
CONCS = [1.0, 2.0, 3.0, 4.0]


def load_drugs():
    rows = []
    for r in csv.DictReader(open(REF)):
        if r["set"] == "extra":
            continue
        cmax = num(r["cmax_free_nM"])
        if cmax is None:
            continue
        herg_ic50 = num(r["hERG_static_IC50_nM"]) or num(r["hERG_IC50_nM"])
        herg_h = num(r["hERG_static_h"]) or num(r["hERG_h"]) or 1.0
        rows.append(dict(
            drug=r["drug"], set=r["set"], cls=r["tdp_class"], cmax=cmax,
            herg_ic50=herg_ic50, herg_h=herg_h,
            ical_ic50=num(r["ICaL_IC50_nM"]), ical_h=num(r["ICaL_h"]) or 1.0,
            ina_ic50=num(r["INa_peak_IC50_nM"]), ina_h=num(r["INa_peak_h"]) or 1.0,
            inal_ic50=num(r["INaL_IC50_nM"]), inal_h=num(r["INaL_h"]) or 1.0,
        ))
    return rows


def drug_risk(drug, base_apd90, base_mdp):
    """Repolarization risk across the exposure sweep. Four currents: IKr, ICaL, INa, INaL."""
    detail = []
    risk = -np.inf
    first_collapse = None
    for mult in CONCS:
        c = mult * drug["cmax"]
        scales = {"g_Kr": 1.0 - hill_block(c, drug["herg_ic50"], drug["herg_h"])}
        if drug["ical_ic50"]:
            scales["g_CaL"] = 1.0 - hill_block(c, drug["ical_ic50"], drug["ical_h"])
        if drug["ina_ic50"]:
            scales["g_Na"] = 1.0 - hill_block(c, drug["ina_ic50"], drug["ina_h"])
        if drug["inal_ic50"]:
            scales["g_NaL"] = 1.0 - hill_block(c, drug["inal_ic50"], drug["inal_h"])
        m = IPSCModelINaL(scales)
        t, v = m.simulate()
        bm = extract_biomarkers(t, v)
        a90 = bm["apd90"]
        if not np.isnan(a90):
            r = (a90 - base_apd90) / base_apd90
            state = "beat"
        elif repolarization_collapse(t, v, base_mdp) or bm["repol_failure"]:
            r = ABN_RISK
            state = "collapse"
            if first_collapse is None:
                first_collapse = mult
        else:
            r = 0.0
            state = "quiescent"
        detail.append(dict(mult=mult,
                           herg_block=round(1 - scales["g_Kr"], 3),
                           inal_block=round((scales.get("g_NaL", 1.0) and 1 - scales.get("g_NaL", 1.0)), 3),
                           apd90=(None if np.isnan(a90) else round(a90, 1)),
                           state=state, r=round(r, 3)))
        risk = max(risk, r)
    if first_collapse is not None:
        graded = ABN_RISK + (max(CONCS) - first_collapse) / max(CONCS)
    else:
        graded = risk
    return risk, graded, detail


def report_block(name, drugs, scores, cuts):
    y = np.array([CLS_IDX[d["cls"]] for d in drugs])
    pred = classify(scores, cuts)
    M = confusion(y, pred)
    acc = np.mean(pred == y)
    adj = np.mean(np.abs(pred - y) <= 1)
    yh, ph = (y == 2).astype(int), (pred == 2).astype(int)
    tp = int(np.sum((yh == 1) & (ph == 1))); fn = int(np.sum((yh == 1) & (ph == 0)))
    fp = int(np.sum((yh == 0) & (ph == 1)))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    return dict(name=name, n=len(drugs), acc=acc, adj=adj, confusion=M.tolist(),
                high_sens=sens, high_fp=fp, cuts=cuts)


def main():
    drugs = load_drugs()
    assert len(drugs) == 28, f"expected 28 canonical drugs, got {len(drugs)}"
    base = IPSCModelINaL({}).biomarkers()
    base_apd90, base_mdp = base["apd90"], base["mdp"]
    print(f"baseline (drug-free, WITH INaL g_NaL=0.10): APD90={base_apd90:.1f} ms, "
          f"MDP={base_mdp:.1f} mV, rate={base['rate_bpm']:.1f} bpm\n")

    for i, d in enumerate(drugs):
        d["risk"], d["graded"], d["detail"] = drug_risk(d, base_apd90, base_mdp)
        print(f"  [{i+1:2d}/28] {d['drug']:15s} {d['set']:10s} {d['cls']:12s} "
              f"risk={d['risk']:.3f} graded={d['graded']:.3f}")

    for d in drugs:
        margin = d["herg_ic50"] / d["cmax"] if d["herg_ic50"] else 1e9
        d["herg_score"] = -np.log10(margin)

    train = [d for d in drugs if d["set"] == "training"]
    valid = [d for d in drugs if d["set"] == "validation"]

    results = {}
    for key, score_key in [("ipsc_multichannel", "risk"), ("ipsc_graded", "graded"),
                           ("herg_only_baseline", "herg_score")]:
        cuts = fit_two_thresholds([d[score_key] for d in train], [d["cls"] for d in train])
        results[key] = dict(cuts=cuts,
                            train=report_block(f"{key} [TRAIN]", train, [d[score_key] for d in train], cuts),
                            valid=report_block(f"{key} [VALID]", valid, [d[score_key] for d in valid], cuts))

    for key in ("ipsc_multichannel", "ipsc_graded", "herg_only_baseline"):
        r = results[key]
        print("\n" + "=" * 70)
        print(key, f"cuts(fit train, frozen)={r['cuts']}")
        for split in ("train", "valid"):
            b = r[split]
            print(f"  {split.upper()} (n={b['n']}): acc={b['acc']:.2f} adj={b['adj']:.2f} "
                  f"High-sens={b['high_sens']:.2f} High-fp={b['high_fp']}")
            print(f"     confusion {b['confusion']}")

    # per-drug CSV with INaL classes
    out = os.path.join(HERE, "cipa_validation_results_inal.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drug", "set", "true_class", "ipsc_risk_inal", "ipsc_pred_inal",
                    "herg_score", "herg_pred"])
        ip = {d["drug"]: CLASSES[classify([d["risk"]], results["ipsc_multichannel"]["cuts"])[0]]
              for d in drugs}
        hp = {d["drug"]: CLASSES[classify([d["herg_score"]], results["herg_only_baseline"]["cuts"])[0]]
              for d in drugs}
        for d in drugs:
            w.writerow([d["drug"], d["set"], d["cls"], round(d["risk"], 3), ip[d["drug"]],
                        round(d["herg_score"], 3), hp[d["drug"]]])
    json.dump({k: {"cuts": list(map(float, results[k]["cuts"])),
                   "train": {kk: results[k]["train"][kk] for kk in ("acc", "adj", "high_sens", "high_fp", "confusion")},
                   "valid": {kk: results[k]["valid"][kk] for kk in ("acc", "adj", "high_sens", "high_fp", "confusion")}}
               for k in results},
              open(os.path.join(HERE, "cipa_validation_summary_inal.json"), "w"), indent=2)
    print(f"\nwrote {out} and cipa_validation_summary_inal.json")


if __name__ == "__main__":
    main()
