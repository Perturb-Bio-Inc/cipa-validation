#!/usr/bin/env python3
"""
T-151 — CiPA validation harness for the Kernik-2019 iPSC-CM model.

Question: does a substrate-specific iPSC-CM action-potential model classify the CiPA
28-drug TdP-risk categories, and how does it compare to the classic hERG-only baseline?

Design (CiPA discipline):
  1. Reference set = science/cipa_validation/cipa_28drug_reference.csv (T-150), built from the
     FDA/CiPA primary artifact. 12 training + 16 validation drugs, TdP class High/Interm/Low.
  2. Apply multi-channel Hill block at 1-4x free Cmax to the Kernik-2019 iPSC-CM model
     (ta1_causal_prototype/ipsc_cm_ap_model.py). Common panel across all 28 drugs =
     hERG (g_Kr, static Hill fit), ICaL (g_CaL), INa-peak (g_Na). Kernik has NO distinct
     late-sodium current, so INaL block is NOT representable (documented limitation; it
     biases against INaL-protective low-risk drugs e.g. ranolazine, mexiletine).
  3. Metric = worst-case repolarization risk over the exposure sweep: fractional APD90
     prolongation vs drug-free baseline, with repolarization collapse (loss of normal beats
     into a depolarized state) scored as a severe fixed value. Metric is defined here and
     FROZEN before any threshold is fit.
  4. Fit two ordinal thresholds on the 12 TRAINING drugs only, freeze, score the 16
     VALIDATION drugs blind.
  5. Baseline = hERG-only safety margin (Cmax / hERG_IC50), the pre-CiPA classifier CiPA
     showed to be insufficient. Same train/freeze/score protocol.

Run: ~/.venvs/myokit/bin/python cipa_validation_harness.py
Outputs: prints a report and writes cipa_validation_results.csv next to this file.
"""
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TA1 = os.path.join(HERE, "..", "ta1_causal_prototype")
sys.path.insert(0, TA1)
from ipsc_cm_ap_model import IPSCModel, extract_biomarkers  # noqa: E402

REF = os.path.join(HERE, "cipa_28drug_reference.csv")
CONCS = [1.0, 2.0, 3.0, 4.0]          # multiples of free Cmax (CiPA exposure range)
ABN_RISK = 2.0                         # risk value assigned to a repolarization collapse
CLASSES = ["Low", "Intermediate", "High"]
CLS_IDX = {"Low": 0, "Interm": 1, "Intermediate": 1, "High": 2}


def hill_block(conc, ic50, h):
    """Fraction of channel blocked at concentration `conc` (same units as ic50)."""
    if ic50 is None or ic50 <= 0 or conc <= 0:
        return 0.0
    return 1.0 / (1.0 + (ic50 / conc) ** h)


def num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def load_drugs():
    rows = []
    for r in csv.DictReader(open(REF)):
        if r["set"] == "extra":         # flecainide: not part of canonical 28
            continue
        cmax = num(r["cmax_free_nM"])
        if cmax is None:
            continue
        # hERG: prefer static Hill fit; fall back to dynamic value if static missing
        herg_ic50 = num(r["hERG_static_IC50_nM"]) or num(r["hERG_IC50_nM"])
        herg_h = num(r["hERG_static_h"]) or num(r["hERG_h"]) or 1.0
        rows.append(dict(
            drug=r["drug"], set=r["set"], cls=r["tdp_class"], cmax=cmax,
            herg_ic50=herg_ic50, herg_h=herg_h,
            ical_ic50=num(r["ICaL_IC50_nM"]), ical_h=num(r["ICaL_h"]) or 1.0,
            ina_ic50=num(r["INa_peak_IC50_nM"]), ina_h=num(r["INa_peak_h"]) or 1.0,
        ))
    return rows


def repolarization_collapse(t, v, base_mdp):
    """True if, with no clean beats, the cell sits in an abnormally depolarized state
    (mean V well above baseline MDP), i.e. repolarization failure / loss of excitability
    rather than benign quiescence at rest."""
    return bool(np.mean(v) > base_mdp + 20.0)


def drug_risk(drug, base_apd90, base_mdp):
    """Repolarization risk across the exposure sweep, plus the per-conc detail.
    Common panel: hERG (g_Kr), ICaL (g_CaL), INa-peak (g_Na).

    Returns (risk_primary, risk_graded, detail).
      risk_primary = max repolarization risk over 1-4x (prolongation, or ABN_RISK on collapse).
      risk_graded  = same, but collapse is graded by the LOWEST multiple at which it appears
                     (collapse at 1x is worse than at 4x), so strong blockers don't all
                     saturate at one value. Non-collapsing drugs keep max prolongation.
    """
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
        m = IPSCModel(scales)
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
            r = 0.0                      # quiescent at rest, not a TdP signal
            state = "quiescent"
        detail.append(dict(mult=mult, herg_block=round(scales["g_Kr"] and 1 - scales["g_Kr"], 3),
                           apd90=(None if np.isnan(a90) else round(a90, 1)), state=state, r=round(r, 3)))
        risk = max(risk, r)
    # graded: separate collapsing drugs by dose-to-collapse (lower dose = higher risk)
    if first_collapse is not None:
        graded = ABN_RISK + (max(CONCS) - first_collapse) / max(CONCS)
    else:
        graded = risk
    return risk, graded, detail


def fit_two_thresholds(scores, labels):
    """Pick two cut-points on a 1-D score to map into 3 ordinal classes, maximizing
    accuracy on (scores, labels). Brute-force over midpoints of sorted unique scores."""
    s = np.asarray(scores, float)
    y = np.asarray([CLS_IDX[c] for c in labels])
    cand = np.unique(s)
    mids = [(-np.inf,)] + [((cand[i] + cand[i + 1]) / 2,) for i in range(len(cand) - 1)]
    cuts = sorted(set([m[0] for m in mids] + [c for c in cand]))
    best, best_acc = (None, None), -1
    for i in range(len(cuts)):
        for j in range(i, len(cuts)):
            lo, hi = cuts[i], cuts[j]
            pred = np.where(s <= lo, 0, np.where(s <= hi, 1, 2))
            acc = np.mean(pred == y)
            if acc > best_acc:
                best_acc, best = acc, (lo, hi)
    return best


def classify(scores, cuts):
    lo, hi = cuts
    s = np.asarray(scores, float)
    return np.where(s <= lo, 0, np.where(s <= hi, 1, 2))


def confusion(y_true, y_pred):
    M = np.zeros((3, 3), int)
    for a, b in zip(y_true, y_pred):
        M[a, b] += 1
    return M


def report_block(name, drugs, scores, cuts):
    y = np.array([CLS_IDX[d["cls"]] for d in drugs])
    pred = classify(scores, cuts)
    M = confusion(y, pred)
    acc = np.mean(pred == y)
    adj = np.mean(np.abs(pred - y) <= 1)          # adjacent (off-by-one-class) accuracy
    # high-vs-not (the safety-critical call)
    yh, ph = (y == 2).astype(int), (pred == 2).astype(int)
    tp = int(np.sum((yh == 1) & (ph == 1))); fn = int(np.sum((yh == 1) & (ph == 0)))
    fp = int(np.sum((yh == 0) & (ph == 1)))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    return dict(name=name, n=len(drugs), acc=acc, adj=adj, confusion=M.tolist(),
                high_sens=sens, high_fp=fp, cuts=cuts,
                per_drug=[(d["drug"], d["cls"], CLASSES[pred[i]], round(float(scores[i]), 3))
                          for i, d in enumerate(drugs)])


def main():
    drugs = load_drugs()
    assert len(drugs) == 28, f"expected 28 canonical drugs, got {len(drugs)}"
    base = IPSCModel().biomarkers()
    base_apd90, base_mdp = base["apd90"], base["mdp"]
    print(f"baseline (drug-free) Kernik-2019: APD90={base_apd90:.1f} ms, MDP={base_mdp:.1f} mV, "
          f"rate={base['rate_bpm']:.1f} bpm\n")

    # ---- our model: multi-channel iPSC-CM AP risk ----
    for i, d in enumerate(drugs):
        d["risk"], d["graded"], d["detail"] = drug_risk(d, base_apd90, base_mdp)
        print(f"  [{i+1:2d}/28] {d['drug']:15s} {d['set']:10s} {d['cls']:12s} "
              f"risk={d['risk']:.3f} graded={d['graded']:.3f}")

    # ---- baseline: hERG-only safety margin ----
    # score = -log10(hERG_IC50 / Cmax): higher score = tighter margin = higher predicted risk
    for d in drugs:
        margin = d["herg_ic50"] / d["cmax"] if d["herg_ic50"] else 1e9
        d["herg_score"] = -np.log10(margin)

    train = [d for d in drugs if d["set"] == "training"]
    valid = [d for d in drugs if d["set"] == "validation"]

    results = {}
    for key, score_key in [("ipsc_multichannel", "risk"), ("ipsc_graded", "graded"),
                           ("herg_only_baseline", "herg_score")]:
        cuts = fit_two_thresholds([d[score_key] for d in train], [d["cls"] for d in train])
        tr = report_block(f"{key} [TRAIN]", train, [d[score_key] for d in train], cuts)
        va = report_block(f"{key} [VALID]", valid, [d[score_key] for d in valid], cuts)
        results[key] = dict(cuts=cuts, train=tr, valid=va)

    # ---- print report ----
    for key in ("ipsc_multichannel", "ipsc_graded", "herg_only_baseline"):
        r = results[key]
        print("\n" + "=" * 70)
        print(key)
        print(f"  thresholds (fit on training, frozen): {r['cuts']}")
        for split in ("train", "valid"):
            b = r[split]
            print(f"  {split.upper()} (n={b['n']}): accuracy={b['acc']:.2f}  "
                  f"adjacent={b['adj']:.2f}  High-sensitivity={b['high_sens']:.2f}  "
                  f"High-false-pos={b['high_fp']}")
            print(f"     confusion rows=true[L,I,H] cols=pred[L,I,H]: {b['confusion']}")

    # ---- write per-drug CSV ----
    out = os.path.join(HERE, "cipa_validation_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drug", "set", "true_class", "ipsc_risk", "ipsc_pred",
                    "herg_score", "herg_pred"])
        ip = {d["drug"]: p for d in drugs
              for p in [CLASSES[classify([d["risk"]], results["ipsc_multichannel"]["cuts"])[0]]]}
        hp = {d["drug"]: p for d in drugs
              for p in [CLASSES[classify([d["herg_score"]], results["herg_only_baseline"]["cuts"])[0]]]}
        for d in drugs:
            w.writerow([d["drug"], d["set"], d["cls"], round(d["risk"], 3), ip[d["drug"]],
                        round(d["herg_score"], 3), hp[d["drug"]]])
    # machine-readable summary
    json.dump({k: {"cuts": list(map(float, results[k]["cuts"])),
                   "train": {kk: results[k]["train"][kk] for kk in ("acc", "adj", "high_sens", "high_fp", "confusion")},
                   "valid": {kk: results[k]["valid"][kk] for kk in ("acc", "adj", "high_sens", "high_fp", "confusion")}}
               for k in results},
              open(os.path.join(HERE, "cipa_validation_summary.json"), "w"), indent=2)
    print(f"\nwrote {out}")
    print("wrote cipa_validation_summary.json")
    return results, drugs


if __name__ == "__main__":
    main()
