#!/usr/bin/env python3
"""
I2 — uncertainty-aware risk calls: margin-to-threshold confidence analysis.

Task (LIMITATIONS_PROGRAM_2026-09-06, item I2). The committed CiPA-28 benchmark reports a
single three-class call per drug. A point call hides how close it sat to a decision boundary,
which is where the model's own uncertainty lives.

Design (uses ONLY committed frozen data; no re-fit, no model re-run):
  1. Load cipa_validation_results.csv (the committed 28-drug output) and the frozen
     multichannel cuts from cipa_validation_summary.json (lo = -0.104378, hi = 0.349137).
  2. For each drug, margin = distance from its ipsc_risk to the nearest cut it must clear to
     keep its predicted class. A Low call's margin is (lo - risk) if risk < lo else negative
     (already crossed); general rule below.
  3. The confidence claim to test: are the 11 misses systematically nearer a boundary than the
     17 correct calls? If yes, the margin is an honest confidence signal within the model's
     own self-consistency.
  4. Classify each miss as near (margin <= 0.05) or clear (margin > 0.05) on the fractional-APD90
     risk scale, and report the actual margin for every miss.

Caveat, printed in the output: margin-to-boundary is a self-consistency confidence measure, not
a calibrated probability. A true P(class | data) needs a population-of-models or
input-uncertainty propagation (I6); this is the transparent first step.

Run: ~/.venvs/myokit/bin/python risk_uncertainty.py
Outputs: risk_uncertainty_summary.json, report to stdout.
"""
import os, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "cipa_validation_results.csv")
SUM = os.path.join(HERE, "cipa_validation_summary.json")

CUTS = tuple(json.load(open(SUM))["ipsc_multichannel"]["cuts"])  # (lo, hi)
NEAR_BAND = 0.05  # fractional APD90


def margin(risk, lo, hi):
    """Signed distance from risk to the cut governing the call.

    Low (risk < lo):  margin = lo - risk (positive below the boundary).
    High (risk > hi): margin = risk - hi.
    Intermediate (lo <= risk <= hi): margin = distance to the nearer cut.
    """
    if risk < lo:
        return lo - risk
    if risk > hi:
        return risk - hi
    return min(risk - lo, hi - risk)


def main():
    rows = list(csv.DictReader(open(RES)))
    lo, hi = CUTS
    out = []
    for r in rows:
        risk = float(r["ipsc_risk"])
        m = margin(risk, lo, hi)
        correct = r["ipsc_pred"] == r["true_class"]
        out.append({
            "drug": r["drug"], "set": r["set"], "true": r["true_class"],
            "pred": r["ipsc_pred"], "risk": round(risk, 4),
            "margin": round(m, 4), "near": m <= NEAR_BAND, "correct": correct,
        })

    misses = [o for o in out if not o["correct"]]
    corrects = [o for o in out if o["correct"]]
    m_miss = np.mean([o["margin"] for o in misses])
    m_corr = np.mean([o["margin"] for o in corrects])
    near_misses = [o for o in misses if o["near"]]

    print(f"frozen cuts: lo={lo:.4f}, hi={hi:.4f}; near band = {NEAR_BAND}")
    print(f"correct calls: {len(corrects)}/28   misses: {len(misses)}/28")
    print(f"mean margin-to-boundary, correct calls: {m_corr:.4f}")
    print(f"mean margin-to-boundary, misses:        {m_miss:.4f}")
    print(f"misses that are near-misses (margin<={NEAR_BAND}): {len(near_misses)}/{len(misses)}")
    print("\nall misses, with margin:")
    for o in sorted(misses, key=lambda x: x["margin"]):
        flag = "NEAR" if o["near"] else "clear"
        print(f"  {o['drug']:14s} true={o['true']:<6s} pred={o['pred']:<13s} "
              f"risk={o['risk']:+.3f} margin={o['margin']:.3f}  {flag}")
    print("\ncorrect calls with the smallest margins (model's least-confident correct calls):")
    for o in sorted(corrects, key=lambda x: x["margin"])[:5]:
        print(f"  {o['drug']:14s} true={o['true']:<6s} pred={o['pred']:<13s} "
              f"risk={o['risk']:+.3f} margin={o['margin']:.3f}")

    summary = {
        "frozen_cuts": list(CUTS), "near_band": NEAR_BAND,
        "n_total": len(out), "n_correct": len(corrects), "n_miss": len(misses),
        "mean_margin_correct": round(float(m_corr), 4),
        "mean_margin_miss": round(float(m_miss), 4),
        "n_near_misses": len(near_misses),
        "per_drug": sorted(out, key=lambda x: x["drug"]),
    }
    with open(os.path.join(HERE, "risk_uncertainty_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nwrote risk_uncertainty_summary.json")


if __name__ == "__main__":
    main()
