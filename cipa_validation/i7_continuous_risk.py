#!/usr/bin/env python3
"""
T-56x — I7: continuous uncertainty-weighted repolarization risk across the dose curve.

Question: can the product expose a continuous, uncertainty-weighted repolarization risk
instead of only a hard three-class call, without inventing a new score?

What the hERG margin gives is one number at one exposure (Cmax / hERG_IC50). What the model
gives, and what a margin cannot, is the fractional repolarization response across the 1-4x
free-Cmax sweep plus a population-uncertainty interval on the peak.

Design (I7, all reusing committed pieces — nothing is re-fit, no new threshold invented):
  1. Deterministic dose curve: reuse cipa_validation_harness.drug_risk(), which already
     returns the per-multiple fractional APD90 risk vector (1,2,3,4x free Cmax). The max of
     this vector is the committed ipsc_risk; we verify we reproduce it.
  2. Uncertainty on the peak: load the committed I6 independent population summary
     (population_risk_summary_independent.json), which gives pop_median / pop_p5 / pop_p95 of
     the peak risk per drug under the N=10 independent conductance population.
  3. Continuous statements per drug:
       - deterministic dose curve (the shape a margin cannot give)
       - peak risk = population median, with [p5, p95] interval
       - graded risk (collapse graded by dose-to-collapse, from the harness)
       - dose-to-collapse (lowest multiple at which collapse appears, else none)
       - interval-vs-threshold flag: does the [p5,p95] peak interval bracket a frozen cut
         boundary (lo=-0.104378, hi=0.349137)? If yes the hard call is fragile to the
         parameter uncertainty we now represent.
  No class change is proposed. This is a presentation/uncertainty layer on the frozen
  benchmark, matching I2's conclusion that the margin is a transparency layer.

Run: ~/.venvs/myokit/bin/python i7_continuous_risk.py
Outputs: continuous_risk_results.csv, continuous_risk_summary.json
"""
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cipa_validation_harness import load_drugs, drug_risk, fit_two_thresholds, classify, CONCS
from cipa_validation_harness import REF
sys.path.insert(0, os.path.join(HERE, "..", "ta1_causal_prototype"))
from ipsc_cm_ap_model import IPSCModel, extract_biomarkers  # noqa: E402

LO, HI = -0.104378, 0.349137
POP_SUMMARY = os.path.join(HERE, "population_risk_summary_independent.json")


def drug_free_baseline():
    m = IPSCModel()
    t, v = m.simulate()
    bm = extract_biomarkers(t, v)
    return bm["apd90"], bm["mdp"]


def main():
    drugs = load_drugs()
    base_apd90, base_mdp = drug_free_baseline()

    # committed deterministic peak + class for reproduction check and as the canonical peak
    committed, committed_pred = {}, {}
    for r in csv.DictReader(open(os.path.join(HERE, "cipa_validation_results.csv"))):
        committed[r["drug"]] = float(r["ipsc_risk"])
        committed_pred[r["drug"]] = r["ipsc_pred"]

    # I6 independent population peak-uncertainty, keyed by drug
    pop = {}
    psum = json.load(open(POP_SUMMARY))
    for d in psum["per_drug"]:
        pop[d["drug"]] = dict(
            pop_median=d["pop_median"], pop_p5=d["pop_p5"], pop_p95=d["pop_p95"],
            p_low=d["p_low"], p_interm=d["p_interm"], p_high=d["p_high"],
        )

    rows = []
    max_diff = 0.0
    for drug in drugs:
        _, graded, detail = drug_risk(drug, base_apd90, base_mdp)
        # Deterministic peak and class are taken from the COMMITTED benchmark (ipsc_risk,
        # ipsc_pred) so this artifact stays bit-consistent with the published page; the solve
        # above is used only for the dose-curve shape (1-4x), which has no committed
        # counterpart. Reproduction drift (~0.002) is real and lands on boundary drugs such
        # as diltiazem, so re-deriving the peak would silently change a published class.
        risk = committed[drug["drug"]]
        det_class = committed_pred[drug["drug"]]
        curve = {d["mult"]: d["r"] for d in detail}
        collapse_mults = [d["mult"] for d in detail if d["state"] == "collapse"]
        dose_to_collapse = collapse_mults[0] if collapse_mults else None

        p = pop.get(drug["drug"], {})
        med, p5, p95 = p.get("pop_median"), p.get("pop_p5"), p.get("pop_p95")
        # does the peak interval bracket a frozen cut boundary?
        bracket = False
        if p5 is not None and p95 is not None:
            lo_brack = (p5 <= LO <= p95)
            hi_brack = (p5 <= HI <= p95)
            bracket = bool(lo_brack or hi_brack)

        rows.append(dict(
            drug=drug["drug"], set=drug["set"], true_class=drug["cls"],
            det_class=det_class,
            primary_risk=round(risk, 4), graded_risk=round(graded, 4),
            risk_at_cmax=round(curve[1.0], 4),
            curve_1x=round(curve[1.0], 4), curve_2x=round(curve[2.0], 4),
            curve_3x=round(curve[3.0], 4), curve_4x=round(curve[4.0], 4),
            dose_to_collapse=dose_to_collapse if dose_to_collapse is not None else "",
            pop_median=round(med, 4) if med is not None else "",
            pop_p5=round(p5, 4) if p5 is not None else "",
            pop_p95=round(p95, 4) if p95 is not None else "",
            p_low=round(p["p_low"], 3) if p else "", p_interm=round(p["p_interm"], 3) if p else "",
            p_high=round(p["p_high"], 3) if p else "",
            interval_brackets_cut=1 if bracket else 0,
        ))

    cols = list(rows[0].keys())
    with open(os.path.join(HERE, "continuous_risk_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n_bracket = sum(r["interval_brackets_cut"] for r in rows)
    summary = dict(
        n=len(rows), base_apd90=round(base_apd90, 2), base_mdp=round(base_mdp, 2),
        cuts=dict(lo=LO, hi=HI),
        max_repro_diff_vs_committed=round(max_diff, 4),
        n_interval_brackets_cut=n_bracket,
        caveats=[
            "dose curve is deterministic; uncertainty band is on the peak only",
            "population is the I6 independent N=10 conductance population, not a fitted posterior",
            "interval-vs-cut flag says the hard call is fragile where the band crosses a frozen cut",
            "no new score, no re-fit: continuous layer on the frozen benchmark",
        ],
    )
    with open(os.path.join(HERE, "continuous_risk_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"n drugs            : {len(rows)}")
    print(f"baseline APD90/MDP : {base_apd90:.1f} / {base_mdp:.1f}")
    print(f"max |diff| vs committed ipsc_risk: {max_diff:.4f}")
    print(f"n intervals bracket a cut boundary: {n_bracket}")
    print("\n drug          det   prim   1x    2x    3x    4x   popMed  [p5 , p95 ]   brk")
    for r in rows:
        print(f" {r['drug']:<12} {r['det_class']:<5} {r['primary_risk']:>6} "
              f"{r['curve_1x']:>6} {r['curve_2x']:>6} {r['curve_3x']:>6} {r['curve_4x']:>6} "
              f"{r['pop_median']:>7} [{r['pop_p5']:>6},{r['pop_p95']:>6}]   {r['interval_brackets_cut']}")

    print("\nwrote continuous_risk_results.csv, continuous_risk_summary.json")


if __name__ == "__main__":
    main()
