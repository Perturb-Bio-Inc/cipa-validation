#!/usr/bin/env python3
"""
I6 — population-of-models uncertainty for the CiPA-28 risk call.

Question: if we represent parameter uncertainty in the Kernik-2019 iPSC-CM model as a small
population of conductance-scaled myocytes, what does the existing frozen product (the risk score
with its frozen cuts lo=-0.104378, hi=0.349137) report per drug? I.e. propagate a defensible
spread of cell states through the already-frozen classifier and read out P(Low|Intermed|High).

Design (kept deliberately simple and defensible):
  - Population = N members. Each member scales ALL 8 conductances by one shared log-normal factor
    f = exp(g), g ~ N(0, sigma=0.25). This represents "a somewhat bigger / smaller myocyte" with
    fully correlated current densities, which is the simplest honest reading of model parameter
    uncertainty. It deliberately does NOT model which channel dominates, and it does not tune any
    parameter to make the benchmark pass.
  - Each member has its OWN drug-free baseline (APD90, MDP). Risk is fractional APD90 prolongation
    relative to that member's own baseline, identical block logic to cipa_validation_harness.py.
    A member whose drug-free baseline itself fails to repolarize is not a usable healthy cell; it
    is redrawn.
  - Frozen cuts (from the deterministic run) are applied to each member's risk. P(class) per drug =
    fraction of members whose risk lands in each band. The product's threshold is NOT re-fit.

Outputs: population_risk_summary.json, population_risk_results.csv, and a printed per-drug table.
"""
import os, sys, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TA1 = os.path.join(HERE, "..", "ta1_causal_prototype")
sys.path.insert(0, TA1)
sys.path.insert(0, HERE)

from ipsc_cm_ap_model import IPSCModel, extract_biomarkers  # noqa: E402
from cipa_validation_harness import (load_drugs, hill_block, repolarization_collapse,  # noqa: E402
                                     CONCS, ABN_RISK)

CONDS = ["g_Kr", "g_CaL", "g_Na", "g_Ks", "g_K1", "g_to", "g_f", "g_CaT"]
LO, HI = -0.104378, 0.349137          # frozen multichannel cuts (deterministic run)
SIGMA = 0.25
SEED = 20260906
N_POP = 10


def member_scales(rng, mode):
    """Draw one member's conductance scales.
    mode='correlated': a single shared log-normal factor on all conductances.
    mode='independent': each conductance drawn independently from log-normal."""
    if mode == "correlated":
        f = float(np.exp(rng.normal(0.0, SIGMA)))
        return {c: f for c in CONDS}
    return {c: float(np.exp(rng.normal(0.0, SIGMA))) for c in CONDS}


def baseline(m_scales):
    """A usable healthy myocyte: its drug-free APD90 is defined (it beats and repolarizes).
    repolarization_collapse is only meaningful when there are no clean beats, so it is NOT used
    here; a nan APD90 is the gate (a baseline that is quiescent or collapsed has no APD90)."""
    m = IPSCModel(m_scales)
    t, v = m.simulate()
    bm = extract_biomarkers(t, v)
    if np.isnan(bm["apd90"]):
        return None
    return dict(m_scales=m_scales, apd90=bm["apd90"], mdp=bm["mdp"])


def drug_risk_member(drug, base):
    risk = -np.inf
    first_collapse = None
    for mult in CONCS:
        c = mult * drug["cmax"]
        sc = dict(base["m_scales"])
        sc["g_Kr"] = base["m_scales"]["g_Kr"] * (1.0 - hill_block(c, drug["herg_ic50"], drug["herg_h"]))
        if drug["ical_ic50"]:
            sc["g_CaL"] = base["m_scales"]["g_CaL"] * (1.0 - hill_block(c, drug["ical_ic50"], drug["ical_h"]))
        if drug["ina_ic50"]:
            sc["g_Na"] = base["m_scales"]["g_Na"] * (1.0 - hill_block(c, drug["ina_ic50"], drug["ina_h"]))
        m = IPSCModel(sc)
        t, v = m.simulate()
        bm = extract_biomarkers(t, v)
        a90 = bm["apd90"]
        if not np.isnan(a90):
            r = (a90 - base["apd90"]) / base["apd90"]
        elif repolarization_collapse(t, v, base["mdp"]) or bm["repol_failure"]:
            r = ABN_RISK
            if first_collapse is None:
                first_collapse = mult
        else:
            r = 0.0
        risk = max(risk, r)
    return risk


def classify(risk, lo=LO, hi=HI):
    if risk <= lo:
        return "Low"
    if risk <= hi:
        return "Intermediate"
    return "High"


def run_mode(mode):
    rng = np.random.default_rng(SEED)
    drugs = load_drugs()
    assert len(drugs) == 28

    det = {}
    for r in csv.DictReader(open(os.path.join(HERE, "cipa_validation_results.csv"))):
        det[r["drug"]] = (float(r["ipsc_risk"]), r["ipsc_pred"], r["true_class"])

    members = []
    tries = 0
    while len(members) < N_POP and tries < N_POP * 50:
        tries += 1
        b = baseline(member_scales(rng, mode))
        if b is not None:
            members.append(b)
    print(f"[{mode}] population: {len(members)} usable members in {tries} draws\n")
    if not members:
        return

    summary = dict(seed=SEED, sigma=SIGMA, mode=mode, n=len(members),
                   cuts={"lo": LO, "hi": HI})
    rows = []
    for d in drugs:
        risks = np.array([drug_risk_member(d, m) for m in members])
        det_risk, det_cls, true_cls = det[d["drug"]]
        med, p5, p95 = float(np.median(risks)), float(np.percentile(risks, 5)), float(np.percentile(risks, 95))
        pc = [float(np.mean(risks <= LO)), float(np.mean((risks > LO) & (risks <= HI))),
              float(np.mean(risks > HI))]
        pop_cls = ["Low", "Intermediate", "High"][int(np.argmax(pc))]
        iqr = float(np.percentile(risks, 75) - np.percentile(risks, 25))
        rows.append(dict(drug=d["drug"], set=d["set"], true_class=true_cls,
                         det_risk=round(det_risk, 3), det_class=det_cls,
                         pop_median=round(med, 3), pop_p5=round(p5, 3), pop_p95=round(p95, 3),
                         iqr=round(iqr, 3), p_low=round(pc[0], 3), p_interm=round(pc[1], 3),
                         p_high=round(pc[2], 3), pop_class=pop_cls,
                         pop_correct=pop_cls == true_cls, det_correct=det_cls == true_cls))
        print(f"{d['drug']:15s} {true_cls:12s} det={det_cls:12s}({det_risk:+.3f}) "
              f"pop={pop_cls:12s} med={med:+.3f} [p5 {p5:+.3f}, p95 {p95:+.3f}] "
              f"P(L/I/H)={pc[0]:.2f}/{pc[1]:.2f}/{pc[2]:.2f} {'OK' if pop_cls==true_cls else 'MISS'}")

    summary["per_drug"] = rows
    summary["accuracy_det"] = round(sum(r["det_correct"] for r in rows) / len(rows), 3)
    summary["accuracy_pop_argmax"] = round(sum(r["pop_correct"] for r in rows) / len(rows), 3)
    amb = [r for r in rows if max(r["p_low"], r["p_interm"], r["p_high"]) <= 0.5]
    summary["n_ambiguous_population"] = len(amb)
    summary["ambiguous_drugs"] = [r["drug"] for r in amb]
    print(f"[{mode}] det accuracy {summary['accuracy_det']}, pop argmax accuracy "
          f"{summary['accuracy_pop_argmax']}, ambiguous (no class >0.5) {len(amb)}")

    with open(os.path.join(HERE, f"population_risk_summary_{mode}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(HERE, f"population_risk_results_{mode}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[{mode}] wrote population_risk_summary_{mode}.json, population_risk_results_{mode}.csv")


if __name__ == "__main__":
    modes = sys.argv[1:] or ["correlated", "independent"]
    for m in modes:
        run_mode(m)
