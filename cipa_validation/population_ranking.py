#!/usr/bin/env python3
"""
Population through the mechanism ranking.

The product's core claim is the mechanism ranking (ranked conductance-change causes from a
trace), and I6 only varied conductances for the risk label. This tests whether the rank-1
single and rank-1 pair mechanism survive the model's conductance uncertainty.

DESIGN
------
- The candidate library is FIXED at baseline: the 1560-hypothesis grid (60 singles + 1500
  pairs) whose biomarker signatures were committed in worked_example_grid_shard_*.json.
- Scoring is on DELTAS from each model's own drug-free baseline, so a population member's
  background scaling (a common offset across all hypotheses) does not dominate: the fit asks
  which candidate mechanism best explains the drug-INDUCED change. This is the same question
  the product answers against a measured trace.
- Targets are generated on each I6 independent-mode member (same seed 20260906, sigma 0.25,
  N=10, background x Hill block on IKr/ICaL/INa-peak). rank-1 single and rank-1 pair are
  compared to the baseline (delta-scored) rank-1 for each compound at 1x and 4x free Cmax.

Run (Myokit venv):  python3 population_ranking.py
"""
import csv
import json
import os
import sys

import numpy as np

sys_path = os.path.dirname(os.path.abspath(__file__))
TA1 = os.path.join(sys_path, "..", "ta1_causal_prototype")
sys.path.insert(0, sys_path)
sys.path.insert(0, TA1)

from ipsc_cm_ap_model import IPSCModel, extract_biomarkers
from inverse_ranker import BIOMARKERS, CANDIDATE_CURRENTS, CHANNEL_NAMES

from population_risk import member_scales, baseline, SEED, SIGMA, N_POP
from cipa_validation_harness import load_drugs, hill_block, CONCS
SHARD = os.path.join(TA1, "worked_example_grid_shard_{k}.json")
RANKS = os.path.join(TA1, "worked_example_rankings.json")
OUT = os.path.join(sys_path, "population_ranking_results.csv")
OUT_SUM = os.path.join(sys_path, "population_ranking_summary.json")

COMPOUNDS = ["dofetilide", "verapamil", "ranolazine", "astemizole"]
RANK_AT = [1.0, 4.0]
COLLAPSE_MATCH = 0.05
COLLAPSE_PENALTY = 10.0


def load_grid():
    grid = []
    k = 0
    while os.path.exists(SHARD.format(k=k)):
        grid += json.load(open(SHARD.format(k=k)))
        k += 1
    singles = [g for g in grid if len(g["currents"]) == 1]
    pairs = [g for g in grid if len(g["currents"]) == 2]
    return singles, pairs


def is_num(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def delta_of(bm, base):
    """Delta biomarker vector of a drug signature vs a drug-free baseline. None if collapsed."""
    if not is_num(bm.get("apd90")):
        return None
    return {k: bm[k] - base[k] for k in BIOMARKERS if is_num(bm.get(k))}


def delta_fit(target_delta, got_delta):
    """Normalized RMSE over deltas. Lower better. Mirrors inverse_ranker.fit_score."""
    if got_delta is None:
        return COLLAPSE_MATCH if target_delta is None else COLLAPSE_PENALTY
    if target_delta is None:
        return COLLAPSE_PENALTY
    errs = []
    for k, scale in BIOMARKERS.items():
        if k in target_delta and target_delta[k] is not None:
            gv = got_delta.get(k)
            if gv is None:
                return COLLAPSE_PENALTY
            errs.append((gv - target_delta[k]) / scale)
    return float(np.sqrt(np.mean(np.square(errs))))


def precompute_delta(hyp, base):
    """Delta vector of a grid hypothesis vs baseline. None if that hypothesis collapsed."""
    return delta_of(hyp["biomarkers"], base)


def rank_delta(hypotheses, deltas, target_delta, top_n=3):
    scored = sorted(
        ((delta_fit(target_delta, d), h["currents"], h["scales"]) for h, d in zip(hypotheses, deltas)),
        key=lambda x: x[0])
    return scored[:top_n]


def main():
    singles, pairs = load_grid()
    print(f"grid: {len(singles)} singles, {len(pairs)} pairs")

    ranks = json.load(open(RANKS))
    base_bm = ranks["baseline"]
    base_delta_ref = {k: float(base_bm[k]) for k in BIOMARKERS if is_num(base_bm.get(k))}

    # delta vector for every grid hypothesis vs the baseline model's own drug-free baseline
    singles_delta = [precompute_delta(h, base_bm) for h in singles]
    pairs_delta = [precompute_delta(h, base_bm) for h in pairs]

    # I6 independent population, same seed/draws as committed population_risk
    rng = np.random.default_rng(SEED)
    members = []
    tries = 0
    while len(members) < N_POP and tries < N_POP * 50:
        tries += 1
        b = baseline(member_scales(rng, "independent"))
        if b is not None:
            members.append(b)
    print(f"population: {len(members)} usable members in {tries} draws")

    drugs = {d["drug"]: d for d in load_drugs()}

    summary_rows = []
    rows = []
    for comp in COMPOUNDS:
        drug = drugs[comp]
        rec = ranks["compounds"][comp]
        for mult in RANK_AT:
            tgt = rec["targets"][str(int(mult))]
            truth = tgt["currents_blocked_over_5pct"]
            t_bm = tgt["target_biomarkers"]
            target_delta = delta_of(t_bm, base_bm)          # baseline reference (absolute -> delta)

            # baseline (delta-scored) rank-1 reference
            b_s = rank_delta(singles, singles_delta, target_delta, 1)
            b_p = rank_delta(pairs, pairs_delta, target_delta, 1)
            ref_s = b_s[0][1] if b_s else []
            ref_p = b_p[0][1] if b_p else []

            per = {"compound": comp, "mult": mult, "truth": truth, "ref_single": ref_s,
                   "ref_pair": ref_p, "n": 0, "stable_single": 0, "stable_pair": 0,
                   "flip_singles": [], "flip_pairs": [], "collapsed": 0}
            for m in members:
                mb_bm = _member_base_bm(m)
                ms = dict(m["m_scales"])
                c = mult * drug["cmax"]
                ms["g_Kr"] = m["m_scales"]["g_Kr"] * (1.0 - hill_block(c, drug["herg_ic50"], drug["herg_h"]))
                if drug["ical_ic50"]:
                    ms["g_CaL"] = m["m_scales"]["g_CaL"] * (1.0 - hill_block(c, drug["ical_ic50"], drug["ical_h"]))
                if drug["ina_ic50"]:
                    ms["g_Na"] = m["m_scales"]["g_Na"] * (1.0 - hill_block(c, drug["ina_ic50"], drug["ina_h"]))
                mod = IPSCModel(ms)
                t, v = mod.simulate()
                bm = extract_biomarkers(t, v)
                per["n"] += 1
                m_target_delta = delta_of(bm, mb_bm)
                if m_target_delta is None:
                    per["collapsed"] += 1
                    continue
                s1 = rank_delta(singles, singles_delta, m_target_delta, 1)
                p1 = rank_delta(pairs, pairs_delta, m_target_delta, 1)
                s_keys = s1[0][1] if s1 else []
                p_keys = p1[0][1] if p1 else []
                per["stable_single"] += int(s_keys == ref_s)
                per["stable_pair"] += int(p_keys == ref_p)
                if s_keys != ref_s:
                    per["flip_singles"].append(",".join(s_keys) or "-")
                if p_keys != ref_p:
                    per["flip_pairs"].append(",".join(p_keys) or "-")
            rows.append(per)
            fr_s = per["stable_single"] / per["n"] if per["n"] else 0.0
            fr_p = per["stable_pair"] / per["n"] if per["n"] else 0.0
            print(f"\n{comp} @{mult:.0f}x  truth {truth}  ref single {ref_s} / pair {ref_p}")
            print(f"  single rank-1 stable {fr_s:.2f} ({per['stable_single']}/{per['n']})"
                  f"  pair rank-1 stable {fr_p:.2f} ({per['stable_pair']}/{per['n']})"
                  f"  collapsed {per['collapsed']}")
            if per["flip_singles"]:
                print(f"  flip singles: {per['flip_singles'][:4]}")
            if per["flip_pairs"]:
                print(f"  flip pairs: {per['flip_pairs'][:4]}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["compound", "mult", "truth", "ref_single", "ref_pair",
                                          "n", "stable_single", "stable_pair", "collapsed",
                                          "flip_singles", "flip_pairs"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], (list, dict)) else r[k])
                        for k in r})
    json.dump({"seed": SEED, "sigma": SIGMA, "n": len(members), "rows": rows},
              open(OUT_SUM, "w"), indent=1)
    print("\nwritten:", OUT, ",", OUT_SUM)


def _member_base_bm(m):
    mod = IPSCModel(m["m_scales"])
    t, v = mod.simulate()
    return extract_biomarkers(t, v)


if __name__ == "__main__":
    main()
