#!/usr/bin/env python3
"""
I1 - mechanism recovery on the worked-example compounds, plus a curated
channel-block ground truth for all 28 CiPA drugs.

Question (LIMITATIONS_PROGRAM item I1): the mechanism ranking is tested only
against the model's own output. How well does the inverse ranking recover the
channels the forward model actually blocked?

This is a self-recovery test, not independent validation. The target phenotype
and the ground-truth block set both derive from the same CiPA reference IC50
data. It measures resolving power and its failure modes.

Ground truth: for each drug and multiple, the set of channels blocked >5% of the
maximal conductance by a static Hill block at that multiple of free Cmax,
recomputed here from cipa_28drug_reference.csv and compared against the
`currents_blocked_over_5pct` field already stored in worked_example_rankings.json
(consistency check).

Recovery: rank-1 single and rank-1 pair from worked_example_rankings.json,
mapped from human labels to conductance keys, compared to the ground-truth set.

Run: ~/.venvs/myokit/bin/python i1_mechanism_recovery.py
Outputs: i1_mechanism_recovery_results.csv, i1_mechanism_recovery_summary.json
"""
import os, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TA1 = os.path.join(HERE, "..", "ta1_causal_prototype")
REF = os.path.join(HERE, "cipa_28drug_reference.csv")
RANK = os.path.join(TA1, "worked_example_rankings.json")

# human label -> conductance key (the keys the ground-truth set uses)
LABEL_MAP = {
    "hERG / IKr": "g_Kr", "Cav1.2 / ICaL": "g_CaL", "NaV / INa": "g_Na",
    "Kir2.1 / IK1": "g_K1", "IKs": "g_Ks", "Ito": "g_to", "I_f": "g_f",
    "I_CaT": "g_CaT",
}
# reference columns for the six ranker conductances
CH = {
    "g_Kr": ("hERG_IC50_nM", "hERG_h"),
    "g_CaL": ("ICaL_IC50_nM", "ICaL_h"),
    "g_Na": ("INa_peak_IC50_nM", "INa_peak_h"),
    "g_Ks": ("IKs_IC50_nM", "IKs_h"),
    "g_K1": ("IK1_IC50_nM", "IK1_h"),
    "g_to": ("Ito_IC50_nM", "Ito_h"),
}
BLOCK_THRESHOLD = 0.05   # >5% counts as 'blocked'


def num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def hill_block(conc, ic50, h):
    if ic50 is None or conc <= 0:
        return 0.0
    return 1.0 / (1.0 + (ic50 / conc) ** (h or 1.0))


def load_reference():
    rows = {}
    for r in csv.DictReader(open(REF)):
        if r.get("set") == "extra":
            continue
        rows[r["drug"]] = r
    return rows


def curated_blocked(ref, mult):
    """Set of channels with block fraction >5% at `mult` x free Cmax."""
    cmax = num(ref["cmax_free_nM"])
    blocked = set()
    fracs = {}
    if cmax is None:
        return blocked, fracs
    for k, (ic50c, hc) in CH.items():
        f = hill_block(mult * cmax, num(ref[ic50c]), num(ref[hc]))
        fracs[k] = round(f, 4)
        if f > BLOCK_THRESHOLD:
            blocked.add(k)
    return blocked, fracs


def load_rankings():
    return json.load(open(RANK))


def mk(names):
    return set(LABEL_MAP[n] for n in names if n in LABEL_MAP)


def tag(a, b):
    if a == b:
        return "RECOVER"
    if not a:
        return "NULL"
    if a <= b or b <= a:
        return "PARTIAL"
    return "MISS"


def main():
    ref = load_reference()
    rankings = load_rankings()

    rows = []
    per_compound = {}
    for name, c in rankings["compounds"].items():
        per_compound[name] = {"sing": {"recover": 0, "total": 0},
                              "pair": {"recover": 0, "total": 0}}
        for mult in sorted(c["targets"].keys(), key=float):
            t = c["targets"][mult]
            # the stored set is the 3-channel panel that actually generated the
            # target phenotype (the ranking's own block_frac covers g_Kr/g_CaL/g_Na)
            true_stored = set(t["currents_blocked_over_5pct"])
            # full 6-channel profile from the reference IC50 data (diagnostic only:
            # it includes channels the forward target never blocked)
            true_curated, fracs = curated_blocked(ref[name], float(mult))
            s1 = mk(t["singles"][0]["currents"])
            p1 = mk(t["pairs"][0]["currents"])
            ts, tp = tag(s1, true_stored), tag(p1, true_stored)
            rows.append({
                "compound": name, "mult": mult,
                "true": ",".join(sorted(true_stored)) or "-",
                "full6_profile": ",".join(sorted(true_curated)) or "-",
                "sing1": ",".join(sorted(s1)) or "-", "sing_tag": ts,
                "pair1": ",".join(sorted(p1)) or "-", "pair_tag": tp,
            })
            per_compound[name]["sing"]["total"] += 1
            per_compound[name]["pair"]["total"] += 1
            if ts == "RECOVER":
                per_compound[name]["sing"]["recover"] += 1
            if tp == "RECOVER":
                per_compound[name]["pair"]["recover"] += 1

    # all-28 ground truth at 1x
    gt28 = []
    for drug in sorted(ref):
        blocked, fracs = curated_blocked(ref[drug], 1.0)
        gt28.append({
            "drug": drug, "set": ref[drug]["set"], "tdp_class": ref[drug]["tdp_class"],
            "blocked_gt5pct": ",".join(sorted(blocked)) or "-",
            "dominant": max(fracs, key=fracs.get) if any(fracs.values()) else "-",
            "fracs": {k: fracs[k] for k in sorted(fracs)},
        })

    total_sing = sum(v["sing"]["total"] for v in per_compound.values())
    total_pair = sum(v["pair"]["total"] for v in per_compound.values())
    rec_sing = sum(v["sing"]["recover"] for v in per_compound.values())
    rec_pair = sum(v["pair"]["recover"] for v in per_compound.values())

    summary = {
        "question": "mechanism recovery of the inverse ranking against its own forward inputs",
        "n_compounds": len(per_compound),
        "single_recovery": f"{rec_sing}/{total_sing}",
        "pair_recovery": f"{rec_pair}/{total_pair}",
        "per_compound": per_compound,
        "ground_truth_n28": gt28,
        "caveat": "self-recovery test: target and ground truth share the CiPA IC50 input family; measures resolving power, not wet accuracy",
    }

    with open(os.path.join(HERE, "i1_mechanism_recovery_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "i1_mechanism_recovery_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"single-current recovery: {rec_sing}/{total_sing}")
    print(f"pair recovery:           {rec_pair}/{total_pair}")
    print("per compound:")
    for name, v in per_compound.items():
        print(f"  {name}: single {v['sing']['recover']}/{v['sing']['total']}, "
              f"pair {v['pair']['recover']}/{v['pair']['total']}")
    print("\nall-28 ground truth (blocked>5% at 1x):")
    for g in gt28:
        print(f"  {g['drug']:12} {g['tdp_class']:3} {g['blocked_gt5pct']}")
    print("\nwrote i1_mechanism_recovery_results.csv, i1_mechanism_recovery_summary.json")


if __name__ == "__main__":
    main()
