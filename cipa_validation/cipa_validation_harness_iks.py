#!/usr/bin/env python3
"""
I4 — does adding IKs (g_Ks) block to the risk-score panel change the CiPA-28 result?

Baseline (committed, cipa_validation_harness.py): panel = hERG (g_Kr), ICaL (g_CaL),
INa-peak (g_Na). Accuracy 8/12 train, 9/16 blind, 17/28 all.

This variant adds g_Ks (IKs) to the panel wherever an IKs IC50 exists in the reference file.
Only 7 of 28 drugs carry IKs data (bepridil, cisapride, ondansetron, quinidine, ranolazine,
sotalol, terfenadine). The question is empirical: does IKs block help or hurt, and which calls
move?

Protocol is byte-for-byte the committed one: same CONCS (1-4x free Cmax), same Hill block,
same fractional APD90 risk with collapse ceiling, thresholds fit on the 12 TRAINING drugs and
frozen, 16 VALIDATION scored blind. Only the block panel differs.

Run: ~/.venvs/myokit/bin/python cipa_validation_harness_iks.py
Outputs: cipa_validation_results_iks.csv, cipa_validation_summary_iks.json
"""
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the frozen protocol pieces verbatim so block logic and metrics are identical.
from cipa_validation_harness import (  # noqa: E402
    load_drugs, hill_block, repolarization_collapse, extract_biomarkers,
    fit_two_thresholds, classify, report_block, CONCS, ABN_RISK, CLASSES, CLS_IDX,
)
from ipsc_cm_ap_model import IPSCModel  # noqa: E402  (via cipa_validation_harness import path)


def load_drugs_iks():
    """Same as harness load_drugs, but also read IKs columns (IKs_IC50_nM, IKs_h)."""
    ref = os.path.join(HERE, "cipa_28drug_reference.csv")

    def num(x):
        try:
            v = float(x)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    drugs = []
    for r in csv.DictReader(open(ref)):
        if r.get("set", "").strip() == "extra":
            continue
        herg = num(r["hERG_static_IC50_nM"]) or num(r["hERG_IC50_nM"])
        herg_h = num(r["hERG_static_h"]) or num(r["hERG_h"]) or 1.0
        drugs.append(dict(
            drug=r["drug"], set=r["set"], cls=r["tdp_class"].strip(),
            cmax=num(r["cmax_free_nM"]),
            herg_ic50=herg, herg_h=herg_h,
            ical_ic50=num(r["ICaL_IC50_nM"]), ical_h=num(r["ICaL_h"]) or 1.0,
            ina_ic50=num(r["INa_peak_IC50_nM"]), ina_h=num(r["INa_peak_h"]) or 1.0,
            iks_ic50=num(r["IKs_IC50_nM"]), iks_h=num(r["IKs_h"]) or 1.0,
        ))
    return drugs


def drug_risk_iks(drug, base_apd90, base_mdp):
    """Identical to harness drug_risk, plus g_Ks block when an IKs IC50 exists."""
    detail = []
    risk = -np.inf
    for mult in CONCS:
        c = mult * drug["cmax"]
        scales = {"g_Kr": 1.0 - hill_block(c, drug["herg_ic50"], drug["herg_h"])}
        if drug["ical_ic50"]:
            scales["g_CaL"] = 1.0 - hill_block(c, drug["ical_ic50"], drug["ical_h"])
        if drug["ina_ic50"]:
            scales["g_Na"] = 1.0 - hill_block(c, drug["ina_ic50"], drug["ina_h"])
        if drug["iks_ic50"]:
            scales["g_Ks"] = 1.0 - hill_block(c, drug["iks_ic50"], drug["iks_h"])
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
        else:
            r = 0.0
            state = "quiescent"
        detail.append(dict(mult=mult, state=state, r=round(r, 3),
                           iks_block=(None if not drug["iks_ic50"]
                                      else round(hill_block(c, drug["iks_ic50"], drug["iks_h"]), 3))))
        risk = max(risk, r)
    return risk, detail


def main():
    drugs = load_drugs_iks()
    assert len(drugs) == 28, f"expected 28 canonical drugs, got {len(drugs)}"

    base = IPSCModel().biomarkers()
    base_apd90, base_mdp = base["apd90"], base["mdp"]
    print(f"baseline (drug-free) Kernik-2019: APD90={base_apd90:.1f} ms, MDP={base_mdp:.1f} mV")

    for i, d in enumerate(drugs):
        d["risk"], d["detail"] = drug_risk_iks(d, base_apd90, base_mdp)
        print(f"  [{i+1:2d}/28] {d['drug']:15s} {d['cls']:12s} risk={d['risk']:.3f} "
              f"(IKsIC50={'Y' if d['iks_ic50'] else 'n'})")

    train = [d for d in drugs if d["set"] == "training"]
    valid = [d for d in drugs if d["set"] == "validation"]

    cuts = fit_two_thresholds([d["risk"] for d in train], [d["cls"] for d in train])
    tr = report_block("ipsc_iks [TRAIN]", train, [d["risk"] for d in train], cuts)
    va = report_block("ipsc_iks [VALID]", valid, [d["risk"] for d in valid], cuts)
    al = report_block("ipsc_iks [ALL]", drugs, [d["risk"] for d in drugs], cuts)

    print("\n" + "=" * 70)
    print(f"committed (no IKs): train 8/12 (0.67) | blind 9/16 (0.56) | all 17/28 (0.61)")
    print(f"with IKs:  cuts={cuts}")
    for b in (tr, va, al):
        print(f"  {b['name']} (n={b['n']}): acc={b['acc']:.2f} adj={b['adj']:.2f} "
              f"High-sens={b['high_sens']:.2f} High-fp={b['high_fp']} confusion={b['confusion']}")

    # per-drug class comparison: committed vs +IKs
    print("\nper-drug prediction (true -> committed -> +IKs), only where they differ:")
    committed_pred = {d["drug"]: d["ipsc_pred"] for d in load_drugs_committed_pred()}
    iks_pred = {d["drug"]: CLASSES[classify([d["risk"]], cuts)[0]] for d in drugs}
    for d in drugs:
        c_old = committed_pred.get(d["drug"], "?")
        c_new = iks_pred[d["drug"]]
        flag = "" if c_old == c_new else "  <-- CHANGED"
        print(f"  {d['drug']:15s} true={d['cls']:12s} committed={c_old:12s} +IKs={c_new:12s}{flag}")

    # write results
    out = os.path.join(HERE, "cipa_validation_results_iks.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drug", "set", "true_class", "ipsc_risk_iks", "ipsc_pred_iks",
                    "has_iks_ic50", "committed_pred", "changed"])
        for d in drugs:
            c_old = committed_pred.get(d["drug"], "?")
            c_new = iks_pred[d["drug"]]
            w.writerow([d["drug"], d["set"], d["cls"], round(d["risk"], 4), c_new,
                        d["iks_ic50"] is not None, c_old, "CHANGED" if c_old != c_new else ""])

    summary = dict(cuts=cuts,
                   train=dict(n=tr["n"], acc=tr["acc"], confusion=tr["confusion"],
                              high_sens=tr["high_sens"], high_fp=tr["high_fp"]),
                   valid=dict(n=va["n"], acc=va["acc"], confusion=va["confusion"],
                              high_sens=va["high_sens"], high_fp=va["high_fp"]),
                   all28=dict(n=al["n"], acc=al["acc"], confusion=al["confusion"],
                              high_sens=al["high_sens"], high_fp=al["high_fp"]),
                   committed_no_iks=dict(train=8.0/12, valid=9.0/16, all28=17.0/28))
    with open(os.path.join(HERE, "cipa_validation_summary_iks.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nwrote {out} and cipa_validation_summary_iks.json")


def load_drugs_committed_pred():
    """Committed per-drug predictions from the existing results file, for comparison."""
    out = []
    for r in csv.DictReader(open(os.path.join(HERE, "cipa_validation_results.csv"))):
        out.append(dict(drug=r["drug"], ipsc_pred=r["ipsc_pred"]))
    return out


if __name__ == "__main__":
    main()
