#!/usr/bin/env python3
"""
Blinova-2018 validation: the model's repolarization response vs a real, measured
iPSC-CM dataset. Task (from LIMITATIONS_PROGRAM_2026-09-06, item V).

Question: the mechanism ranking (the product) has never been tested against a wet
recording. The CiPA-28 benchmark measures only a three-class risk label. Blinova et
al. 2018 (Cell Rep 24(13):3582-3592, doi:10.1016/j.celrep.2018.08.079) is the CiPA myocyte validation study: a
multi-site MEA field-potential-duration (FPD) screen of the 28 CiPA drugs at 1-4x free
Cmax, with EAD annotations. Does the Kernik-2019 iPSC-CM AP model, run with the SAME
multi-channel Hill block and the SAME 1-4x free-Cmax exposure, reproduce the measured
direction and magnitude of repolarization prolongation?

Metric is FPD (extracellular) against the model's APD90 (patch-clamp action potential).
They are different readouts; the honest claim is direction and rank/magnitude agreement
of repolarization prolongation, not waveform equivalence. This is stated in the output.

Design:
  1. Load Blinova data (source_data/Blinova_etal_2018_data.xlsx). conc is the multiple
     of free Cmax (1,2,3,4), identical to the committed harness's CONCS.
  2. Aggregate measured ddFPDc (ms) per drug x conc as the MEDIAN across all sites,
     cell types and platforms (CiPA protocol aggregates across sites).
  3. For each of the 28 drugs, run the model at 1-4x free Cmax with the identical
     multi-channel Hill block (g_Kr, g_CaL, g_Na) and record the ABSOLUTE APD90
     prolongation in ms (a90 - baseline), plus repolarization-collapse flags.
  4. Compare at matched (drug, multiple) points.

Metrics (honest, bounded):
  - Direction agreement: fraction of measurable points where sign(model) == sign(meas).
  - Spearman rank correlation rho between model dAPD90 and median ddFPDc.
  - Collapse/EAD concordance: does a model repolarization collapse (risk>=ABN_RISK)
    co-occur with Blinova EADs, and vice versa.

Run: ~/.venvs/myokit/bin/python blinova_validation.py
Outputs: blinova_validation_results.csv, blinova_validation_summary.json, report to stdout.
"""
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ta1_causal_prototype"))
from ipsc_cm_ap_model import IPSCModel, extract_biomarkers  # noqa: E402

# reuse the committed harness's block/risk primitives so the comparison is consistent
from cipa_validation_harness import hill_block, repolarization_collapse, CONCS, ABN_RISK  # noqa: E402

REF = os.path.join(HERE, "cipa_28drug_reference.csv")
BLINOVA = os.path.join(HERE, "source_data", "Blinova_etal_2018_data.xlsx")
ABSOLUTE_THRESH_MS = 5.0   # points with |model| and |meas| both above this count as "measurable"

# Blinova drug-name -> canonical reference drug name (lowercase)
NAME_FIX = {"d,l sotalol": "sotalol", "d,l,sotalol": "sotalol"}


def num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def load_reference():
    """Load the canonical 28 with free Cmax and the three common-panel IC50s."""
    drugs = []
    for r in csv.DictReader(open(REF)):
        if r["set"] == "extra":
            continue
        cmax = num(r["cmax_free_nM"])
        if cmax is None:
            continue
        herg = num(r["hERG_static_IC50_nM"]) or num(r["hERG_IC50_nM"])
        drugs.append(dict(
            drug=r["drug"], cls=r["tdp_class"], cmax=cmax,
            herg_ic50=herg, herg_h=num(r["hERG_static_h"]) or num(r["hERG_h"]) or 1.0,
            ical_ic50=num(r["ICaL_IC50_nM"]), ical_h=num(r["ICaL_h"]) or 1.0,
            ina_ic50=num(r["INa_peak_IC50_nM"]), ina_h=num(r["INa_peak_h"]) or 1.0,
        ))
    return drugs


def load_blinova():
    """Return {(drug, mult): [ddFPDc ms, ...]} across all sites/celltypes/platforms."""
    try:
        import openpyxl
    except ImportError:
        sys.exit("openpyxl required; install into ~/.venvs/myokit")
    wb = openpyxl.load_workbook(BLINOVA, read_only=True)
    ws = wb["Sheet1"]
    it = ws.iter_rows(values_only=True)
    header = next(it)
    di = {n: i for i, n in enumerate(header)}
    meas = {}
    ead_by_drug = {}
    for r in it:
        dname = str(r[di["Drug_Name"]]).strip().lower()
        drug = NAME_FIX.get(dname, dname)
        mult = r[di["conc"]]
        dd = r[di["ddFPDc"]]
        if mult is not None and dd is not None:
            try:
                dd = float(dd)
            except (TypeError, ValueError):
                continue
            meas.setdefault((drug, int(mult)), []).append(dd)
        ead = r[di["EAD"]]
        ead_by_drug.setdefault(drug, 0)
        if ead is not None:
            try:
                if float(ead) != 0:
                    ead_by_drug[drug] = max(ead_by_drug[drug], 1)
            except (TypeError, ValueError):
                if str(ead).strip() not in ("", "0", "none", "no"):
                    ead_by_drug[drug] = max(ead_by_drug[drug], 1)
    return meas, ead_by_drug


def _spearman(x, y):
    """Spearman rank correlation between two 1-D arrays.

    Prefer scipy's spearmanr (standard tie handling); fall back to an
    average-rank estimator so the module runs without scipy installed.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or len(y) != len(x):
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except ImportError:
        pass
    rx = np.argsort(np.argsort(x), kind="stable") + 1.0
    ry = np.argsort(np.argsort(y), kind="stable") + 1.0
    return float(np.corrcoef(rx, ry)[0, 1])


def model_dapd90(drug, base_apd90, base_mdp):
    """Model absolute APD90 prolongation (ms) at each of 1-4x free Cmax, plus collapse."""
    out = {}
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
            d = a90 - base_apd90
            state = "beat"
        elif repolarization_collapse(t, v, base_mdp) or bm["repol_failure"]:
            d = ABN_RISK * base_apd90   # map collapse onto a large ms prolongation
            state = "collapse"
        else:
            d = 0.0
            state = "quiescent"
        out[mult] = dict(dapd90_ms=round(float(d), 2), state=state)
    return out


def main():
    drugs = load_reference()
    assert len(drugs) == 28, f"expected 28, got {len(drugs)}"
    meas, ead_by_drug = load_blinova()

    base = IPSCModel().biomarkers()
    base_apd90, base_mdp = base["apd90"], base["mdp"]
    print(f"baseline Kernik-2019: APD90={base_apd90:.2f} ms, MDP={base_mdp:.1f} mV\n")

    rows = []
    paired = []          # (model dAPD90 ms, measured median ddFPDc ms, model_state)
    dir_ok, dir_n = 0, 0
    coll_model, coll_blin = [], []
    for d in drugs:
        mod = model_dapd90(d, base_apd90, base_mdp)
        for mult in CONCS:
            key = (d["drug"], mult)
            dd_list = meas.get(key)
            med = float(np.median(dd_list)) if dd_list else None
            n = len(dd_list) if dd_list else 0
            mm = mod[mult]
            rows.append(dict(
                drug=d["drug"], cls=d["cls"], mult=mult,
                model_dapd90_ms=mm["dapd90_ms"], model_state=mm["state"],
                blinova_median_ms=(None if med is None else round(med, 3)),
                blinova_n=n,
            ))
            if med is not None:
                paired.append((mm["dapd90_ms"], med, mm["state"]))
                # direction agreement on measurable points
                if abs(mm["dapd90_ms"]) >= ABSOLUTE_THRESH_MS and abs(med) >= ABSOLUTE_THRESH_MS:
                    dir_n += 1
                    if np.sign(mm["dapd90_ms"]) == np.sign(med):
                        dir_ok += 1
        # collapse / EAD concordance at drug level
        mod_coll = any(mm["state"] == "collapse" for mm in mod.values())
        blin_e = ead_by_drug.get(d["drug"], 0)
        coll_model.append((d["drug"], mod_coll))
        coll_blin.append((d["drug"], blin_e))

    # ---- metrics ----
    pair = np.array([[p[0], p[1]] for p in paired])
    rho = _spearman(pair[:, 0], pair[:, 1]) if len(pair) >= 3 else float("nan")
    # beat-only rho: drop collapse points, which carry an arbitrary large value (ABN_RISK*base)
    beat = np.array([[p[0], p[1]] for p in paired if p[2] == "beat"])
    rho_beat = _spearman(beat[:, 0], beat[:, 1]) if len(beat) >= 3 else float("nan")
    dir_frac = dir_ok / dir_n if dir_n else float("nan")

    # collapse<->EAD 2x2 at drug level; reframe as specificity/sensitivity of collapse
    cm = {d: c for d, c in coll_model}
    cb = {d: c for d, c in coll_blin}
    tab = {"model_collapse & blinova_ead": 0, "model_collapse & blinova_no_ead": 0,
           "no_model_collapse & blinova_ead": 0, "no_model_collapse & blinova_no_ead": 0}
    for d in cm:
        k = ("model_collapse" if cm[d] else "no_model_collapse") + " & " + \
            ("blinova_ead" if cb[d] else "blinova_no_ead")
        tab[k] += 1
    collapse_specificity = tab["model_collapse & blinova_no_ead"] == 0  # no collapse without EAD
    n_ead_drugs = tab["model_collapse & blinova_ead"] + tab["no_model_collapse & blinova_ead"]
    collapse_sensitivity = (tab["model_collapse & blinova_ead"] / n_ead_drugs
                            if n_ead_drugs else float("nan"))

    summary = dict(
        baseline_apd90_ms=round(base_apd90, 2),
        n_drugs=len(drugs),
        n_paired_points=len(pair),
        spearman_rho=round(rho, 4),
        spearman_rho_beat_only=round(rho_beat, 4),
        direction_agreement=dict(match=dir_ok, measurable=dir_n, frac=round(dir_frac, 4)),
        collapse_ead_2x2=tab,
        collapse_specificity_all_match=bool(collapse_specificity),
        collapse_sensitivity_frac=round(collapse_sensitivity, 4),
        note="FPD (measured) vs APD90 (model): direction + rank agreement of repolarization "
             "prolongation; not waveform equivalence. Model over-prolongs absolute ms.",
    )

    # ---- write results ----
    with open(os.path.join(HERE, "blinova_validation_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(summary, open(os.path.join(HERE, "blinova_validation_summary.json"), "w"),
              indent=2)

    # ---- print report ----
    print("=" * 78)
    print("BLINOVA-2018 VALIDATION: model dAPD90 (ms) vs measured median ddFPDc (ms)")
    print("=" * 78)
    print(f"{'drug':15s} {'cls':6s} {'x':>2s} {'model ms':>9s} {'state':9s} {'meas ms':>9s} {'n':>3s}")
    for r in rows:
        m = "nan" if r["blinova_median_ms"] is None else f"{r['blinova_median_ms']:7.1f}"
        nn = "0" if r["blinova_n"] == 0 else str(r["blinova_n"])
        print(f"{r['drug']:15s} {r['cls']:6s} {r['mult']:>2.0f} {r['model_dapd90_ms']:>9.1f} "
              f"{r['model_state']:9s} {m:>9s} {nn:>3s}")
    print("\nmetrics:")
    print(f"  Spearman rho (model dAPD90 vs measured ddFPDc): {rho:.4f}  (n={len(pair)})")
    print(f"  Spearman rho, beat-only (no collapse convention): {rho_beat:.4f}  (n={len(beat)})")
    print(f"  direction agreement: {dir_ok}/{dir_n} = {dir_frac:.3f}  (both |.|>={ABSOLUTE_THRESH_MS} ms)")
    print(f"  collapse<->EAD 2x2 (per drug): {tab}")
    print(f"  collapse specificity (no collapse without measured EAD): {collapse_specificity}")
    print(f"  collapse sensitivity (EAD drugs the model flags): {collapse_sensitivity:.3f}")
    print(f"\nwrote blinova_validation_results.csv, blinova_validation_summary.json")


if __name__ == "__main__":
    main()
