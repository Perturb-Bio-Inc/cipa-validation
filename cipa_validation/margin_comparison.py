#!/usr/bin/env python3
"""
Margin comparison: does the multi-channel block beat an hERG-only margin on the
Blinova-2018 repolarization endpoint, and on the Lee-2025 anchored-drug direction?

The classification benchmark already answers this for the three-class risk label
(cipa_validation_summary.json and BENCHMARK_REANALYSIS_2026-09-06.md §2). This script
answers it on the continuous repolarization readout, which is the endpoint the paper
actually reports against Blinova. If the hERG-only model reproduces measured FPD
prolongation as well as the multi-channel model, the extra channels add nothing on this
endpoint, and the paper must say so.

Channel sets:
  multichannel  g_Kr + g_CaL + g_Na, static hERG fit   (the committed model)
  herg_static   g_Kr only, static hERG fit
  herg_dynamic  g_Kr only, dynamic hERG fit

Run: ~/.venvs/myokit/bin/python margin_comparison.py
Outputs: margin_comparison_results.csv, margin_comparison_summary.json
"""
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ta1_causal_prototype"))
from ipsc_cm_ap_model import IPSCModel, extract_biomarkers  # noqa: E402
from cipa_validation_harness import hill_block, repolarization_collapse, CONCS, ABN_RISK  # noqa: E402

REF = os.path.join(HERE, "cipa_28drug_reference.csv")
ABS_THRESH_MS = 5.0
MODES = ["multichannel", "herg_static", "herg_dynamic"]

# Lee et al. 2025 anchored drugs and measured direction (from the study text).
LEE_ANCHORED = [
    ("quinidine", "High", "prolong"),
    ("dofetilide", "High", "prolong"),
    ("ibutilide", "High", "prolong"),
    ("sotalol", "High", "prolong"),
    ("droperidol", "Intermediate", "prolong"),
    ("ranolazine", "Low/no", "prolong"),
    ("nifedipine", "Low/no", "shorten"),
    ("nitrendipine", "Low/no", "shorten"),
    ("diltiazem", "Low/no", "shorten"),
    ("verapamil", "Low/no", "shorten"),
]


def num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def load_drugs():
    drugs = []
    for r in csv.DictReader(open(REF)):
        if r["set"] == "extra":
            continue
        cmax = num(r["cmax_free_nM"])
        if cmax is None:
            continue
        drugs.append(dict(
            drug=r["drug"], cls=r["tdp_class"], cmax=cmax,
            herg_static=num(r["hERG_static_IC50_nM"]) or num(r["hERG_IC50_nM"]),
            herg_static_h=num(r["hERG_static_h"]) or num(r["hERG_h"]) or 1.0,
            herg_dynamic=num(r["hERG_IC50_nM"]),
            herg_dynamic_h=num(r["hERG_h"]) or 1.0,
            ical=num(r["ICaL_IC50_nM"]), ical_h=num(r["ICaL_h"]) or 1.0,
            ina=num(r["INa_peak_IC50_nM"]), ina_h=num(r["INa_peak_h"]) or 1.0,
        ))
    return drugs


def load_blinova():
    import openpyxl
    path = os.path.join(HERE, "source_data", "Blinova_etal_2018_data.xlsx")
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Sheet1"]
    it = ws.iter_rows(values_only=True)
    header = next(it)
    di = {n: i for i, n in enumerate(header)}
    fix = {"d,l sotalol": "sotalol", "d,l,sotalol": "sotalol"}
    meas = {}
    for r in it:
        d = fix.get(str(r[di["Drug_Name"]]).strip().lower(), str(r[di["Drug_Name"]]).strip().lower())
        mult = r[di["conc"]]
        dd = r[di["ddFPDc"]]
        if mult is not None and dd is not None:
            try:
                meas.setdefault((d, int(mult)), []).append(float(dd))
            except (TypeError, ValueError):
                pass
    return meas


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def model_point(drug, mult, base_apd90, base_mdp, mode):
    c = mult * drug["cmax"]
    if mode == "multichannel":
        scales = {"g_Kr": 1.0 - hill_block(c, drug["herg_static"], drug["herg_static_h"])}
        if drug["ical"]:
            scales["g_CaL"] = 1.0 - hill_block(c, drug["ical"], drug["ical_h"])
        if drug["ina"]:
            scales["g_Na"] = 1.0 - hill_block(c, drug["ina"], drug["ina_h"])
    elif mode == "herg_static":
        scales = {"g_Kr": 1.0 - hill_block(c, drug["herg_static"], drug["herg_static_h"])}
    elif mode == "herg_dynamic":
        scales = {"g_Kr": 1.0 - hill_block(c, drug["herg_dynamic"], drug["herg_dynamic_h"])}
    else:
        raise ValueError(mode)
    m = IPSCModel(scales)
    t, v = m.simulate()
    bm = extract_biomarkers(t, v)
    a90 = bm["apd90"]
    if not np.isnan(a90):
        return a90 - base_apd90, "beat", a90, bm["cycle_length"]
    if repolarization_collapse(t, v, base_mdp) or bm["repol_failure"]:
        return ABN_RISK * base_apd90, "collapse", float("nan"), float("nan")
    return 0.0, "quiescent", float("nan"), float("nan")


def run_mode(drugs, meas, mode, base_apd90, base_mdp):
    rows = []
    for d in drugs:
        for mult in CONCS:
            dd = meas.get((d["drug"], mult))
            med = float(np.median(dd)) if dd else None
            mm, state, a90, cl = model_point(d, mult, base_apd90, base_mdp, mode)
            rows.append(dict(drug=d["drug"], cls=d["cls"], mult=mult,
                             model_dapd90_ms=round(mm, 3),
                             model_apd90_ms=(None if np.isnan(a90) else round(a90, 3)),
                             model_cl_ms=(None if np.isnan(cl) else round(cl, 3)),
                             state=state,
                             measured_ms=(None if med is None else round(med, 3))))
    return rows


def metrics(rows):
    pts = [r for r in rows if r["measured_ms"] is not None]
    m = np.array([r["model_dapd90_ms"] for r in pts])
    y = np.array([r["measured_ms"] for r in pts])
    rho = spearman(m, y)
    beat = [(r["model_dapd90_ms"], r["measured_ms"]) for r in pts if r["state"] == "beat"]
    rho_beat = spearman([b[0] for b in beat], [b[1] for b in beat])
    per_dose = {}
    for mult in CONCS:
        sub = [(r["model_dapd90_ms"], r["measured_ms"]) for r in pts if r["mult"] == mult]
        per_dose[int(mult)] = round(spearman([s[0] for s in sub], [s[1] for s in sub]), 4)
    # per-drug mean over doses and per-drug max signed measured response
    by_drug = {}
    for r in pts:
        by_drug.setdefault(r["drug"], []).append((r["model_dapd90_ms"], r["measured_ms"]))
    drug_mean = {d: (float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v])))
                 for d, v in by_drug.items()}
    rho_drug_mean = spearman([v[0] for v in drug_mean.values()], [v[1] for v in drug_mean.values()])
    drug_max = {d: (float(v[int(np.argmax([p[1] for p in v]))][0]),
                    float(np.max([p[1] for p in v]))) for d, v in by_drug.items()}
    rho_drug_max = spearman([v[0] for v in drug_max.values()], [v[1] for v in drug_max.values()])
    # direction
    meas_points = [r for r in pts if abs(r["model_dapd90_ms"]) >= ABS_THRESH_MS
                   and abs(r["measured_ms"]) >= ABS_THRESH_MS]
    dir_ok = sum(1 for r in meas_points if np.sign(r["model_dapd90_ms"]) == np.sign(r["measured_ms"]))
    dir_all = sum(1 for r in pts if np.sign(r["model_dapd90_ms"]) == np.sign(r["measured_ms"]))
    return dict(rho=round(rho, 4), rho_beat=round(rho_beat, 4), n=len(pts), n_beat=len(beat),
                per_dose=per_dose, rho_drug_mean=round(rho_drug_mean, 4),
                rho_drug_max_signed=round(rho_drug_max, 4),
                direction_measurable=f"{dir_ok}/{len(meas_points)}",
                direction_all=f"{dir_all}/{len(pts)}")


def main():
    drugs = load_drugs()
    assert len(drugs) == 28
    meas = load_blinova()
    base = IPSCModel().biomarkers()
    base_apd90, base_mdp = base["apd90"], base["mdp"]
    print(f"baseline APD90={base_apd90:.2f} ms\n")

    all_rows, summary = [], {}
    for mode in MODES:
        print(f"running {mode} ...", flush=True)
        rows = run_mode(drugs, meas, mode, base_apd90, base_mdp)
        for r in rows:
            r["mode"] = mode
        all_rows.extend(rows)
        summary[mode] = metrics(rows)
        print(f"  {summary[mode]}")

    # Lee anchored direction per mode: sign of model prolongation at the drug's
    # CiPA-reference-dose exposure is not in this file, so use the sign over 1-4x.
    lee = {}
    for mode in MODES:
        agree = []
        for drug, cls, direction in LEE_ANCHORED:
            dm = [r for r in all_rows if r["mode"] == mode and r["drug"] == drug]
            model_dir = "prolong" if np.mean([r["model_dapd90_ms"] for r in dm]) > 0 else "shorten"
            agree.append((drug, direction, model_dir, direction == model_dir))
        lee[mode] = dict(match=sum(1 for a in agree if a[3]), total=len(agree),
                         detail=[dict(drug=a[0], lee=a[1], model=a[2], agree=a[3]) for a in agree])
    summary["lee_anchored_direction"] = lee

    with open(os.path.join(HERE, "margin_comparison_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mode", "drug", "cls", "mult", "model_dapd90_ms",
                                           "model_apd90_ms", "model_cl_ms", "state", "measured_ms"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    json.dump(summary, open(os.path.join(HERE, "margin_comparison_summary.json"), "w"), indent=2)

    print("\n" + "=" * 78)
    print("MARGIN COMPARISON SUMMARY")
    print("=" * 78)
    for mode in MODES:
        s = summary[mode]
        print(f"{mode:14s} rho={s['rho']:+.4f}  beat={s['rho_beat']:+.4f}  "
              f"per-dose={s['per_dose']}  drugmax={s['rho_drug_max_signed']:+.4f}  "
              f"dir={s['direction_measurable']}")
    print("\nLee anchored direction (mean sign over 1-4x):")
    for mode in MODES:
        print(f"  {mode:14s} {lee[mode]['match']}/{lee[mode]['total']}")
    print("\nwrote margin_comparison_results.csv, margin_comparison_summary.json")


if __name__ == "__main__":
    main()
