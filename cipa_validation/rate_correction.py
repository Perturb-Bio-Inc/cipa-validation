#!/usr/bin/env python3
"""
Fridericia rate correction of the model APD90, to match the measured FPDc.

Blinova et al. (2018) and Lee et al. (2025) report Fridericia-corrected field
potential duration (FPDc). The model is a spontaneously-beating iPSC-CM whose cycle
length changes with drug, so its uncorrected APD90 prolongation is not the quantity
the measurement reports. This script applies the same correction to the model:

    APD90c = APD90 / (CL_s)^{1/3},   CL_s = cycle length in seconds

and recomputes every Blinova statistic on the corrected delta. It reads the raw
APD90 and cycle length emitted by margin_comparison.py, so run that first.

Run: ~/.venvs/myokit/bin/python rate_correction.py
Outputs: rate_correction_summary.json
"""
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ta1_causal_prototype"))
from ipsc_cm_ap_model import IPSCModel  # noqa: E402
from cipa_validation_harness import ABN_RISK  # noqa: E402

ABS_THRESH_MS = 5.0
MODES = ["multichannel", "herg_static", "herg_dynamic"]


def fridericia(apd90, cl_ms):
    return apd90 / (cl_ms / 1000.0) ** (1.0 / 3.0)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def load_rows():
    rows = []
    with open(os.path.join(HERE, "margin_comparison_results.csv")) as fh:
        for r in csv.DictReader(fh):
            for k in ("model_dapd90_ms", "model_apd90_ms", "model_cl_ms", "measured_ms"):
                r[k] = None if r[k] in ("", "None") else float(r[k])
            r["mult"] = int(float(r["mult"]))
            rows.append(r)
    return rows


def metrics(rows, base_c):
    pts = [r for r in rows if r["measured_ms"] is not None]
    m = np.array([r["delta_c"] for r in pts])
    y = np.array([r["measured_ms"] for r in pts])
    beat = [(r["delta_c"], r["measured_ms"]) for r in pts if r["state"] == "beat"]
    per_dose = {}
    for mult in sorted({r["mult"] for r in pts}):
        sub = [(r["delta_c"], r["measured_ms"]) for r in pts if r["mult"] == mult]
        per_dose[mult] = round(spearman([s[0] for s in sub], [s[1] for s in sub]), 4)
    by_drug = {}
    for r in pts:
        by_drug.setdefault(r["drug"], []).append((r["delta_c"], r["measured_ms"]))
    drug_mean = [(float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v])))
                 for v in by_drug.values()]
    drug_max = [(float(v[int(np.argmax([p[1] for p in v]))][0]),
                 float(np.max([p[1] for p in v]))) for v in by_drug.values()]
    measurable = [r for r in pts if abs(r["delta_c"]) >= ABS_THRESH_MS
                  and abs(r["measured_ms"]) >= ABS_THRESH_MS]
    dir_ok = sum(1 for r in measurable if np.sign(r["delta_c"]) == np.sign(r["measured_ms"]))
    ms_only = [r for r in pts if abs(r["measured_ms"]) >= ABS_THRESH_MS]
    dir_ms = sum(1 for r in ms_only if np.sign(r["delta_c"]) == np.sign(r["measured_ms"]))
    base_pos = sum(1 for r in ms_only if r["measured_ms"] > 0)
    return dict(
        n_measured_side=len(ms_only),
        always_prolong_measured_side=f"{base_pos}/{len(ms_only)}",
        direction_measured_side=f"{dir_ms}/{len(ms_only)}",
        rho=round(spearman(m, y), 4),
        rho_beat=round(spearman([b[0] for b in beat], [b[1] for b in beat]), 4),
        per_dose=per_dose,
        rho_drug_mean=round(spearman([d[0] for d in drug_mean], [d[1] for d in drug_mean]), 4),
        rho_drug_max_signed=round(spearman([d[0] for d in drug_max], [d[1] for d in drug_max]), 4),
        direction_measurable=f"{dir_ok}/{len(measurable)}",
    )


def main():
    base = IPSCModel().biomarkers()
    base_cl = base["cycle_length"]
    base_c = fridericia(base["apd90"], base_cl)
    print(f"baseline APD90={base['apd90']:.2f} ms  CL={base_cl:.2f} ms  APD90c={base_c:.2f} ms")

    rows = load_rows()
    for r in rows:
        if r["state"] == "collapse":
            r["delta_c"] = ABN_RISK * base_c
        elif r["model_apd90_ms"] is None or r["model_cl_ms"] is None:
            r["delta_c"] = 0.0
        else:
            r["delta_c"] = fridericia(r["model_apd90_ms"], r["model_cl_ms"]) - base_c

    summary = {}
    for mode in MODES:
        sub = [r for r in rows if r["mode"] == mode]
        summary[mode] = metrics(sub, base_c)
        print(f"{mode:14s} {summary[mode]}")

    raw_cl = {}
    for mode in MODES:
        sub = [r for r in rows if r["mode"] == mode and r["state"] == "beat"]
        raw_cl[mode] = {
            "cl_min": round(min(r["model_cl_ms"] for r in sub), 1),
            "cl_max": round(max(r["model_cl_ms"] for r in sub), 1),
        }
    summary["baseline"] = dict(apd90_ms=round(base["apd90"], 3), cl_ms=round(base_cl, 3),
                               apd90c_ms=round(base_c, 3))
    summary["cycle_length_range_beat"] = raw_cl
    json.dump(summary, open(os.path.join(HERE, "rate_correction_summary.json"), "w"), indent=2)
    print("\nwrote rate_correction_summary.json")


if __name__ == "__main__":
    main()
