"""Collapse-detector calibration against the Blinova-2018 measured EAD labels.

Replaces the binary model collapse test (mean V > base_mdp+20, or repol_failure
fraction > 0.5) with a continuous severity = max over 1-4x of the fraction of the
trace spent depolarized above -20 mV. Calibrates a threshold on that severity
against the measured early-afterdepolarization (EAD) rate per drug.

Honest scope: model collapse (repolarization failure) is a proxy; Blinova EAD is
the measured pro-arrhythmic endpoint. This tunes the proxy to the measured endpoint,
the same family as the risk-label validation, at the collapse level.
"""
import csv, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TA1 = os.path.join(HERE, "..", "ta1_causal_prototype")
for p in (HERE, TA1):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from openpyxl import load_workbook
from scipy.stats import spearmanr

from cipa_validation_harness import load_drugs, hill_block, CONCS, ABN_RISK, repolarization_collapse
from ipsc_cm_ap_model import IPSCModel, extract_biomarkers
from blinova_validation import NAME_FIX, load_blinova

XLSX = os.path.join(HERE, "source_data", "Blinova_etal_2018_data.xlsx")


def ead_rate_per_drug():
    wb = load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    rows = ws.iter_rows(min_row=2, values_only=True)
    count = {}
    total = {}
    for r in rows:
        if r is None or r[0] is None:
            continue
        drug = NAME_FIX.get(r[0].lower().strip(), r[0].lower().strip())
        ead = r[6]  # EAD column
        if ead is None:
            continue
        total[drug] = total.get(drug, 0) + 1
        if int(ead) == 1:
            count[drug] = count.get(drug, 0) + 1
    return {d: (count.get(d, 0), total.get(d, 0), count.get(d, 0) / total[d]) for d in total}


def model_severity(drug, base_apd90, base_mdp):
    """Max over 1-4x of the mean-V-to-MDP margin (mV), the continuous quantity the
    committed collapse detector thresholds. Binary collapse = the committed harness
    decision (mean V > base_mdp+20, i.e. margin>20, or repol_failure). Also keeps the
    fraction-above--20mV as a secondary severity."""
    max_sev = None
    collapsed = False
    worst_mult = None
    margin_by_mult = {}
    frac_by_mult = {}
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
        mean_v = float(np.mean(v))
        margin = mean_v - base_mdp
        frac = float(np.mean(v > -20.0))
        margin_by_mult[mult] = round(margin, 4)
        frac_by_mult[mult] = round(frac, 4)
        if not np.isnan(bm["apd90"]):
            state = "beat"
        elif repolarization_collapse(t, v, base_mdp) or bm["repol_failure"]:
            state = "collapse"
            collapsed = True
        else:
            state = "quiescent"
        if max_sev is None or margin > max_sev:
            max_sev = margin
            worst_mult = mult
    return max_sev, collapsed, worst_mult, margin_by_mult, frac_by_mult


def roc(sev, y):
    """ROC over severity thresholds. Returns list of (thr, sens, spec, youden) and AUC."""
    uniq = sorted(set(sev))
    pts = []
    for thr in sorted(set([0.0, 20.0] + uniq)):
        pred = [1 if s >= thr else 0 for s in sev]
        tp = sum(1 for p, yy in zip(pred, y) if p and yy)
        tn = sum(1 for p, yy in zip(pred, y) if not p and not yy)
        fp = sum(1 for p, yy in zip(pred, y) if p and not yy)
        fn = sum(1 for p, yy in zip(pred, y) if not p and yy)
        sens = tp / (tp + fn) if (tp + fn) else 1.0
        spec = tn / (tn + fp) if (tn + fp) else 1.0
        pts.append((thr, sens, spec, sens + spec - 1))
    # AUC via Mann-Whitney U on the score
    pos = [s for s, yy in zip(sev, y) if yy]
    neg = [s for s, yy in zip(sev, y) if not yy]
    auc = None
    if pos and neg:
        auc = (sum(1 for p in pos for n in neg if p > n) +
               0.5 * sum(1 for p in pos for n in neg if p == n)) / (len(pos) * len(neg))
    return pts, auc


def main():
    base = IPSCModel()
    bb = base.biomarkers()
    base_apd90 = bb["apd90"]
    base_mdp = bb["mdp"]

    drugs = load_drugs()
    meas, any_ead = load_blinova()
    erate = ead_rate_per_drug()

    rows = []
    for d in drugs:
        margin, collapsed, worst_mult, margin_by, frac_by = model_severity(d, base_apd90, base_mdp)
        ec, et, er = erate.get(d["drug"], (0, 0, 0.0))
        rows.append(dict(
            drug=d["drug"], set=d["set"], true_class=d["cls"],
            severity=round(margin, 4), collapsed=collapsed, worst_mult=worst_mult,
            margin_by_mult=json.dumps(margin_by), frac_by_mult=json.dumps(frac_by),
            ead_count=ec, ead_total=et, ead_rate=round(er, 4),
            any_ead=any_ead.get(d["drug"], 0),
        ))
    rows.sort(key=lambda r: r["drug"])

    sev = [r["severity"] for r in rows]
    ye = [r["any_ead"] for r in rows]
    yr = [r["ead_rate"] for r in rows]

    rho_rate, p_rate = spearmanr(sev, yr)
    pts, auc = roc(sev, ye)

    youden = max(pts, key=lambda p: p[3])
    at90 = min([p for p in pts if p[2] >= 0.90], key=lambda p: -p[1])
    # committed detector = binary `collapsed` flag (margin>20 or repol_failure)
    cc = [int(r["collapsed"]) for r in rows]
    tp = sum(1 for p, y in zip(cc, ye) if p and y)
    fp = sum(1 for p, y in zip(cc, ye) if p and not y)
    fn = sum(1 for p, y in zip(cc, ye) if not p and y)
    cur_sens = tp / (tp + fn) if (tp + fn) else 1.0
    cur_spec = 1.0 - (fp / (len(ye) - sum(ye)) if (len(ye) - sum(ye)) else 0.0)

    summary = {
        "n_drugs": len(rows),
        "n_ead": sum(ye), "n_noead": len(ye) - sum(ye),
        "baseline_apd90": base_apd90, "baseline_mdp": base_mdp,
        "severity_ead_spearman_rho": rho_rate, "p": p_rate,
        "auc_anyead": auc,
        "current_binary_collapse": {
            "n_collapse": sum(cc), "sens": cur_sens, "spec": cur_spec,
        },
        "youden_operating_point": {"threshold": youden[0], "sens": youden[1], "spec": youden[2], "j": youden[3]},
        "spec90_operating_point": {"threshold": at90[0], "sens": at90[1], "spec": at90[2]},
    }

    with open(os.path.join(HERE, "collapse_calibration_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "collapse_calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"baseline APD90 {base_apd90:.2f}  MDP {base_mdp:.2f}")
    print(f"EAD drugs {sum(ye)}/28, no-EAD {len(ye)-sum(ye)}/28")
    print(f"Spearman(model severity, measured EAD rate) = {rho_rate:.4f}  (p={p_rate:.4f})")
    print(f"AUC (severity vs any-EAD) = {auc:.3f}")
    print(f"committed binary collapse: n_collapse {sum(cc)} sens {cur_sens:.3f} spec {cur_spec:.3f}")
    print(f"Youden optimum: threshold {youden[0]:.3f} sens {youden[1]:.3f} spec {youden[2]:.3f} J {youden[3]:.3f}")
    print(f"spec-90% floor: threshold {at90[0]:.3f} sens {at90[1]:.3f} spec {at90[2]:.3f}")
    print("\nper-drug (severity, collapsed, ead_rate, any_ead):")
    for r in rows:
        print(f"  {r['drug']:<14} sev {r['severity']:.3f} coll {int(r['collapsed'])} "
              f"ead {r['ead_rate']:.3f} ({r['ead_count']}/{r['ead_total']}) any {int(r['any_ead'])}")


if __name__ == "__main__":
    main()
