#!/usr/bin/env python3
"""
I2 (completion) — is the population P(class|data) a CALIBRATED confidence signal?

The I6 population build produced, per drug, P(Low|Intermed|High) as the fraction of N=10
conductance-draw myocytes whose frozen-score risk lands in each band. I6 explicitly deferred the
calibration question ("Not established: a calibrated Bayesian posterior over the score"). This is
that deferred piece: does the population's own confidence (max class probability) predict whether
the call is correct? If confidence is calibrated, drugs the population is sure about are more often
right than drugs it is split on. If it is not, a high confidence is not a trustworthy label.

Reads ONLY committed outputs: population_risk_summary_{correlated,independent}.json and
cipa_validation_results.csv. No re-fit, no model re-run.

Outputs: i2_calibration_summary.json and a printed reliability table.
"""
import os, sys, json, csv

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rows(mode):
    s = json.load(open(os.path.join(HERE, f"population_risk_summary_{mode}.json")))
    return s, s["per_drug"]


def confidence(d):
    """max class probability and the argmax class."""
    probs = [d["p_low"], d["p_interm"], d["p_high"]]
    names = ["Low", "Intermediate", "High"]
    i = max(range(3), key=lambda k: probs[k])
    return probs[i], names[i]


def reliability(mode):
    s, rows = load_rows(mode)
    recs = []
    for d in rows:
        conf, arg = confidence(d)
        recs.append({
            "drug": d["drug"], "true": d["true_class"],
            "p_low": d["p_low"], "p_interm": d["p_interm"], "p_high": d["p_high"],
            "conf": round(conf, 3), "argmax": arg,
            "arg_correct": arg == d["true_class"],
            "det_class": d["det_class"], "det_correct": d["det_correct"],
        })
    n = len(recs)
    n_correct = sum(r["arg_correct"] for r in recs)
    # point-biserial correlation between confidence and arg-correct
    import numpy as np
    conf = np.array([r["conf"] for r in recs])
    corr = np.array([float(r["arg_correct"]) for r in recs])
    pb = np.corrcoef(conf, corr)[0, 1] if conf.std() > 0 else float("nan")
    # binned reliability: mean confidence vs empirical accuracy per bin
    bands = [(0.0, 0.7), (0.7, 0.9), (0.9, 1.0001)]
    bins = []
    for lo, hi in bands:
        sel = [r for r in recs if lo <= r["conf"] < hi]
        if not sel:
            continue
        bins.append({
            "band": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
            "mean_conf": round(float(np.mean([r["conf"] for r in sel])), 3),
            "accuracy": round(sum(r["arg_correct"] for r in sel) / len(sel), 3),
        })
    # confidently-wrong set: confidence >= 0.9 but the argmax call is wrong
    conf_wrong = sorted(
        [r["drug"] for r in recs if r["conf"] >= 0.9 and not r["arg_correct"]])
    # split set: max class probability < 0.9 (no clear winner)
    split = sorted([r["drug"] for r in recs if r["conf"] < 0.9])
    return {
        "mode": mode,
        "n": n,
        "accuracy_det": s["accuracy_det"],
        "accuracy_pop_argmax": s["accuracy_pop_argmax"],
        "accuracy_argmax": round(n_correct / n, 3),
        "point_biserial_conf_correct": round(float(pb), 3),
        "bins": bins,
        "confident_wrong_drugs": conf_wrong,
        "split_drugs": split,
    }


def main():
    out = {"notes": (
        "I2 completion. Population P(class|data) from committed I6 (N=10 conductance draws, "
        "symmetric log-normal, frozen cuts). 'argmax' = the class the population most members "
        "land in; 'conf' = that class's probability. A calibrated signal would show accuracy "
        "rising with conf. Reads committed outputs only."),
        "per_mode": {}, "summary": {}}
    for mode in ["correlated", "independent"]:
        out["per_mode"][mode] = reliability(mode)
    ind = out["per_mode"]["independent"]
    cor = out["per_mode"]["correlated"]
    out["summary"] = {
        "verdict": "the population P(class|data) is severely overconfident: not a calibrated posterior",
        "independent_reliability": {
            "accuracy_by_confidence_band": [b["accuracy"] for b in ind["bins"]],
            "point_biserial_conf_vs_correct": ind["point_biserial_conf_correct"],
            "accuracy_at_max_confidence_band": ind["bins"][-1]["accuracy"],
            "mean_conf_at_max_band": ind["bins"][-1]["mean_conf"],
        },
        "why": (
            f"Independent mode: accuracy does rise with confidence "
            f"({[b['accuracy'] for b in ind['bins']]}), so the signal is weakly monotonic, but at "
            f"the highest-confidence band (mean conf {ind['bins'][-1]['mean_conf']}) accuracy is "
            f"only {ind['bins'][-1]['accuracy']}, far below the probability it claims. The "
            f"confident-but-wrong set ({len(ind['confident_wrong_drugs'])} drugs at conf>=0.9) is "
            f"the systematic low/no-risk overcall (metoprolol, mexiletine, loratadine, tamoxifen, "
            f"nifedipine, nitrendipine) plus sotalol. The overconfidence concentrates on the "
            f"structural bias I1/I4 already name."),
        "det_vs_pop_argmax": (
            "population argmax does not beat the deterministic call (det "
            f"{cor['accuracy_det']} vs pop {ind['accuracy_pop_argmax']}), consistent with I6"),
    }
    with open(os.path.join(HERE, "i2_calibration_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    for mode, r in out["per_mode"].items():
        print(f"\n== {mode} ==")
        print(f"  det accuracy {r['accuracy_det']} | pop argmax {r['accuracy_pop_argmax']} "
              f"| argmax-of-rows {r['accuracy_argmax']}")
        print(f"  point-biserial conf vs correctness: {r['point_biserial_conf_correct']}")
        for b in r["bins"]:
            print(f"  band {b['band']}: n={b['n']} mean_conf={b['mean_conf']} "
                  f"accuracy={b['accuracy']}")
        print(f"  confident-but-wrong (conf>=0.9, wrong): {r['confident_wrong_drugs']}")
        print(f"  split (no class >= 0.9): {r['split_drugs']}")
    print("\nverdict:", out["summary"]["verdict"])


if __name__ == "__main__":
    main()
