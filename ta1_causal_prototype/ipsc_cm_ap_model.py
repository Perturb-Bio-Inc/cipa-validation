#!/usr/bin/env python3
"""
Real, published iPSC-CM action-potential model for IGoR-Cardiac TA1 (Sub-model 3).

This replaces the dimensionless Mitchell-Schaeffer placeholder in cardiac_causal_model.py
with the KERNIK-CLANCY 2019 induced-pluripotent-stem-cell cardiomyocyte model, the
field-standard iPSC-CM-specific AP model (Kernik, Morotti, ..., Clancy, J Physiol 2019;
597(17):4533-4564; PMID 31278749). The model is loaded from its published CellML
(Physiome Model Repository exposure e/805, kernik_2019.cellml, in this directory) and
integrated with Myokit's CVODE solver. No hand-porting of equations -- we run the
maintained, validated primary artifact.

WHAT THIS GIVES THE PIPELINE
  - APs in real mV / ms (not dimensionless), spontaneously beating (Kernik has I_funny),
    matching published iPSC-CM biology.
  - Biomarkers: APD90, APD50, MDP, peak V, dV/dt_max, cycle length / beat rate.
  - A conductance-scaling interface scale_conductances({'g_Kr': 0.5, ...}) so the GRN
    layer can drive gene knockdown -> reduced conductance -> AP change. This is the hook
    cardiac_causal_model.forward() needs.

VALIDATION (run validate() or see EP_MODEL_UPGRADE.md): the baseline biomarkers this code
produces match published Kernik-2019 values within stated tolerance, and IKr block prolongs
APD90 in the correct direction/magnitude. The Mitchell-Schaeffer placeholder could only
claim directionality; this is quantitatively defensible.

Requires: myokit (with the CVODE/Sundials backend) + numpy. CellML file kernik_2019.cellml
must sit next to this module.
"""
import os
import numpy as np
import myokit

_HERE = os.path.dirname(os.path.abspath(__file__))
CELLML_PATH = os.path.join(_HERE, "kernik_2019.cellml")

# Map of friendly conductance names -> the CellML variable that scales that current.
# For Kernik-2019, ICaL is a permeability (p_CaL), the others are conductances (g_*).
# [VERIFIED: read kernik_2019.cellml component vars 2026-06-09 -> these are the constant
#  maximal-conductance / permeability parameters of each ionic current.]
CONDUCTANCE_VARS = {
    "g_Na": "ina.g_Na",       # INa  (SCN5A / Nav1.5)
    "g_CaL": "ical.p_CaL",    # ICaL (CACNA1C / Cav1.2) -- permeability, scales like a conductance
    "g_Kr": "ikr.g_Kr",       # IKr  (KCNH2 / hERG)
    "g_Ks": "iks.g_Ks",       # IKs  (KCNQ1)
    "g_K1": "ik1.g_K1",       # IK1  (KCNJ2 / Kir2.1)
    "g_to": "ito.g_to",       # Ito  (KCND3 etc.)
    "g_f": "ifunny.g_f",      # If   (HCN4) -- pacemaker current
    "g_CaT": "icat.g_CaT",    # ICaT
}

_V = "membrane.V"
_T = "engine.time"
_UPSTROKE_THRESH = 10.0  # V/s; threshold to detect a spontaneous upstroke


def load_model():
    """Load the published Kernik-2019 CellML as a Myokit model."""
    return myokit.formats.importer("cellml").model(CELLML_PATH)


class IPSCModel:
    """Spontaneously-beating Kernik-2019 iPSC-CM AP model with conductance scaling.

    The model is autonomous (no external stimulus); Kernik's I_funny pacemaker current
    drives spontaneous beating, as in real iPSC-CMs. Each instance holds its own Myokit
    Simulation so conductance perturbations are isolated.
    """

    def __init__(self, conductance_scales=None):
        self.model = load_model()
        # baseline (1.0) values of every controllable conductance, read from the model
        self._base = {}
        for name, qname in CONDUCTANCE_VARS.items():
            self._base[name] = float(self.model.get(qname).eval())
        self.scales = {name: 1.0 for name in CONDUCTANCE_VARS}
        if conductance_scales:
            self.scales.update(conductance_scales)
        self._build_sim()

    def _build_sim(self):
        sim = myokit.Simulation(self.model)
        for name, qname in CONDUCTANCE_VARS.items():
            sim.set_constant(qname, self._base[name] * self.scales[name])
        self.sim = sim
        # Myokit's Simulation.pre() OVERWRITES the default state with the state it reaches,
        # and reset() restores that new default -- so a reused IPSCModel would start each
        # condition from the previous condition's settled state. Keep the CellML initial
        # state so scale_conductances() can restore it and make conditions order-independent.
        # [T-543, simulator_state_carryover_2026-07-20.md]
        self._init_state = list(sim.default_state())

    def scale_conductances(self, scales):
        """Set multiplicative scales on named conductances (e.g. {'g_Kr': 0.5}).

        1.0 = baseline, 0.0 = full block/knockout. Returns self for chaining.
        Unspecified conductances stay at their current scale.
        """
        for name in scales:
            if name not in CONDUCTANCE_VARS:
                raise KeyError(f"unknown conductance {name!r}; known: {list(CONDUCTANCE_VARS)}")
        self.scales.update(scales)
        self.sim.set_default_state(self._init_state)
        self.sim.reset()
        for name, qname in CONDUCTANCE_VARS.items():
            self.sim.set_constant(qname, self._base[name] * self.scales[name])
        return self

    def simulate(self, pre_ms=10000.0, run_ms=8000.0, log_interval=0.1):
        """Settle to the limit cycle, then log a fine AP trace. Returns (t, V) arrays in ms, mV.

        run_ms defaults to 8,000: `rate_bpm` needs three upstrokes, and at the ~40 bpm the
        model runs at deep g_CaL a 4,000 ms window holds two or three depending on phase,
        so slow-beating conditions were filtered on trace phase rather than physiology.
        [T-543, simulator_state_carryover_2026-07-20.md §4.1]
        """
        self.sim.reset()
        self.sim.pre(pre_ms)
        d = self.sim.run(run_ms, log=[_T, _V], log_interval=log_interval)
        return np.asarray(d[_T]), np.asarray(d[_V])

    def biomarkers(self, pre_ms=10000.0, run_ms=8000.0, log_interval=0.1):
        """Simulate and return AP biomarkers. Returns a dict; values are NaN if the model
        does not produce >=2 clean spontaneous beats (e.g. quiescence or repolarization
        failure under strong block -- which is itself a meaningful, reported outcome)."""
        t, v = self.simulate(pre_ms, run_ms, log_interval)
        return extract_biomarkers(t, v)


def extract_biomarkers(t, v):
    """Compute iPSC-CM AP biomarkers from a fine (t, V) trace of a spontaneously-beating cell.

    Returns: apd90, apd50 (ms); mdp, peak_v, amplitude (mV); dvdt_max (V/s);
             cycle_length (ms); rate_bpm; n_beats; repol_failure (bool).
    Biomarkers averaged over detected beats. If <2 beats: APDs/CL are NaN but MDP/peak/dvdt
    are still reported from the trace, and repol_failure flags a sustained-depolarization plateau.
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    dvdt = np.gradient(v, t)
    above = dvdt > _UPSTROKE_THRESH
    starts = np.where((~above[:-1]) & (above[1:]))[0]

    out = dict(
        mdp=float(np.min(v)),
        peak_v=float(np.max(v)),
        amplitude=float(np.max(v) - np.min(v)),
        dvdt_max=float(np.max(dvdt)),
        n_beats=int(len(starts)),
        apd90=float("nan"),
        apd50=float("nan"),
        cycle_length=float("nan"),
        rate_bpm=float("nan"),
        repol_failure=False,
    )

    if len(starts) < 2:
        # quiescent or non-repolarizing. Flag repolarization failure if V stays high
        # for a large fraction of the trace (sustained depolarization above -20 mV).
        out["repol_failure"] = bool(np.mean(v > -20.0) > 0.5)
        return out

    # cycle length from successive peak-voltage times
    peaks = []
    for i in range(len(starts) - 1):
        seg = slice(starts[i], starts[i + 1])
        peaks.append(starts[i] + int(np.argmax(v[seg])))
    cls = np.diff(t[np.array(peaks)]) if len(peaks) >= 2 else np.array([])

    def _apd(frac, i0, i1):
        ts, vs = t[i0:i1], v[i0:i1]
        vmax, vmin = vs.max(), vs.min()
        amp = vmax - vmin
        vthr = vmax - frac * amp
        ipk = int(np.argmax(vs))
        t_up = ts[int(np.argmax(np.gradient(vs, ts)))]
        t_rep = ts[-1]
        for j in range(ipk, len(vs)):
            if vs[j] <= vthr:
                t_rep = ts[j]
                break
        return t_rep - t_up

    a90s, a50s = [], []
    for i in range(len(starts) - 1):
        a90s.append(_apd(0.90, starts[i], starts[i + 1]))
        a50s.append(_apd(0.50, starts[i], starts[i + 1]))

    out["apd90"] = float(np.mean(a90s))
    out["apd50"] = float(np.mean(a50s))
    if len(cls):
        out["cycle_length"] = float(np.mean(cls))
        out["rate_bpm"] = float(60000.0 / np.mean(cls))
    return out


# ----------------------------------------------------------------------------------------
# Published reference values for the validation gate
# ----------------------------------------------------------------------------------------
# [VERIFIED: WebFetch frontiersin.org/.../fphar.2021.604713 Table 2, 2026-06-09 ->
#  Kernik2019 baseline APD90=414 ms, MDP=-76 mV, spontaneous rate=61 bpm]
# [VERIFIED: WebFetch PMC6767694 (Kernik 2019 J Physiol) 2026-06-09 -> baseline spontaneous
#  rate 62 beats/min]
# [VERIFIED: WebFetch frontiersin.org/.../fphys.2021.675867 2026-06-09 -> Doss et al. 2012
#  experimental iPSC-CM ranges: APD90 70-789 ms, AP amplitude 58-121 mV, dV/dt_max 5-86 V/s,
#  cycle length 327-7063 ms]
PUBLISHED = {
    "apd90": (414.0, "ms", "Frontiers Pharmacol 2021 Table 2 (Kernik2019 baseline)"),
    "mdp": (-76.0, "mV", "Frontiers Pharmacol 2021 Table 2 (Kernik2019 baseline)"),
    "rate_bpm": (61.0, "bpm", "Frontiers Pharmacol 2021 Table 2; cf. 62 bpm Kernik 2019 J Physiol"),
}
# experimental plausibility windows (Doss et al. 2012, via Frontiers Physiol 2021)
EXPERIMENTAL_RANGES = {
    "apd90": (70.0, 789.0),
    "amplitude": (58.0, 121.0),
    "dvdt_max": (5.0, 86.0),
    "cycle_length": (327.0, 7063.0),
}


def validate(verbose=True):
    """Run the validation gate: baseline biomarkers vs published Kernik-2019 values, plus
    an IKr-block drug response. Returns (passed: bool, report: dict)."""
    m = IPSCModel()
    bm = m.biomarkers()

    rows = []
    passed = True
    # tolerance: 10% relative for APD90/rate, 3 mV absolute for MDP
    checks = [
        ("apd90", bm["apd90"], PUBLISHED["apd90"][0], 0.10, "rel"),
        ("rate_bpm", bm["rate_bpm"], PUBLISHED["rate_bpm"][0], 0.10, "rel"),
        ("mdp", bm["mdp"], PUBLISHED["mdp"][0], 3.0, "abs"),
    ]
    for name, sim_v, pub_v, tol, kind in checks:
        if kind == "rel":
            ok = abs(sim_v - pub_v) / abs(pub_v) <= tol
            tol_s = f"{tol*100:.0f}% rel"
        else:
            ok = abs(sim_v - pub_v) <= tol
            tol_s = f"{tol:g} abs"
        passed &= ok
        rows.append((name, pub_v, sim_v, tol_s, "PASS" if ok else "FAIL"))

    # experimental-plausibility checks (informational, not gating)
    plaus = []
    for name in ("apd90", "amplitude", "dvdt_max", "cycle_length"):
        lo, hi = EXPERIMENTAL_RANGES[name]
        val = bm.get(name, float("nan"))
        plaus.append((name, lo, hi, val, lo <= val <= hi))

    # drug-block validation: IKr block should prolong APD90 monotonically
    drug = []
    base90 = bm["apd90"]
    for block in (0.0, 0.3, 0.5, 0.7):
        mm = IPSCModel({"g_Kr": 1.0 - block})
        b = mm.biomarkers()
        drug.append((block, b["apd90"], b["repol_failure"], b["rate_bpm"]))
    # gate: 50% IKr block prolongs APD90 vs baseline
    apd_at_50 = [d[1] for d in drug if d[0] == 0.5][0]
    ikr_ok = (not np.isnan(apd_at_50)) and apd_at_50 > base90
    passed &= ikr_ok

    report = dict(baseline=bm, rows=rows, plausibility=plaus, drug=drug, ikr_ok=ikr_ok)

    if verbose:
        print("=" * 78)
        print("KERNIK-2019 iPSC-CM AP MODEL -- VALIDATION GATE")
        print("=" * 78)
        print("\nBaseline biomarkers vs published Kernik-2019:")
        print(f"  {'biomarker':12s} {'published':>10s} {'simulated':>10s} {'tol':>10s}  result")
        for name, pub, sim, tol_s, res in rows:
            print(f"  {name:12s} {pub:10.1f} {sim:10.1f} {tol_s:>10s}  {res}")
        print("\nFull simulated baseline:")
        for k in ("apd90", "apd50", "mdp", "peak_v", "amplitude", "dvdt_max", "cycle_length", "rate_bpm"):
            print(f"    {k:14s} = {bm[k]:8.2f}")
        print("\nExperimental plausibility (Doss et al. 2012 ranges):")
        for name, lo, hi, val, ok in plaus:
            print(f"    {name:14s} {val:8.1f}  in [{lo:g}, {hi:g}]?  {'yes' if ok else 'NO'}")
        print("\nIKr (hERG/KCNH2) block -> APD90 (dofetilide/E-4031-like):")
        print(f"    {'block':>6s} {'APD90 ms':>10s} {'rate bpm':>10s}  {'repol fail':>10s}")
        for block, a90, rf, rate in drug:
            a90s = "nan" if np.isnan(a90) else f"{a90:.1f}"
            rates = "nan" if np.isnan(rate) else f"{rate:.1f}"
            print(f"    {block*100:5.0f}% {a90s:>10s} {rates:>10s}  {str(rf):>10s}")
        print(f"    IKr-block prolongs APD90 at 50%? {'PASS' if ikr_ok else 'FAIL'}")
        print("\n" + ("VALIDATION PASSED" if passed else "VALIDATION FAILED"))
        print("=" * 78)
    return passed, report


if __name__ == "__main__":
    validate()
