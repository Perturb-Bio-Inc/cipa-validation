#!/usr/bin/env python3
"""
T-561 — Kernik-2019 iPSC-CM model with an added ORd-style late-sodium current.

The published Kernik-2019 CellML has no distinct late-sodium current (INaL). The CiPA
harness therefore cannot represent INaL block, which biases the model against INaL-protective
low-risk drugs such as ranolazine and mexiletine. This module adds a phenomenological,
ORd-style INaL on top of the published CellML (which stays pristine on disk) and exposes it
as a fourth scaled conductance, g_NaL.

Formulation (O'Hara-Rudy 2011, doi:10.1371/journal.pcbi.1002061):
  i_NaL = g_NaL * mL * hL * (V - E_Na)
  dmL/dt = (mL_inf - mL)/mL_tau,   mL_inf = 1/(1+exp((V+42.85)/-5.264)),  mL_tau = 1.5 ms
  dhL/dt = (hL_inf - hL)/hL_tau,   hL_inf = 1/(1+exp((V+87.1)/7.488)),    hL_tau = 600 ms
E_Na is the model's own dynamic Nernst potential (erev.E_Na), not ORd's fixed value, so the
late current stays consistent with the Kernik ion balance.

g_NaL calibration: g_NaL = 0.10 mS/uF gives a sustained late-sodium fraction of ~1% of peak
INa at 200 ms into the plateau (measured from this model's own traces). That is within the
reported range (~0.5-2%) for persistent sodium in human ventricular myocytes, so it is a
pre-specified, physiologically-motivated value, not one tuned to make the benchmark pass.
A side effect is that the drug-free baseline APD90 lengthens (413 ms -> 588 ms at this
g_NaL); that is reported as a limitation rather than hidden.

The published kernik_2019.cellml is never modified. The INaL is injected by splicing a
component into the model's parsed text and re-parsing. The sustained late-sodium fraction
measured with this module's own late_fraction() is ~1.03% of peak INa at 200 ms into the
plateau (run 2026-09-06). The ~0.5-2% figure cited for human persistent sodium is taken from
the cardiac modeling literature [UNVERIFIED: not re-checked against the primary in this
session].

Run: none (import module). Simulate via IPSCModelINaL below.
"""
import os
import myokit
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CELLML_PATH = os.path.join(_HERE, "kernik_2019.cellml")
HERE = _HERE

# Pre-specified late-sodium maximal conductance (mS/uF). See module docstring.
G_NAL = 0.10


def build_inal_model(g_NaL=G_NAL):
    """Return a Kernik-2019 Myokit model with an injected ORd-style INaL component."""
    m = myokit.formats.importer("cellml").model(CELLML_PATH)
    code = m.code()
    hdr_marker = "# Initial values\n"
    assert hdr_marker in code
    code = code.replace(hdr_marker, hdr_marker + f"inaL.mL = 0.0\ninaL.hL = 1.0\n", 1)
    # wire the late current into the membrane sum and the sodium balance
    code = code.replace("ina.i_Na + inaca.i_NaCa",
                        "ina.i_Na + inaL.i_NaL + inaca.i_NaCa", 1)
    code = code.replace("+ina.i_Na + ibna.i_b_Na + ifunny.i_fNa",
                        "+ina.i_Na + inaL.i_NaL + ibna.i_b_Na + ifunny.i_fNa", 1)
    comp = f"""
[inaL]
g_NaL = {g_NaL}
    in [mS/uF]
E_Na = erev.E_Na
    in [mV]
mL_inf = 1/(1+exp((membrane.V+42.85)/-5.264))
    in [1]
mL_tau = 1.5
    in [ms]
hL_inf = 1/(1+exp((membrane.V+87.1)/7.488))
    in [1]
hL_tau = 600
    in [ms]
i_NaL = g_NaL * mL * hL * (membrane.V - E_Na)
    in [A/F]
dot(mL) = (mL_inf - mL)/mL_tau
    in [1/ms]
dot(hL) = (hL_inf - hL)/hL_tau
    in [1/ms]
"""
    return myokit.parse_model(code + comp)


# Reuse the standard CONDUCTANCE_VARS (g_NaL is handled separately).
from ipsc_cm_ap_model import CONDUCTANCE_VARS, extract_biomarkers  # noqa: E402


class IPSCModelINaL:
    """Kernik-2019 iPSC-CM model plus INaL, with conductance scaling including g_NaL."""

    def __init__(self, conductance_scales=None, g_NaL=G_NAL):
        self.g_NaL = g_NaL
        self.model = build_inal_model(g_NaL)
        self._base = {}
        for name, qname in CONDUCTANCE_VARS.items():
            self._base[name] = float(self.model.get(qname).eval())
        self._base["g_NaL"] = g_NaL
        self.scales = {name: 1.0 for name in list(CONDUCTANCE_VARS) + ["g_NaL"]}
        if conductance_scales:
            self.scales.update(conductance_scales)
        self._build_sim()

    def _build_sim(self):
        sim = myokit.Simulation(self.model)
        for name, qname in CONDUCTANCE_VARS.items():
            sim.set_constant(qname, self._base[name] * self.scales[name])
        sim.set_constant("inaL.g_NaL", self._base["g_NaL"] * self.scales["g_NaL"])
        self.sim = sim
        self._init_state = list(sim.default_state())

    def scale_conductances(self, scales):
        for name in scales:
            if name not in self.scales:
                raise KeyError(f"unknown conductance {name!r}")
        self.scales.update(scales)
        self.sim.set_default_state(self._init_state)
        self.sim.reset()
        for name, qname in CONDUCTANCE_VARS.items():
            self.sim.set_constant(qname, self._base[name] * self.scales[name])
        self.sim.set_constant("inaL.g_NaL", self._base["g_NaL"] * self.scales["g_NaL"])
        return self

    def simulate(self, pre_ms=10000.0, run_ms=8000.0, log_interval=0.1):
        self.sim.reset()
        self.sim.pre(pre_ms)
        d = self.sim.run(run_ms, log=["engine.time", "membrane.V"], log_interval=log_interval)
        return np.asarray(d["engine.time"]), np.asarray(d["membrane.V"])

    def biomarkers(self, pre_ms=10000.0, run_ms=8000.0, log_interval=0.1):
        t, v = self.simulate(pre_ms, run_ms, log_interval)
        return extract_biomarkers(t, v)


def late_fraction(g_NaL=G_NAL, pre_ms=10000, run_ms=8000, log_interval=0.1):
    """Measure sustained late-sodium as % of peak INa, 200 ms into each plateau."""
    m = IPSCModelINaL({}, g_NaL=g_NaL)
    m.sim.reset()
    m.sim.pre(pre_ms)
    d = m.sim.run(run_ms, log=["engine.time", "membrane.V", "inaL.i_NaL", "ina.i_Na"],
                  log_interval=log_interval)
    t, V = d["engine.time"], d["membrane.V"]
    inal, ina = d["inaL.i_NaL"], d["ina.i_Na"]
    dv = np.gradient(V, t)
    above = dv > 10.0
    starts = np.where((~above[:-1]) & (above[1:]))[0]
    fracs = []
    for s0 in starts:
        i0 = np.searchsorted(t, t[s0])
        ipeak = slice(i0, np.searchsorted(t, t[s0] + 20.0))
        i200 = np.searchsorted(t, t[s0] + 200.0)
        if i200 >= len(t):
            break
        pk = np.max(np.abs(ina[ipeak]))
        if pk > 0:
            fracs.append(abs(inal[i200]) / pk * 100.0)
    return float(np.mean(fracs)) if fracs else float("nan")


if __name__ == "__main__":
    print("baseline (drug-free, with INaL):")
    print({k: round(v, 2) for k, v in IPSCModelINaL({}).biomarkers().items()})
    print(f"late-sodium fraction @200ms: {late_fraction():.3f}%")
