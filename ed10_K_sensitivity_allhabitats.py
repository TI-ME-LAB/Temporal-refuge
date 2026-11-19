"""
(D-v2) Panel-sized K-sensitivity for ALL habitats (not just rhizosphere).

The habitat MC (analysis_b_v2) fixed K=1e8/mL, but real host density spans
~1e5 (open ocean) to ~1e12 (gut). Benefit is strongly K-dependent (analysis_c),
so the fixed-K per-habitat magnitudes are not on a common footing. Here we sweep
K over a broad common range for each habitat (sampling delta,m,B0 from its box) and
plot benefit vs K, shading each habitat's REAL K window and marking the K=1e8 MC
reference. We also flag virus EXTINCTION (S_base/K>0.95): with beta0 held fixed,
low-density habitats (marine) fall below R0=1 and the virus cannot persist — an
artefact of the coupling R0=beta0*K*(B0-1)/m, and the reason a faithful cross-
habitat comparison needs per-habitat (K,beta0,B0), not a global model with K slid.

Parallelised. Box/source refs: see final_regime_diagram_v2.py header.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT = _os.path.abspath(_os.path.join(HERE, "..", ".."))   # the submit/ folder
OUTDIR = _os.path.join(HERE, "output"); _os.makedirs(OUTDIR, exist_ok=True)
import matplotlib as _mpl; _mpl.rc_file(_os.path.join(ROOT, "scripts", "matplotlibrc"))

r = 0.8; beta0 = 2.0e-9; T_day = 1.0; omega = 2.0 * np.pi / T_day
N_PER_K = 120
CACHE   = _os.path.join(ROOT, "data", "model", "ksweep_allhabitats.npz")
SEED = 41
K_VALS = np.logspace(5.0, 12.0, 12)   # 1e5 .. 1e12 cells/mL

# (delta, m, B0 box) + real K window (cells/mL) + color
HAB = {
    "Open ocean":       dict(d=(1.0, 2.4),  m=(0.20, 1.0),  B0=(30, 80),  K=(1e5, 1e6),  c="#2ca02c"),
    "Coastal":          dict(d=(2.0, 6.0),  m=(0.50, 2.0),  B0=(30, 100), K=(1e6, 1e7),  c="#1f77b4"),
    "Bulk soil":        dict(d=(0.5, 3.0),  m=(0.05, 0.15), B0=(10, 44),  K=(1e7, 1e8),  c="#8c564b"),
    "Rice rhizosphere": dict(d=(2.0, 10.0), m=(0.20, 0.50), B0=(20, 50),  K=(1e9, 1e10), c="#9467bd"),
    "Mammalian gut":    dict(d=(8.0, 48.0), m=(1.0, 8.0),   B0=(50, 200), K=(1e11, 1e12),c="#ff7f0e"),
    "Lab E.coli+T4/T7": dict(d=(36.0, 72.0),m=(0.5, 2.0),   B0=(50, 200), K=(1e8, 1e9),  c="#d62728"),
}


def starting_point(K, delta, m, B0):
    S = m / (beta0 * (B0 - 1.0))
    if S >= 0.99 * K:
        return [0.9 * K, 1.0, 1.0e4]
    V = (K - S) / (S / delta + K / r) / beta0
    I = beta0 * S * V / delta
    return [S * 1.1, max(I * 0.9, 1.0), max(V * 0.9, 1.0)]


def mean_S(a, K, delta, m, B0, t_end=200.0, t_avg=80.0):
    y0 = starting_point(K, delta, m, B0)

    def rhs(t, y):
        S, I, V = y
        s = 1.0 + a * np.cos(omega * t)
        inf = beta0 * s * S * V
        return [r * S * (1.0 - (S + I) / K) - inf,
                inf - delta * I, B0 * delta * I - m * V - inf]

    t_eval = np.linspace(t_end - t_avg, t_end, 4000)
    try:
        sol = solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                        rtol=1e-7, atol=1.0, t_eval=t_eval, max_step=0.05)
        if not sol.success:
            return np.nan
        S = sol.y[0]
        if (S < 0).any() or (S > 10 * K).any():
            return np.nan
        return float(S.mean())
    except Exception:
        return np.nan


def one(args):
    hab, K, delta, m, B0 = args
    Sb = mean_S(0.0, K, delta, m, B0)
    Sr = mean_S(0.5, K, delta, m, B0)
    if not (np.isfinite(Sb) and np.isfinite(Sr) and Sb > 1.0):
        return (hab, K, np.nan, False)
    extinct = (Sb / K) > 0.95          # virus failed to persist
    return (hab, K, (Sr - Sb) / Sb * 100.0, extinct)


def lu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def main():
    rng = np.random.default_rng(SEED)
    jobs = []
    for hab, p in HAB.items():
        for K in K_VALS:
            for _ in range(N_PER_K):
                jobs.append((hab, K, lu(rng, *p["d"]), lu(rng, *p["m"]), lu(rng, *p["B0"])))
    print(f"running {len(jobs)} draws ({N_PER_K}/K x {len(K_VALS)}K x {len(HAB)}hab) x2 sims...")
    t0 = time.time()
    with Pool(processes=max(1, cpu_count() - 1)) as p:
        res = p.map(one, jobs)
    print(f"done in {time.time()-t0:.1f}s")

    agg = {h: {K: {"g": [], "ext": 0, "n": 0} for K in K_VALS} for h in HAB}
    for hab, K, g, ext in res:
        cell = agg[hab][K]; cell["n"] += 1
        if ext:
            cell["ext"] += 1
        if np.isfinite(g) and abs(g) < 1e3:
            cell["g"].append(g)

    out = {}
    for hab in HAB:
        med, p25, p75, fext = [], [], [], []
        for K in K_VALS:
            g = np.array(agg[hab][K]["g"]); n = agg[hab][K]["n"]
            med.append(np.median(g) if len(g) >= 5 else np.nan)
            p25.append(np.percentile(g, 25) if len(g) >= 5 else np.nan)
            p75.append(np.percentile(g, 75) if len(g) >= 5 else np.nan)
            fext.append(agg[hab][K]["ext"] / n * 100 if n else np.nan)
        out[hab] = np.array([med, p25, p75, fext])
    np.savez(CACHE, K=K_VALS, **{h.replace(" ", "_"): v for h, v in out.items()})
    print("cached:", CACHE)
    _plot(K_VALS, out)


def _plot(K_VALS, out):
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.4,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.33), sharex=True, sharey=True)
    for ax, (hab, p) in zip(axes.ravel(), HAB.items()):
        med, p25, p75, fext = out[hab]
        ok = np.isfinite(med)
        ax.fill_between(np.asarray(K_VALS)[ok], p25[ok], p75[ok], color=p["c"],
                        alpha=0.20, lw=0)
        ax.plot(np.asarray(K_VALS)[ok], med[ok], "o-", color=p["c"], lw=1.2, ms=3.0)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvspan(p["K"][0], p["K"][1], color=p["c"], alpha=0.13, lw=0)
        ax.axvline(1e8, color="k", ls=":", lw=0.7)
        # grey hatch over the CONTIGUOUS K range where the virus failed to persist
        ext = np.asarray(fext, float)
        bad = np.where(ext > 50.0)[0]
        if bad.size:
            runs, start = [], bad[0]
            for a_, b_ in zip(bad[:-1], bad[1:]):
                if b_ != a_ + 1:
                    runs.append((start, a_)); start = b_
            runs.append((start, bad[-1]))
            Kv = np.asarray(K_VALS, float)
            for i0, i1 in runs:
                lo = Kv[i0] / 1.6 if i0 == 0 else np.sqrt(Kv[i0 - 1] * Kv[i0])
                hi = Kv[i1] * 1.6 if i1 == len(Kv) - 1 else np.sqrt(Kv[i1] * Kv[i1 + 1])
                ax.axvspan(lo, hi, facecolor="none", edgecolor="#9a9a9a", lw=0,
                           hatch="////", alpha=0.55, zorder=0)
        ax.set_xscale("log")
        # species name italic (2026-11-05); other titles unchanged
        title = {"Lab E.coli+T4/T7": r"Lab $\mathbfit{E}$. $\mathbfit{coli}$ + T4/T7"}.get(hab, hab)
        tt = ax.set_title(title, fontsize=9, color=p["c"], pad=3)
        if "$" in title:   # bold-italic Arial for the species name, matching the bold Arial titles
            plt.rcParams.update({"mathtext.bfit": "Arial:italic:bold", "mathtext.rm": "Arial"})
            tt.set_math_fontfamily("custom")
        ax.grid(alpha=0.18, which="major", lw=0.4)
        ax.tick_params(labelsize=8, pad=1.5)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$K$ (cells mL$^{-1}$)", fontsize=9, labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"benefit $\Delta\langle S\rangle/\langle S\rangle_0$ (%)",
                      fontsize=9, labelpad=2)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.8)
    base = _os.path.join(OUTDIR, "EDFig8_K_sensitivity_allhabitats")
    for ext_ in ("png", "pdf"):
        fig.savefig(base + "." + ext_, dpi=600, facecolor="white")
    print("figure:", base + ".png")


if __name__ == "__main__":
    import os, sys
    if os.path.exists(CACHE) and "--rerun" not in sys.argv:
        d = np.load(CACHE)
        print("replotting from cache:", CACHE)
        _plot(d["K"], {h: d[h.replace(" ", "_")] for h in HAB})
    else:
        main()
