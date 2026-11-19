"""
(C-v2) Panel-sized K-sensitivity of the rice-rhizosphere host-rhythm benefit.

The habitat Monte-Carlo (analysis_b_v2) fixed K=1e8/mL, but the rhizosphere's real
host density is ~1e9-1e10/g — 1-2 orders higher than bulk soil. Here we sweep K
across the bulk-soil -> rhizosphere range, drawing (delta, m, B0) from the rice
rhizosphere box at each K, and measure the benefit Delta<S>/<S>_0 (a=0.5, T=24h).

Reports median + IQR + %positive vs K, so we can state how the rhizosphere magnitude
moves once its true (high) host density is used instead of the fixed 1e8.
Parallelised. Sources for the rhizosphere box: see final_regime_diagram_v2.py header.
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

r       = 0.8
beta0   = 2.0e-9
T_day   = 1.0
omega   = 2.0 * np.pi / T_day

# rice rhizosphere parameter box
BOX = {"delta": (2.0, 10.0), "m": (0.20, 0.50), "B0": (20, 50)}
INKG    = "#6b6b6b"
N_PER_K = 200
CACHE   = _os.path.join(ROOT, "data", "model", "ksweep_rhizosphere.npz")
SEED    = 23
K_VALS  = np.logspace(7.0, 10.0, 10)   # 1e7 .. 1e10 cells/mL


def starting_point(K, delta, m, B0):
    S = m / (beta0 * (B0 - 1.0))
    if S >= 0.99 * K:
        return [0.9 * K, 1.0, 1.0e4]
    V = (K - S) / (S / delta + K / r) / beta0
    I = beta0 * S * V / delta
    return [S * 1.1, max(I * 0.9, 1.0), max(V * 0.9, 1.0)]


def sim_mean_S(a, K, delta, m, B0, t_end=200.0, t_avg=80.0):
    y0 = starting_point(K, delta, m, B0)

    def rhs(t, y):
        S, I, V = y
        s = 1.0 + a * np.cos(omega * t)
        inf = beta0 * s * S * V
        return [r * S * (1.0 - (S + I) / K) - inf,
                inf - delta * I,
                B0 * delta * I - m * V - inf]

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


def one_draw(args):
    K, delta, m, B0 = args
    Sb = sim_mean_S(0.0, K, delta, m, B0)
    Sr = sim_mean_S(0.5, K, delta, m, B0)
    if not (np.isfinite(Sb) and np.isfinite(Sr) and Sb > 1.0):
        return (K, np.nan)
    return (K, (Sr - Sb) / Sb * 100.0)


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def main():
    rng = np.random.default_rng(SEED)
    jobs = []
    for K in K_VALS:
        for _ in range(N_PER_K):
            jobs.append((K,
                         log_uniform(rng, *BOX["delta"]),
                         log_uniform(rng, *BOX["m"]),
                         log_uniform(rng, *BOX["B0"])))
    print(f"running {len(jobs)} draws ({N_PER_K} per K x {len(K_VALS)} K) "
          f"x2 sims on {cpu_count()} CPUs...")
    t0 = time.time()
    with Pool(processes=max(1, cpu_count() - 1)) as p:
        res = p.map(one_draw, jobs)
    print(f"done in {time.time()-t0:.1f}s")

    by_K = {K: [] for K in K_VALS}
    for K, g in res:
        if np.isfinite(g) and abs(g) < 1e3:   # drop overflow blow-ups
            by_K[K].append(g)

    def stat(K, f, default=np.nan):
        g = np.array(by_K[K])
        return f(g) if len(g) >= 5 else default

    med = np.array([stat(K, np.median) for K in K_VALS])
    p25 = np.array([stat(K, lambda g: np.percentile(g, 25)) for K in K_VALS])
    p75 = np.array([stat(K, lambda g: np.percentile(g, 75)) for K in K_VALS])
    pos = np.array([stat(K, lambda g: (g > 0).mean() * 100) for K in K_VALS])

    lines = [f"{'K (cells/mL)':>14} {'n':>4} {'median':>9} {'p25':>8} {'p75':>8} {'%pos':>6}"]
    for i, K in enumerate(K_VALS):
        lines.append(f"{K:>14.1e} {len(by_K[K]):>4d} {med[i]:>+8.2f}% "
                     f"{p25[i]:>+7.2f}% {p75[i]:>+7.2f}% {pos[i]:>5.0f}%")
    table = "\n".join(lines)
    np.savez(CACHE, K=K_VALS, med=med, p25=p25, p75=p75, pos=pos)
    with open(CACHE.replace(".npz", ".txt"), "w") as fh:
        fh.write(table + "\n")
    print("\n" + table)

    _plot(K_VALS, med, p25, p75)


def _plot(K_VALS, med, p25, p75):
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    ok = np.isfinite(med)
    Kp, medp, p25p, p75p = np.asarray(K_VALS)[ok], med[ok], p25[ok], p75[ok]

    fig, ax = plt.subplots(figsize=(4.33, 3.15))       # 110 x 80 mm
    ax.fill_between(Kp, p25p, p75p, color="#9467bd", alpha=0.22, lw=0,
                    label="IQR (p25\u2013p75)")
    ax.plot(Kp, medp, "o-", color="#9467bd", lw=1.3, ms=3.6, label="median")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(Kp[0], Kp[-1])
    ax.axvspan(2e7, 1e8, color="#8c564b", alpha=0.10, lw=0)
    ax.axvspan(1e9, 1e10, color="#9467bd", alpha=0.10, lw=0)
    ax.axvline(1e8, color="k", ls=":", lw=0.7)
    top = ax.get_ylim()[1]
    ax.text(4.5e7, top * 0.97, "bulk soil", color="#8c564b", fontsize=8,
            ha="center", va="top")
    ax.text(3.2e9, top * 0.97, "rice rhizosphere", color="#5b3a86", fontsize=8,
            ha="center", va="top", fontweight="bold")
    ax.set_xlabel(r"host carrying capacity $K$ (cells mL$^{-1}$)", fontsize=9,
                  labelpad=2)
    ax.set_ylabel(r"host-rhythm benefit $\Delta\langle S\rangle/\langle S\rangle_0$ (%)",
                  fontsize=9, labelpad=2)
    ax.tick_params(labelsize=8, pad=1.5)
    leg = ax.legend(loc="lower left", fontsize=8, frameon=False, handlelength=1.4,
                    borderpad=0.2, labelspacing=0.3)
    ax.grid(alpha=0.18, which="major", lw=0.4)
    fig.tight_layout(pad=0.25)
    base = _os.path.join(OUTDIR, "EDFig7_K_sensitivity_rhizosphere")
    for ext in ("png", "pdf"):
        fig.savefig(base + "." + ext, dpi=600, facecolor="white")
    print("figure:", base + ".png")


if __name__ == "__main__":
    import os, sys
    if os.path.exists(CACHE) and "--rerun" not in sys.argv:
        d = np.load(CACHE)
        print("replotting from cache:", CACHE)
        _plot(d["K"], d["med"], d["p25"], d["p75"])
    else:
        main()
