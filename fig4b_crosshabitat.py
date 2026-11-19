"""
(E-v2) Panel-sized
(E) Fair cross-habitat comparison with PER-HABITAT joint parameterisation.

Fixes the artefact found in analysis_d: holding beta0 fixed made low-density
(marine) habitats fall below R0=1 and go extinct, and inflated others. Here each
draw uses the habitat's real (K, delta, m, B0) AND an effective reproduction number
R0 ~ logU(2, 50) (moderate, as implied by VBR ~5-10 and infection frequencies of
~1-13% in nature; Suttle 2007; Roy et al. 2020), from which the adsorption constant
is back-derived:  beta0 = R0 * m / (K (B0-1)).  This guarantees realistic viral
persistence in every habitat and isolates the dynamical-regime effect of (K, delta, m)
under comparable viral pressure. beta0 thus absorbs the unmodelled losses and the
density differences across habitats (it spans ~1e-6 [ocean] to ~1e-12 [gut]).

This REPLACES the fixed-K=1e8 MC (habitat_montecarlo_v2) as the cross-habitat panel.
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

r = 0.8; T_day = 1.0; omega = 2.0 * np.pi / T_day
N_PER_HAB = 500
CACHE     = _os.path.join(ROOT, "data", "model", "crosshabitat_joint.npz")
SEED = 7
R0_RANGE = (2.0, 50.0)

HAB = {
    "Open ocean":       dict(d=(1.0, 2.4),  m=(0.20, 1.0),  B0=(30, 80),  K=(1e5, 1e6),  c="#2ca02c"),
    "Coastal":          dict(d=(2.0, 6.0),  m=(0.50, 2.0),  B0=(30, 100), K=(1e6, 1e7),  c="#1f77b4"),
    "Bulk soil":        dict(d=(0.5, 3.0),  m=(0.05, 0.15), B0=(10, 44),  K=(1e7, 1e8),  c="#8c564b"),
    "Rice rhizosphere": dict(d=(2.0, 10.0), m=(0.20, 0.50), B0=(20, 50),  K=(1e9, 1e10), c="#9467bd"),
    "Mammalian gut":    dict(d=(8.0, 48.0), m=(1.0, 8.0),   B0=(50, 200), K=(1e11, 1e12),c="#ff7f0e"),
    "Lab E.coli+T4/T7": dict(d=(36.0, 72.0),m=(0.5, 2.0),   B0=(50, 200), K=(1e8, 1e9),  c="#d62728"),
}


def starting_point(beta0, K, delta, m, B0):
    S = m / (beta0 * (B0 - 1.0))          # = K / R0
    if S >= 0.99 * K:
        return [0.9 * K, 1.0, 1.0e4]
    V = (K - S) / (S / delta + K / r) / beta0
    I = beta0 * S * V / delta
    return [S * 1.1, max(I * 0.9, 1.0), max(V * 0.9, 1.0)]


def mean_S(a, beta0, K, delta, m, B0, t_end=200.0, t_avg=80.0):
    y0 = starting_point(beta0, K, delta, m, B0)

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
    hab, K, delta, m, B0, R0 = args
    beta0 = R0 * m / (K * (B0 - 1.0))
    Sb = mean_S(0.0, beta0, K, delta, m, B0)
    Sr = mean_S(0.5, beta0, K, delta, m, B0)
    if not (np.isfinite(Sb) and np.isfinite(Sr) and Sb > 1.0):
        return (hab, np.nan)
    g = (Sr - Sb) / Sb * 100.0
    return (hab, g if abs(g) < 1e3 else np.nan)


def lu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def main():
    rng = np.random.default_rng(SEED)
    jobs = []
    for hab, p in HAB.items():
        for _ in range(N_PER_HAB):
            jobs.append((hab, lu(rng, *p["K"]), lu(rng, *p["d"]),
                         lu(rng, *p["m"]), lu(rng, *p["B0"]), lu(rng, *R0_RANGE)))
    print(f"running {len(jobs)} draws ({N_PER_HAB}/habitat) x2 sims, R0~logU{R0_RANGE}...")
    t0 = time.time()
    with Pool(processes=max(1, cpu_count() - 1)) as p:
        res = p.map(one, jobs)
    print(f"done in {time.time()-t0:.1f}s")

    by = {h: [] for h in HAB}
    for hab, g in res:
        if np.isfinite(g):
            by[hab].append(g)

    labels = list(HAB.keys())
    lines = [f"{'habitat':18s} {'n':>4} {'median':>9} {'mean':>8} {'p10':>7} {'p90':>7} {'%pos':>5}"]
    for h in labels:
        g = np.array(by[h])
        lines.append(f"{h:18s} {len(g):>4} {np.median(g):>+8.2f}% {np.mean(g):>+7.2f}% "
                     f"{np.percentile(g,10):>+6.2f}% {np.percentile(g,90):>+6.2f}% "
                     f"{(g>0).mean()*100:>4.0f}%")
    table = "\n".join(lines)
    np.savez(CACHE, **{h.replace(" ", "_").replace(".", ""): np.array(by[h]) for h in labels})
    with open("/tmp/cross_joint.txt", "w") as fh:
        fh.write(table + "\n")
    print("\n" + table)

    _plot({h: np.array(by[h]) for h in labels})


SHORT = {"Open ocean": "Ocean", "Coastal": "Coastal", "Bulk soil": "Soil",
         "Rice rhizosphere": "Rhizosphere", "Mammalian gut": "Gut",
         "Lab E.coli+T4/T7": "Lab"}


def _plot(by):
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    labels = list(HAB.keys())
    data = [by[h] for h in labels]
    colors = [HAB[h]["c"] for h in labels]
    pos = np.arange(1, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(3.54, 1.54))          # 90 x 39 mm
    vp = ax.violinplot(data, positions=pos, widths=0.82, showextrema=False)
    for b, c in zip(vp["bodies"], colors):
        b.set_facecolor(c); b.set_alpha(0.45); b.set_edgecolor(c); b.set_linewidth(0.5)
    bp = ax.boxplot(data, positions=pos, widths=0.11, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.0),
                    boxprops=dict(facecolor="white", alpha=0.9, linewidth=0.5),
                    whiskerprops=dict(color="#555555", linewidth=0.5),
                    capprops=dict(color="#555555", linewidth=0.5))
    ax.axhline(0, color="grey", lw=0.6)
    allg = np.concatenate(data)
    lo = max(-12.0, np.percentile(allg, 1.0) - 1.5)
    hi = min(18.0, np.percentile(allg, 99.0) + 5.0)
    ax.set_ylim(lo, hi)
    for i, g in enumerate(data):
        ax.annotate("%.0f%%+" % ((g > 0).mean() * 100),
                    xy=(pos[i], min(np.percentile(g, 95), hi * 0.72)), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=colors[i], fontweight="bold")
    ax.set_xticks(pos)
    ax.set_xticklabels([SHORT[h] for h in labels], fontsize=7.5)
    for t, h in zip(ax.get_xticklabels(), labels):
        if h == "Rice rhizosphere":
            t.set_fontweight("bold"); t.set_color(HAB[h]["c"])
    ax.set_ylabel(r"gain $\Delta\langle S\rangle/\langle S\rangle_0$ (%)",
                  fontsize=8.5, labelpad=2)
    ax.tick_params(labelsize=7.5, pad=1)
    ax.grid(alpha=0.18, axis="y", lw=0.4)
    fig.tight_layout(pad=0.25)
    fig.canvas.draw()
    fw = fig.get_figwidth() * fig.dpi
    fh = fig.get_figheight() * fig.dpi
    for art in (ax.xaxis.label, ax.yaxis.label):
        if not art.get_text():
            continue
        bb = art.get_window_extent(fig.canvas.get_renderer())
        if bb.x0 < -0.5 or bb.x1 > fw + 0.5 or bb.y0 < -0.5 or bb.y1 > fh + 0.5:
            raise SystemExit("axis label overflows the canvas: %r" % art.get_text())
    base = _os.path.join(OUTDIR, "Fig4b_crosshabitat")
    for ext in ("png", "pdf"):
        fig.savefig(base + "." + ext, dpi=600, facecolor="white")
    print("figure:", base + ".png")


if __name__ == "__main__":
    import os, sys
    if os.path.exists(CACHE) and "--rerun" not in sys.argv:
        d = np.load(CACHE)
        print("replotting from cache:", CACHE)
        _plot({h: d[h.replace(" ", "_").replace(".", "")] for h in HAB})
    else:
        main()
