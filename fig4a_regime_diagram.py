"""
Regime map v5 — panel-sized redraw of v4 for Fig. 5a.

Changes from v4:
  * Drawn at its final physical size (88 x 78 mm) with Nature-scale type,
    so nothing is shrunk further at assembly.
  * Source footnote removed (habitat parameter citations belong in Methods /
    Supplementary Table, not on the panel).
  * Title, model parameters and the "NOT % gain" caveat moved to the legend.
  * Habitat boxes carry names only; the dynamical regime is read from the zone
    the box sits in. Rice rhizosphere drawn heavier as the system of this study.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT = _os.path.abspath(_os.path.join(HERE, "..", ".."))   # the submit/ folder
OUTDIR = _os.path.join(HERE, "output"); _os.makedirs(OUTDIR, exist_ok=True)
import matplotlib as _mpl; _mpl.rc_file(_os.path.join(ROOT, "scripts", "matplotlibrc"))

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

r = 0.8; K = 1.0e8; beta0 = 2.0e-9; B0 = 50.0
T_day = 1.0; omega = 2.0 * np.pi / T_day


def endemic(delta, m):
    S = m / (beta0 * (B0 - 1.0))
    x = (K - S) / (S / delta + K / r)
    return S, beta0 * S * (x / beta0) / delta, x / beta0


def A_matrix(delta, m):
    S, _, V = endemic(delta, m)
    a = r * S / K; bV = beta0 * V; bS = beta0 * S; me = m + bS
    return np.array([[-a, -a, -bS], [bV, -delta, bS], [-bV, B0 * delta, -me]])


def max_real_eig(delta, m):
    return float(np.max(np.linalg.eigvals(A_matrix(delta, m)).real))


def Phi_V_full(delta, m):
    A = A_matrix(delta, m).astype(complex)
    M = 1j * omega * np.eye(3, dtype=complex) - A
    X = np.linalg.solve(M, np.array([-1.0, 1.0, -1.0], dtype=complex))
    return (-np.angle(X[2])) % (2.0 * np.pi)


ng = 320
delta_vals = np.logspace(np.log10(0.3), np.log10(120.0), ng)
m_vals = np.logspace(np.log10(0.005), np.log10(20.0), ng)
DD, MM = np.meshgrid(delta_vals, m_vals, indexing="ij")

Phi = np.zeros_like(DD); Stab = np.zeros_like(DD)
for i in range(ng):
    for j in range(ng):
        Phi[i, j] = Phi_V_full(DD[i, j], MM[i, j])
        Stab[i, j] = max_real_eig(DD[i, j], MM[i, j])

benefit = -np.cos(Phi)
m_extinct = beta0 * K * (B0 - 1.0)
extinct = MM > m_extinct
unstable = (Stab > 0.0) & (~extinct)
stable = (~unstable) & (~extinct)
benefit_stable = np.where(stable, benefit, np.nan)

# (label, d_lo, d_hi, m_lo, m_hi, colour, label offset in points)
habitats = [
    ("Open ocean",       1.0,  2.4,  0.20, 1.0,  "#2ca02c", (  0,   0)),
    ("Coastal",          2.0,  6.0,  0.50, 2.0,  "#1f77b4", ( 16,  19)),
    ("Bulk soil",        0.5,  3.0,  0.05, 0.15, "#8c564b", (  0,   0)),
    ("Rice rhizosphere", 2.0, 10.0,  0.20, 0.50, "#9467bd", ( 36, -14)),
    ("Gut",              8.0, 48.0,  1.00, 8.0,  "#ff7f0e", (  0,   0)),
    ("Lab phage",       20.0, 72.0,  0.50, 2.0,  "#d62728", (  0,   0)),
]

plt.rcParams["hatch.color"] = "#c0504d"
plt.rcParams["hatch.linewidth"] = 0.45

fig, ax = plt.subplots(figsize=(3.46, 3.07))          # 88 x 78 mm

im = ax.pcolormesh(DD, MM, benefit_stable, cmap="YlGnBu",
                   shading="gouraud", vmin=0.0, vmax=1.0)
cb = plt.colorbar(im, ax=ax, pad=0.025, fraction=0.045)
cb.set_label(r"phase-alignment index $-\cos\Phi_V$", fontsize=7, labelpad=2)
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cb.outline.set_linewidth(0.5)

ax.contourf(DD, MM, unstable.astype(float), levels=[0.5, 1.5],
            colors=["#e9e2d0"], alpha=0.85)
ax.contourf(DD, MM, extinct.astype(float), levels=[0.5, 1.5],
            colors="none", hatches=["///"])
ax.contour(DD, MM, Stab, levels=[0.0], colors=["#5b4636"], linewidths=1.3)
ax.axhline(m_extinct, color="firebrick", lw=1.1)
ax.text(delta_vals[-1] * 0.92, m_extinct * 1.10, "virus extinct",
        color="firebrick", fontsize=6.5, ha="right", va="bottom")

cs = ax.contour(DD, MM, benefit_stable, levels=[0.2, 0.4, 0.6, 0.8],
                colors="white", linewidths=0.5, alpha=0.85)
ax.clabel(cs, fmt="%.1f", fontsize=6, inline=True)

ax.text(0.85, 5.0, "self-sustained\noscillation", fontsize=7, color="#5b4636",
        ha="center", va="center", linespacing=1.15,
        bbox=dict(boxstyle="round,pad=0.25", fc="#faf6ec", ec="#5b4636",
                  lw=0.5, alpha=0.92))
ax.text(40.0, 0.020, "stable endemic\nequilibrium", fontsize=7, color="#31708e",
        ha="center", va="center", linespacing=1.15)

for label, d_lo, d_hi, m_lo, m_hi, color, off in habitats:
    focal = label == "Rice rhizosphere"
    ax.add_patch(Rectangle((d_lo, m_lo), d_hi - d_lo, m_hi - m_lo,
                           linewidth=1.6 if focal else 1.0,
                           edgecolor=color, facecolor="none", zorder=4))
    cx, cy = np.sqrt(d_lo * d_hi), np.sqrt(m_lo * m_hi)
    ax.annotate(label, xy=(cx, cy), xytext=off, textcoords="offset points",
                ha="center", va="center", fontsize=6.5, zorder=5,
                fontweight="bold" if focal else "normal",
                color="white" if focal else "black",
                bbox=dict(boxstyle="round,pad=0.18", fc=color, ec=color,
                          lw=0.4, alpha=0.95 if focal else 0.85))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"lysis rate $\delta$ (day$^{-1}$)", fontsize=7.5, labelpad=2)
ax.set_ylabel(r"virion clearance rate $m$ (day$^{-1}$)", fontsize=7.5, labelpad=2)
ax.set_xlim(delta_vals[0], delta_vals[-1]); ax.set_ylim(m_vals[0], m_vals[-1])
ax.tick_params(labelsize=6.5, pad=1.5)
ax.grid(True, which="major", alpha=0.18, lw=0.4)

fig.tight_layout(pad=0.25)
base = _os.path.join(OUTDIR, "Fig4a_regime_diagram")
for ext in ("png", "pdf"):
    fig.savefig(base + "." + ext, dpi=600, facecolor="white")
print("saved:", base + ".png")
