"""
(F-v2) Panel-sized
(F) Minimal MULTI-HOST community SIV model — does rhythm give a community-level
competitive refuge? (matches the empirical R2 claim, which the single-pair model
in analysis_a-e cannot speak to.)

n host species share one carrying capacity K (competition); each has its own lytic
phage. We run the SAME randomly-drawn community twice:
  scenario A: all hosts arrhythmic (a=0)
  scenario B: a random FOCAL half made rhythmic (a=0.5, random phase phi_i)
and measure, for the focal hosts, their change in mean abundance and in community
abundance share (B vs A). A positive focal gain = rhythm confers a competitive
refuge within the community. MC over many random communities.

  dS_i/dt = r_i S_i (1 - (Σ S + Σ I)/K) - beta_i s_i(t) S_i V_i
  dI_i/dt = beta_i s_i(t) S_i V_i - delta_i I_i
  dV_i/dt = B_i delta_i I_i - m_i V_i - beta_i s_i(t) S_i V_i
  s_i(t) = 1 + a_i cos(omega t + phi_i)
beta_i set from per-host R0_i so each phage can persist on its competitive share.
First-pass model: kill-the-winner coexistence, rhizosphere-ish parameter ranges.
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

K = 1.0e8                 # shared carrying capacity (smaller magnitudes -> less overflow/stiffness)
T_day = 1.0; omega = 2 * np.pi / T_day
N_HOST = 10
N_COMM = 60
CACHE  = _os.path.join(ROOT, "data", "model", "community_model.npz")
SEED = 3
A_AMP = 0.5


def draw_community(rng):
    r = rng.uniform(0.5, 1.0, N_HOST)
    delta = np.exp(rng.uniform(np.log(2), np.log(10), N_HOST))
    m = np.exp(rng.uniform(np.log(0.2), np.log(0.5), N_HOST))
    B = np.exp(rng.uniform(np.log(20), np.log(50), N_HOST))
    R0 = np.exp(rng.uniform(np.log(12), np.log(30), N_HOST))   # > N_HOST so phages persist; modest to avoid extreme stiffness
    beta = R0 * m / (K * (B - 1.0))
    phi = rng.uniform(0, 2 * np.pi, N_HOST)
    focal = rng.permutation(N_HOST) < (N_HOST // 2)            # random half
    return dict(r=r, delta=delta, m=m, B=B, beta=beta, phi=phi, focal=focal)


def run(c, rhythmic):
    a = np.where(rhythmic, A_AMP, 0.0)
    r, delta, m, B, beta, phi = c["r"], c["delta"], c["m"], c["B"], c["beta"], c["phi"]
    n = N_HOST
    S0 = np.full(n, 0.3 * K / n); I0 = np.full(n, 1e2); V0 = np.full(n, 1e4)
    y0 = np.concatenate([S0, I0, V0])

    def rhs(t, y):
        y = np.maximum(y, 0.0)          # physical floor (prevents negative-driven blowup)
        S = y[:n]; I = y[n:2*n]; V = y[2*n:]
        s = 1.0 + a * np.cos(omega * t + phi)
        inf = beta * s * S * V
        Stot = S.sum() + I.sum()
        dS = r * S * (1 - Stot / K) - inf
        dI = inf - delta * I
        dV = B * delta * I - m * V - inf
        return np.concatenate([dS, dI, dV])

    t_eval = np.linspace(100, 150, 1500)
    try:
        sol = solve_ivp(rhs, (0, 150), y0, method="BDF", rtol=1e-5, atol=10.0,
                        t_eval=t_eval, max_step=1.0)
        if not sol.success:
            return None
        S = np.maximum(sol.y[:n], 0.0)
        if (S > 10 * K).any() or not np.isfinite(S).all():
            return None
        return S.mean(axis=1)          # mean abundance per host
    except Exception:
        return None


def one(seed):
    rng = np.random.default_rng(seed)
    c = draw_community(rng)
    A = run(c, np.zeros(N_HOST, bool))     # all arrhythmic
    Bsc = run(c, c["focal"])               # focal half rhythmic
    if A is None or Bsc is None or A.sum() <= 0 or Bsc.sum() <= 0:
        return None
    foc = c["focal"]
    # focal hosts: mean abundance gain B vs A, and community share change
    gain = (Bsc[foc].mean() - A[foc].mean()) / (A[foc].mean() + 1e-9) * 100
    share_A = A[foc].sum() / A.sum() * 100
    share_B = Bsc[foc].sum() / Bsc.sum() * 100
    return (gain, share_A, share_B)


def main():
    print(f"community model: {N_HOST} hosts, {N_COMM} communities, K={K:.0e}")
    t0 = time.time()
    with Pool(processes=max(1, cpu_count() - 1)) as p:
        res = [x for x in p.map(one, list(range(SEED, SEED + N_COMM))) if x is not None]
    print(f"done in {time.time()-t0:.1f}s; {len(res)}/{N_COMM} communities valid")
    gain = np.array([x[0] for x in res])
    dshare = np.array([x[2] - x[1] for x in res])
    print(f"\nfocal (made-rhythmic) hosts, B vs A (all-arrhythmic):")
    print(f"  abundance gain   : median {np.median(gain):+.2f}%  "
          f"(IQR {np.percentile(gain,25):+.2f}..{np.percentile(gain,75):+.2f}; {(gain>0).mean()*100:.0f}% positive)")
    print(f"  community share Δ : median {np.median(dshare):+.2f} pp  "
          f"({(dshare>0).mean()*100:.0f}% positive)")

    np.savez(CACHE, gain=gain, dshare=dshare)
    print("cached:", CACHE)
    _plot(gain, dshare)


def _plot(gain, dshare):
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    for tag, vals, col, xlab, unit in (
            ("gain", gain, "#9467bd",
             "rhythmic-host abundance gain (%)", "%"),
            ("share", dshare, "#2ca02c",
             "share of community abundance (pp change)", "pp")):
        fig, ax = plt.subplots(figsize=(3.54, 1.54))       # 90 x 39 mm
        ax.hist(vals, bins=24, color=col, alpha=0.85, lw=0)
        ax.axvline(0, color="k", lw=0.8)
        med = float(np.median(vals))
        fmt = "median %+.1f%%" if unit == "%" else "median %+.2f pp"
        ax.axvline(med, color="#c0392b", ls="--", lw=1.0, label=fmt % med)
        ax.set_xlabel(xlab, fontsize=9, labelpad=2)
        ax.set_ylabel("communities", fontsize=9, labelpad=2)
        ax.tick_params(labelsize=8, pad=1.5)
        ax.legend(fontsize=8, frameon=False, loc="upper left", handlelength=1.4,
                  borderpad=0.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
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
        base = _os.path.join(OUTDIR, "community_model_v2_" + tag)
        for ext in ("png", "pdf"):
            fig.savefig(base + "." + ext, dpi=600, facecolor="white")
        print("figure:", base + ".png")


if __name__ == "__main__":
    import os, sys
    if os.path.exists(CACHE) and "--rerun" not in sys.argv:
        d = np.load(CACHE)
        print("replotting from cache:", CACHE)
        _plot(d["gain"], d["dshare"])
    else:
        main()
