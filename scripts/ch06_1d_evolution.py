"""Figure for Chapter 6: the Turing instability in one dimension (fig:ch06-1d-evolution).

Schnakenberg kinetics, a = 0.1, b = 0.9, d_u = 1, d_v = 20, L = 40 (ex:schnakenberg-turing).
Grid: N = 400 cells (h = 0.1), IMEX Euler with dt = 0.01 (see ch06_solver.py).
Initial data: (ubar, vbar) plus independent uniform noise in [-1e-3, 1e-3] at every
grid point, random seed SEED.

Panels: (a) space-time plot of u; (b) cosine coefficients |hat u_k(t)| on a log
scale with the slopes predicted by linear theory; (c) the profile in the linear
phase compared with cos(6 pi x / L); (d) the final stationary pattern.

Also prints: fitted growth rates vs linear prediction, and the statistics of the
final pattern over 50 random seeds.

Produces figures/ch06-1d-evolution.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.fft import dct, idct
from scipy.linalg import expm

import plotstyle as ps
from ch06_solver import (schnakenberg, imex_1d, grid_1d, cosine_coefficients,
                         mode_rates, discrete_eigs_1d)

A, B = 0.1, 0.9
DU, DV, L, N = 1.0, 20.0, 40.0, 400
DT = 0.01
SEED = 2
JAC = np.array([[0.8, 1.0], [-1.8, -1.0]])
T_EARLY = 25.0          # time of the "linear phase" profile in panel (c)
FIT_WINDOW = (10.0, 20.0)

U_CMAP = LinearSegmentedColormap.from_list("u", ["#f4f7fb", ps.U_COLOR])


def initial_data(seed, x, ub, vb):
    rng = np.random.default_rng(seed)
    n = len(x)
    return ub + 1e-3 * (2 * rng.random(n) - 1), vb + 1e-3 * (2 * rng.random(n) - 1)


def linear_prediction(w1, w2, t):
    """Solution at time t of the linearized (semi-discrete) problem w' = D Delta_h w + A w,
    computed mode by mode: hat w_k(t) = exp(t (A - lambda_k^h D)) hat w_k(0)."""
    c1 = dct(w1, type=2, norm="ortho")
    c2 = dct(w2, type=2, norm="ortho")
    lam = -discrete_eigs_1d(L, len(w1))
    out1, out2 = np.empty_like(c1), np.empty_like(c2)
    for k, lk in enumerate(lam):
        y = expm(t * (JAC - lk * np.diag([DU, DV]))) @ np.array([c1[k], c2[k]])
        out1[k], out2[k] = y
    return idct(out1, type=2, norm="ortho"), idct(out2, type=2, norm="ortho")


def dominant_mode(u, ub):
    c = cosine_coefficients(u - ub)
    k = int(np.argmax(np.abs(c[1:40])) + 1)
    return k, int(np.sign(c[k]))


def main():
    ps.apply()
    f, g, (ub, vb) = schnakenberg(A, B)
    x = grid_1d(L, N)
    u0, v0 = initial_data(SEED, x, ub, vb)

    # long run with snapshots every 0.5 time units
    T_END = 600.0
    u, v, snaps = imex_1d(f, g, u0, v0, DU, DV, L, DT, T_END, every=0.5)
    t = np.array([s[0] for s in snaps])
    U = np.array([s[1] for s in snaps])
    C = np.array([cosine_coefficients(s - ub) for s in U])
    amp = np.abs(U - ub).max(axis=1)

    # ---- numbers for the text
    k_all = np.arange(1, 13)
    mu = mode_rates(JAC, DU, DV, (k_all * np.pi / L) ** 2)
    m = (t >= FIT_WINDOW[0]) & (t <= FIT_WINDOW[1])
    fitted = {k: np.polyfit(t[m], np.log(np.abs(C[m, k])), 1)[0] for k in k_all}
    print("mode  predicted mu_k  fitted rate (t in [%g, %g])" % FIT_WINDOW)
    for k, mk in zip(k_all, mu):
        print(f"  k={k:2d}   {mk:+.4f}        {fitted[k]:+.4f}")
    for tt in (10, 20, 25, 30, 40, 50):
        i = np.argmin(abs(t - tt))
        share = C[i, 6] ** 2 / np.sum(C[i, 1:] ** 2)
        print(f"  t={tt}: max|u-ubar| = {amp[i]:.4f}, share of mode 6 in ||u-ubar||^2 = {share:.2f}")
    # stationarity: change of u over the last 10 time units
    i10 = np.argmin(abs(t - (T_END - 10)))
    print(f"  max|u(T)-u(T-10)| = {np.abs(U[-1] - U[i10]).max():.2e}")
    for tt in (50, 100, 200):
        i = np.argmin(abs(t - tt))
        print(f"  max|u({tt}) - u({T_END:g})| = {np.abs(U[i] - U[-1]).max():.2e}")
    print(f"  final: u in [{u.min():.3f}, {u.max():.3f}], v in [{v.min():.3f}, {v.max():.3f}],"
          f" dominant mode {dominant_mode(u, ub)}")
    up = np.concatenate(([u[1]], u, [u[-2]]))
    peaks = x[(up[1:-1] > up[:-2]) & (up[1:-1] > up[2:])]
    print(f"  peaks of u at x = {np.round(peaks, 2)}")
    vp = np.concatenate(([v[1]], v, [v[-2]]))
    print(f"  peaks of v at x = {np.round(x[(vp[1:-1] > vp[:-2]) & (vp[1:-1] > vp[2:])], 2)}")

    # ---- figure
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 3.8))

    ax = axes[0, 0]
    tmax = 150.0
    sel = t <= tmax
    im = ax.imshow(U[sel], origin="lower", aspect="auto", cmap=U_CMAP,
                   extent=[0, L, 0, tmax], interpolation="nearest", rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.06)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(r"$u$", rotation=0, labelpad=6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"(a) $u(x,t)$")

    ax = axes[0, 1]
    styles = {4: (ps.GREY, ":"), 6: (ps.UNSTABLE_COLOR, "-"), 7: (ps.U_COLOR, "-"),
              9: (ps.STABLE_COLOR, "-")}
    tt = t[t <= 70]
    for k, (col, ls) in styles.items():
        ax.semilogy(tt, np.abs(C[t <= 70, k]), ls, color=col, lw=1.1, label=rf"$k={k}$")
        # reference line with the predicted slope mu_k, through the fit at t = 15
        i15 = np.argmin(abs(t - 15))
        ref = np.abs(C[i15, k]) * np.exp(mu[k - 1] * (tt - 15))
        mref = (tt >= 0) & (tt <= 35)
        ax.semilogy(tt[mref], ref[mref], "--", color="black", lw=0.6)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\hat u_k(t)|$")
    ax.set_ylim(1e-6, 3)
    ax.set_title("(b) cosine coefficients")
    ax.legend(loc="lower right", ncol=2, handlelength=1.5, columnspacing=1.0)

    ax = axes[1, 0]
    i = np.argmin(abs(t - T_EARLY))
    w = U[i] - ub
    wlin, _ = linear_prediction(u0 - ub, v0 - vb, T_EARLY)
    scale = np.abs(w).max()
    print(f"  t={T_EARLY:g}: max|u - u_linear| / max|u - ubar| = {np.abs(w - wlin).max() / scale:.3f}")
    s = np.sign(C[i, 6])
    ax.plot(x, w / scale, color=ps.U_COLOR, label="simulation")
    ax.plot(x, wlin / scale, "--", color="black", lw=0.8, label="linear theory")
    ax.plot(x, s * np.cos(6 * np.pi * x / L), ":", color=ps.UNSTABLE_COLOR, lw=1.1,
            label=r"$-\cos(6\pi x/L)$" if s < 0 else r"$\cos(6\pi x/L)$")
    ax.set_ylim(-1.15, 1.75)
    ax.set_xlim(0, L)
    ax.set_xlabel(r"$x$")
    ax.set_title(rf"(c) $(u-\bar u)/\max|u-\bar u|$ at $t={T_EARLY:g}$")
    ax.legend(loc="upper center", ncol=3, handlelength=1.3, columnspacing=0.6, fontsize=7,
              borderaxespad=0.1)

    ax = axes[1, 1]
    ax.plot(x, u, color=ps.U_COLOR, label=r"$u$")
    ax.plot(x, v, color=ps.V_COLOR, label=r"$v$")
    ax.axhline(ub, color=ps.U_COLOR, lw=0.6, ls=":")
    ax.axhline(vb, color=ps.V_COLOR, lw=0.6, ls=":")
    ax.set_xlim(0, L)
    ax.set_ylim(0, 2.4)
    ax.set_xlabel(r"$x$")
    ax.set_title(rf"(d) stationary pattern ($t={T_END:g}$)")
    ax.legend(loc="upper right", ncol=2)

    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    ps.save(fig, "ch06-1d-evolution")

    # ---- statistics over seeds
    counts = {}
    for seed in range(50):
        a0, b0 = initial_data(seed, x, ub, vb)
        uf, _ = imex_1d(f, g, a0, b0, DU, DV, L, DT, 600.0)
        k, _ = dominant_mode(uf, ub)
        counts[k] = counts.get(k, 0) + 1
    print(f"final dominant mode over 50 seeds (dt = {DT}):", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
