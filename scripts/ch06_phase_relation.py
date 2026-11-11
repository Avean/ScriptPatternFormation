"""Figure for Chapter 6: in-phase and out-of-phase patterns (fig:ch06-phase-relation).

(a) Schnakenberg, a = 0.1, b = 0.9, d_u = 1, d_v = 20, L = 40 (substrate depletion,
    g_u < 0): stationary pattern, out of phase.
(b) Gierer-Meinhardt, a = 0, b = 0.5, d_u = 1, d_v = 30, L = 20 (activator-inhibitor,
    g_u > 0): stationary pattern, in phase. On (0, 20) the band (lambda_-, lambda_+)
    = (0.0390, 0.4277) contains lambda_k = (k pi/20)^2 exactly for k = 2, 3, 4.

Both: N = 10 L cells, IMEX Euler, dt = 0.01, T = 600, noise 1e-3 (seed 2), and the
plotted quantities are (u - ubar)/max|u - ubar| and (v - vbar)/max|v - vbar|.

Produces figures/ch06-phase-relation.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps
from ch06_solver import (schnakenberg, gierer_meinhardt, imex_1d, grid_1d, band,
                         mode_rates)

SEED = 2
CASES = [
    # title, kinetics, Jacobian, d_v, L
    ("(a) Schnakenberg: out of phase", schnakenberg(0.1, 0.9),
     np.array([[0.8, 1.0], [-1.8, -1.0]]), 20.0, 40.0),
    ("(b) Gierer–Meinhardt: in phase", gierer_meinhardt(0.0, 0.5),
     np.array([[0.5, -0.25], [4.0, -1.0]]), 30.0, 20.0),
]


def main():
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 1.55))
    for ax, (title, (f, g, (ub, vb)), jac, dv, L) in zip(axes, CASES):
        lm, lp = band(jac, 1.0, dv)
        k = np.arange(0, 15)
        mu = mode_rates(jac, 1.0, dv, (k * np.pi / L) ** 2)
        print(f"{title}: band = ({lm:.4f}, {lp:.4f}), L_min = {np.pi / np.sqrt(lp):.3f}, "
              f"unstable k = {k[mu > 0]}, fastest k = {k[np.argmax(mu)]} (mu = {mu.max():.4f})")
        N = int(10 * L)
        x = grid_1d(L, N)
        rng = np.random.default_rng(SEED)
        u0 = ub + 1e-3 * (2 * rng.random(N) - 1)
        v0 = vb + 1e-3 * (2 * rng.random(N) - 1)
        u, v, snaps = imex_1d(f, g, u0, v0, 1.0, dv, L, 0.01, 600.0, every=10.0)
        vmin = min(s[2].min() for s in snaps)
        print(f"   final u in [{u.min():.3f}, {u.max():.3f}], v in [{v.min():.3f}, {v.max():.3f}], "
              f"min v over the run = {vmin:.3f}, "
              f"max|u(600)-u(590)| = {np.abs(snaps[-1][1] - snaps[-2][1]).max():.1e}")
        wu = (u - ub) / np.abs(u - ub).max()
        wv = (v - vb) / np.abs(v - vb).max()
        ax.plot(x, wu, color=ps.U_COLOR, label=r"$u$")
        ax.plot(x, wv, color=ps.V_COLOR, label=r"$v$")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlim(0, L)
        ax.set_ylim(-1.15, 1.85)
        ax.set_xlabel(r"$x$")
        ax.set_title(title)
        ax.legend(loc="upper center", ncol=2, borderaxespad=0.1)
    fig.tight_layout(w_pad=2.0)
    ps.save(fig, "ch06-phase-relation")


if __name__ == "__main__":
    main()
