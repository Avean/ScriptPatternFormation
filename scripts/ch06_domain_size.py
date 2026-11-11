"""Figure for Chapter 6: the role of the domain size (fig:ch06-domain-size).

Schnakenberg, a = 0.1, b = 0.9, d_u = 1, d_v = 20 (band (lambda_-, lambda_+) =
(0.0740, 0.6760)).

(a) The (L, k) plane: mode k is unstable on (0, L) iff L sqrt(lambda_-)/pi < k <
    L sqrt(lambda_+)/pi (cor:domain-size). The dotted line k = L sqrt(lambda_max)/pi
    marks the fastest-growing wavenumber of the dispersion relation.
(b) Final patterns of u for L = 3, 10, 40, 80 drawn on a common x-axis. N = 10 L
    cells, IMEX Euler dt = 0.01, noise 1e-3 (seed 2), T = 600 (T = 2000 for L = 80).

Produces figures/ch06-domain-size.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import plotstyle as ps
from ch06_solver import schnakenberg, imex_1d, grid_1d, band, mode_rates, cosine_coefficients

JAC = np.array([[0.8, 1.0], [-1.8, -1.0]])
DU, DV = 1.0, 20.0
SEED = 2
LENGTHS = [(3.0, 600.0), (10.0, 600.0), (40.0, 600.0), (80.0, 2000.0)]


def main():
    ps.apply()
    f, g, (ub, vb) = schnakenberg(0.1, 0.9)
    lm, lp = band(JAC, DU, DV)
    res = minimize_scalar(lambda l: -mode_rates(JAC, DU, DV, [l])[0], bounds=(lm, lp),
                          method="bounded")
    lmax = res.x
    print(f"band ({lm:.4f}, {lp:.4f}); L_min = pi/sqrt(lambda_+) = {np.pi / np.sqrt(lp):.4f}; "
          f"fastest lambda = {lmax:.4f}, mu = {-res.fun:.4f}, "
          f"wavelength 2 pi/sqrt(lambda) = {2 * np.pi / np.sqrt(lmax):.2f}")

    fig = plt.figure(figsize=(6.2, 2.35))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.35], wspace=0.28, hspace=0.25)

    # ---------------------------------------------------------- (a) (L, k) plane
    ax = fig.add_subplot(gs[:, 0])
    Lg = np.linspace(0, 85, 200)
    ax.fill_between(Lg, Lg * np.sqrt(lm) / np.pi, Lg * np.sqrt(lp) / np.pi,
                    color=ps.UNSTABLE_COLOR, alpha=0.18, lw=0)
    ax.plot(Lg, Lg * np.sqrt(lp) / np.pi, color=ps.UNSTABLE_COLOR, lw=1.0)
    ax.plot(Lg, Lg * np.sqrt(lm) / np.pi, color=ps.UNSTABLE_COLOR, lw=1.0)
    ax.plot(Lg, Lg * np.sqrt(lmax) / np.pi, ":", color="black", lw=0.8)
    for L, _ in LENGTHS:
        k = np.arange(1, 25)
        inside = (k > L * np.sqrt(lm) / np.pi) & (k < L * np.sqrt(lp) / np.pi)
        ax.plot(np.full(inside.sum(), L), k[inside], "o", color=ps.UNSTABLE_COLOR, ms=2.2)
        ax.axvline(L, color=ps.GREY, lw=0.4, zorder=0)
        print(f"L = {L:g}: unstable modes k = {k[inside]}")
    Lmin = np.pi / np.sqrt(lp)
    ax.plot([Lmin], [1], "o", mfc="white", color="black", ms=4, zorder=5)
    ax.annotate(r"$L_{\min}\approx3.82$", xy=(Lmin, 1), xytext=(19, 0.9), fontsize=8,
                va="center", arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.text(28, 21, r"$k=L\sqrt{\lambda_+}/\pi$", fontsize=8, color=ps.UNSTABLE_COLOR)
    ax.text(57, 3.2, r"$k=L\sqrt{\lambda_-}/\pi$", fontsize=8, color=ps.UNSTABLE_COLOR)
    ax.set_xlim(0, 85)
    ax.set_ylim(0, 24)
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$k$")
    ax.set_title("(a) unstable modes")

    # ---------------------------------------------------------- (b) final patterns
    axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    for i, (ax, (L, T)) in enumerate(zip(axes, LENGTHS)):
        N = int(10 * L)
        x = grid_1d(L, N)
        rng = np.random.default_rng(SEED)
        u0 = ub + 1e-3 * (2 * rng.random(N) - 1)
        v0 = vb + 1e-3 * (2 * rng.random(N) - 1)
        u, v, snaps = imex_1d(f, g, u0, v0, DU, DV, L, 0.01, T, every=10.0)
        c = cosine_coefficients(u - ub)
        up = np.concatenate(([u[1]], u, [u[-2]]))
        peaks = x[(up[1:-1] > up[:-2]) & (up[1:-1] > up[2:])]
        print(f"L = {L:g}, T = {T:g}: max|u - ubar| = {np.abs(u - ub).max():.2e}, "
              f"dominant mode {np.argmax(np.abs(c[1:])) + 1}, peaks at {np.round(peaks, 2)}, "
              f"max|u(T)-u(T-10)| = {np.abs(snaps[-1][1] - snaps[-2][1]).max():.1e}")
        ax.plot(x, u, color=ps.U_COLOR, lw=1.1)
        ax.plot([0, L], [ub, ub], color=ps.GREY, lw=0.6, ls=":")
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 2.4)
        ax.set_yticks([0, 2])
        ax.text(81, 1.2, rf"$L={L:g}$", fontsize=8, va="center", ha="left")
        if i < 3:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_title(r"(b) stationary $u$ for four lengths $L$")
    axes[-1].set_xlabel(r"$x$")
    ps.save(fig, "ch06-domain-size")


if __name__ == "__main__":
    main()
