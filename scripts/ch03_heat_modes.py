"""Figure for Chapter 3: Neumann eigenfunctions and the heat equation in cosine modes.

(a) The first Neumann eigenfunctions phi_k(x) = cos(k pi x / L), k = 0, ..., 3.
(b) Solution of u_t = d u_xx on (0, L) with u_x = 0 at x = 0, L, computed from the
    cosine series  u(x,t) = sum_k exp(-d lambda_k t) hat u_k(0) cos(k pi x / L),
    lambda_k = (k pi / L)^2, for a rough initial profile (step plus small-scale
    noise). The dashed line is the mean value M = hat u_0(0).

Parameters: L = 1, d = 1.

Produces
  figures/ch03-heat-modes.pdf   (fig:ch03-heat-modes)
"""

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps

L = 1.0
D = 1.0
K_MAX = 4000                     # number of cosine modes used for t > 0

# Initial profile: u0 = HIGH on (0, A), LOW on (A, L), plus "noise" made of the
# cosine modes 12..40 with small random amplitudes.
A, HIGH, LOW = 0.35 * L, 1.0, 0.2
NOISE_MODES = np.arange(12, 41)
NOISE_AMP = 0.035
TIMES = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.2]
TIME_LABELS = ["0", r"10^{-4}", r"10^{-3}", r"10^{-2}", "0.05", "0.2"]


def step_coefficients():
    """Exact cosine coefficients of the step, k = 0..K_MAX (convention of the notes)."""
    c = np.zeros(K_MAX + 1)
    c[0] = (HIGH * A + LOW * (L - A)) / L
    k = np.arange(1, K_MAX + 1)
    c[1:] = 2.0 * (HIGH - LOW) * np.sin(k * np.pi * A / L) / (k * np.pi)
    return c


def noise_coefficients():
    c = np.zeros(K_MAX + 1)
    rng = np.random.default_rng(3)
    c[NOISE_MODES] = NOISE_AMP * rng.uniform(-1.0, 1.0, size=NOISE_MODES.size)
    return c


def cosine_sum(x, coeff):
    k = np.arange(coeff.size)
    return np.cos(np.outer(x, k) * np.pi / L) @ coeff


def heat_solution(x, t, coeff):
    k = np.arange(coeff.size)
    return cosine_sum(x, np.exp(-D * (k * np.pi / L) ** 2 * t) * coeff)


def main():
    ps.apply()
    x = np.linspace(0.0, L, 1201)
    fig, axes = plt.subplots(1, 2, figsize=ps.FULL)

    # (a) eigenfunctions
    ax = axes[0]
    colors = [ps.GREY, ps.U_COLOR, ps.V_COLOR, ps.STABLE_COLOR]
    styles = ["--", "-", "-", "-"]
    for k, col, sty in zip(range(4), colors, styles):
        ax.plot(x, np.cos(k * np.pi * x / L), sty, color=col, label=rf"$\varphi_{k}$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks([0, L / 2, L])
    ax.set_xticklabels(["$0$", "$L/2$", "$L$"])
    ax.set_xlabel(r"$x$")
    ax.set_ylim(-1.15, 1.55)
    ax.set_yticks([-1, 0, 1])
    ax.set_title(r"(a) $\varphi_k(x)=\cos(k\pi x/L)$")
    ax.legend(loc="upper center", ncol=4, handlelength=1.6, columnspacing=1.0)

    # (b) heat equation from a rough initial profile
    ax = axes[1]
    c_step, c_noise = step_coefficients(), noise_coefficients()
    c = c_step + c_noise
    M = c[0]
    u0 = np.where(x < A, HIGH, LOW) + cosine_sum(x, c_noise)   # exact, no Gibbs ringing
    shades = plt.cm.Blues(np.linspace(0.45, 1.0, len(TIMES) - 1))
    ax.plot(x, u0, color=ps.GREY, lw=0.8, label=r"$t=0$")
    for t, lab, col in zip(TIMES[1:], TIME_LABELS[1:], shades):
        ax.plot(x, heat_solution(x, t, c), color=col, label=rf"$t={lab}$")
    ax.axhline(M, color="black", ls="--", lw=0.8)
    ax.text(0.985 * L, M + 0.03, r"$M$", ha="right", va="bottom", fontsize=8)
    ax.set_xticks([0, L / 2, L])
    ax.set_xticklabels(["$0$", "$L/2$", "$L$"])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")
    ax.set_ylim(0.0, 1.2)
    ax.set_title(r"(b) heat equation, $d=1$, $L=1$")
    ax.legend(loc="upper right", fontsize=7, handlelength=1.4, labelspacing=0.25)

    ps.save(fig, "ch03-heat-modes")

    # numbers that may be quoted in the caption / text
    print(f"M = {M:.4f}")
    for t in TIMES[1:]:
        u = heat_solution(x, t, c)
        print(f"t = {t:g}: max|u - M| = {np.abs(u - M).max():.4f}, "
              f"noise factor exp(-d lambda_12 t) = {np.exp(-D * (12 * np.pi / L)**2 * t):.2e}, "
              f"exp(-d lambda_1 t) = {np.exp(-D * np.pi**2 * t / L**2):.4f}")


if __name__ == "__main__":
    main()
