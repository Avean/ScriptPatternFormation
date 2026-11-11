"""Figures for Chapter 5: dispersion relation and the band of unstable modes.

Schnakenberg kinetics with the standard parameters a = 0.1, b = 0.9, so that
A = [[0.8, 1], [-1.8, -1]], tr A = -0.2, det A = 1; d_u = 1; L = 40.

Produces
  figures/ch05-dispersion.pdf   (fig:ch05-dispersion)
  figures/ch05-band.pdf         (fig:ch05-band)
"""

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps

A = np.array([[0.8, 1.0], [-1.8, -1.0]])
FU, FV, GU, GV = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
DET = np.linalg.det(A)
DU = 1.0
L = 40.0
DELTA_C = (np.sqrt(DET) + np.sqrt(-FV * GU)) ** 2 / FU**2   # ~ 8.5676
DELTAS = [5.0, DELTA_C, 20.0]
LABELS = [r"$\delta=5$", r"$\delta=\delta_c\approx 8.57$", r"$\delta=20$"]
COLORS = [ps.STABLE_COLOR, ps.GREY, ps.UNSTABLE_COLOR]
STYLES = ["-", "--", "-"]


def growth_rate(lam, delta):
    """Largest real part of the eigenvalues of A - lam*D, D = diag(DU, delta*DU)."""
    D = np.diag([DU, delta * DU])
    return np.array([np.linalg.eigvals(A - l * D).real.max() for l in np.atleast_1d(lam)])


def h(lam, delta):
    du, dv = DU, delta * DU
    return du * dv * lam**2 - (dv * FU + du * GV) * lam + DET


def dispersion_figure():
    lam = np.linspace(0.0, 1.2, 1201)
    fig, axes = plt.subplots(1, 2, figsize=ps.FULL)

    ax = axes[0]
    for delta, lab, col, sty in zip(DELTAS, LABELS, COLORS, STYLES):
        ax.plot(lam, growth_rate(lam, delta), sty, color=col, label=lab)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\max\,\mathrm{Re}\,\mu$")
    ax.set_ylim(-0.8, 0.35)
    ax.set_title("(a) dispersion relation")
    ax.legend(loc="upper right")

    ax = axes[1]
    delta = 20.0
    ax.plot(lam, growth_rate(lam, delta), color=ps.UNSTABLE_COLOR, lw=1.0, alpha=0.6)
    k = np.arange(0, 16)
    lam_k = (k * np.pi / L) ** 2
    mu_k = growth_rate(lam_k, delta)
    unstable = mu_k > 0
    ax.plot(lam_k[unstable], mu_k[unstable], "o", color=ps.UNSTABLE_COLOR, ms=4,
            label=r"unstable modes $k=4,\dots,10$")
    ax.plot(lam_k[~unstable], mu_k[~unstable], "o", mfc="white", color=ps.GREY, ms=4,
            label="stable modes")
    k_fast = k[np.argmax(mu_k)]
    ax.annotate(rf"$k={k_fast}$", xy=(lam_k[k_fast], mu_k[k_fast]),
                xytext=(lam_k[k_fast] + 0.12, mu_k[k_fast] + 0.03),
                arrowprops=dict(arrowstyle="-", lw=0.6), fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(r"$\lambda$")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.8, 0.35)
    ax.set_title(rf"(b) $\delta=20$, modes $\lambda_k=(k\pi/L)^2$, $L={L:.0f}$")
    ax.legend(loc="lower left")

    ps.save(fig, "ch05-dispersion")


def band_figure():
    lam = np.linspace(0.0, 1.0, 1001)
    fig, ax = plt.subplots(figsize=ps.WIDE)
    for delta, lab, col, sty in zip(DELTAS, LABELS, COLORS, STYLES):
        ax.plot(lam, h(lam, delta), sty, color=col, label=lab)
    delta = 20.0
    dv = delta * DU
    p = dv * FU + DU * GV
    disc = p**2 - 4 * DU * dv * DET
    lam_m, lam_p = (p - np.sqrt(disc)) / (2 * DU * dv), (p + np.sqrt(disc)) / (2 * DU * dv)
    ax.axvspan(lam_m, lam_p, color=ps.UNSTABLE_COLOR, alpha=0.12, lw=0)
    ax.text((lam_m + lam_p) / 2, 3.2, "band of unstable\n" + r"$\lambda$ for $\delta=20$",
            ha="center", va="top", fontsize=8, color=ps.UNSTABLE_COLOR)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$h(\lambda)=\det(A-\lambda D)$")
    ax.set_ylim(-2.2, 3.5)
    ax.legend(loc="lower right")
    ps.save(fig, "ch05-band")
    print(f"delta_c = {DELTA_C:.4f}, band for delta=20: ({lam_m:.4f}, {lam_p:.4f})")


if __name__ == "__main__":
    ps.apply()
    dispersion_figure()
    band_figure()
