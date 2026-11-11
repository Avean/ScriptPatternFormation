"""Figure for Chapter 2: the trace-determinant diagram of a real 2x2 matrix.

Marks the Jacobian matrices of the two running examples at their standard
parameters:
  Schnakenberg      a = 0.1, b = 0.9:  A = [[0.8, 1], [-1.8, -1]],   tr A = -0.2, det A = 1
  Gierer-Meinhardt  a = 0,   b = 0.5:  A = [[0.5, -0.25], [4, -1]],  tr A = -0.5, det A = 0.5

Produces
  figures/ch02-trace-det.pdf   (fig:ch02-trace-det)
"""

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps

TR_MAX = 2.2
DET_MIN, DET_MAX = -0.85, 1.75

SCHNAKENBERG = np.array([[0.8, 1.0], [-1.8, -1.0]])
GIERER_MEINHARDT = np.array([[0.5, -0.25], [4.0, -1.0]])


def trace_det_figure():
    fig, ax = plt.subplots(figsize=(ps.WIDE[0], 2.45))   # WIDE width, a little lower
    tr = np.linspace(-TR_MAX, TR_MAX, 801)

    # regions: stable quadrant (tr < 0, det > 0) and saddles (det < 0)
    ax.fill_between(tr[tr <= 0], 0, DET_MAX, color=ps.STABLE_COLOR, alpha=0.10, lw=0)
    ax.fill_between(tr, DET_MIN, 0, color=ps.UNSTABLE_COLOR, alpha=0.08, lw=0)

    # boundaries: det = 0, tr = 0 (for det > 0), parabola det = tr^2/4
    ax.plot(tr, np.zeros_like(tr), color="black", lw=0.7)
    ax.plot([0, 0], [0, DET_MAX], color="black", lw=0.7)
    ax.plot(tr, tr**2 / 4, color=ps.GREY, lw=1.1)
    ax.annotate(r"$\det B=\frac{1}{4}(\mathrm{tr}\,B)^2$", xy=(1.3, 1.3**2 / 4),
                xytext=(0.62, 0.85), ha="center", va="center", fontsize=8, color=ps.GREY,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.GREY))

    # region labels
    kw = dict(fontsize=8, ha="center", va="center")
    ax.text(-1.45, 1.38, "stable focus", color=ps.STABLE_COLOR, **kw)
    ax.text(-1.86, 0.36, "stable\nnode", color=ps.STABLE_COLOR, **kw)
    ax.text(1.25, 1.38, "unstable focus", color=ps.UNSTABLE_COLOR, **kw)
    ax.text(1.86, 0.36, "unstable\nnode", color=ps.UNSTABLE_COLOR, **kw)
    ax.text(0.0, -0.45, "saddle", color=ps.UNSTABLE_COLOR, **kw)

    # the two running examples
    for A, label, xytext, ha in [
        (SCHNAKENBERG, "Schnakenberg", (-0.62, 1.02), "right"),
        (GIERER_MEINHARDT, "Gierer–Meinhardt", (-0.62, 0.66), "right"),
    ]:
        t, d = np.trace(A), np.linalg.det(A)
        ax.plot(t, d, "o", color=ps.U_COLOR, ms=4, zorder=5)
        ax.annotate(label, xy=(t, d), xytext=xytext, ha=ha, va="center",
                    fontsize=8, color=ps.U_COLOR,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.U_COLOR))
        print(f"{label}: tr = {t:.3f}, det = {d:.3f}, tr^2/4 = {t**2 / 4:.4f}")

    ax.set_xlim(-TR_MAX, TR_MAX)
    ax.set_ylim(DET_MIN, DET_MAX)
    ax.set_xlabel(r"$\mathrm{tr}\,B$")
    ax.set_ylabel(r"$\det B$")
    ps.save(fig, "ch02-trace-det")


if __name__ == "__main__":
    ps.apply()
    trace_det_figure()
