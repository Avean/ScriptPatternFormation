"""Figure for Chapter 6: Turing spaces (fig:ch06-turing-space).

(a) Schnakenberg kinetics in the (a, b) plane. With s = a + b:
      (T1)  b - a < s^3,        (T2) always (det A = s^2),
      f_u > 0  iff  b > a,
      delta_c(a, b) = s^2 (s + sqrt(2 b s))^2 / (b - a)^2      (thm:critical-ratio).
    Turing space for delta = 20 (shaded) and its boundary for delta = 10 (dashed);
    the region where the kinetics are unstable (T1 fails) in grey.
(b) Gierer-Meinhardt kinetics, a = 0, in the (b, delta) plane:
      f_u = b > 0, (T1) iff b < 1, delta_c(b) = (1 + sqrt 2)^2 / b.

Produces figures/ch06-turing-space.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps


def schnakenberg_delta_c(a, b):
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return s**2 * (s + np.sqrt(2 * b * s)) ** 2 / (b - a) ** 2


def gm_delta_c(a, b):
    """General formula from thm:critical-ratio for Gierer-Meinhardt (a < 1)."""
    return (1 + a) * (np.sqrt(1 + a) + np.sqrt(2)) ** 2 / (b * (1 - a) ** 2)


def main():
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=ps.FULL)

    # ------------------------------------------------ (a) Schnakenberg
    ax = axes[0]
    a = np.linspace(0, 0.4, 801)
    b = np.linspace(0, 3.0, 1201)
    AA, BB = np.meshgrid(a, b)
    s = AA + BB
    t1 = (BB - AA) < s**3
    fu_pos = BB > AA
    dc = np.where(fu_pos, schnakenberg_delta_c(AA, BB), np.inf)
    turing20 = t1 & fu_pos & (dc < 20)
    unstable_kin = ~t1

    ax.contourf(AA, BB, unstable_kin.astype(float), levels=[0.5, 1.5],
                colors=[ps.LIGHT_GREY])
    ax.contourf(AA, BB, turing20.astype(float), levels=[0.5, 1.5],
                colors=[ps.UNSTABLE_COLOR], alpha=0.35)
    dc_inside = np.where(t1 & fu_pos, dc, np.nan)       # delta_c where (T1), f_u > 0
    ax.contour(AA, BB, dc_inside, levels=[20], colors=[ps.UNSTABLE_COLOR], linewidths=1.0)
    ax.contour(AA, BB, dc_inside, levels=[10], colors=["black"], linewidths=0.9,
               linestyles="--")
    turing10 = t1 & fu_pos & (dc < 10)
    ax.plot([0.1], [0.9], "o", color="black", ms=4)
    ax.annotate(r"$(0.1,\,0.9)$", xy=(0.1, 0.9), xytext=(0.22, 1.3), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.text(0.02, 0.18, "kinetics\nunstable", fontsize=7.5, color=ps.GREY)
    ax.text(0.16, 1.62, r"$\delta_c=20$", fontsize=8, color=ps.UNSTABLE_COLOR)
    ax.text(0.015, 1.4, r"$\delta_c=10$", fontsize=8)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$b$")
    ax.set_title("(a) Schnakenberg")

    # numbers quoted in the text
    print(f"Schnakenberg delta_c(0.1, 0.9) = {schnakenberg_delta_c(0.1, 0.9):.4f}")
    for a0 in (0.0, 0.1):
        # (T1) boundary on the line a = a0: smallest b > a0 with b - a0 = (a0 + b)^3
        bb = np.linspace(a0 + 1e-6, 3, 300001)
        ok = (bb - a0) < (a0 + bb) ** 3
        print(f"  a = {a0}: (T1) holds for b > {bb[np.argmax(ok & (bb > 0.5))]:.4f}")
    col = turing20[:, np.argmin(abs(a - 0.1))]
    print(f"  a = 0.1, delta = 20: Turing for b in ({b[col].min():.3f}, {b[col].max():.3f})")
    col = turing10[:, np.argmin(abs(a - 0.1))]
    print(f"  a = 0.1, delta = 10: Turing for b in ({b[col].min():.3f}, {b[col].max():.3f})")

    # ------------------------------------------------ (b) Gierer-Meinhardt, a = 0
    ax = axes[1]
    bb = np.linspace(0.05, 1.3, 600)
    dc = gm_delta_c(0.0, bb)
    top = 60
    inside = bb < 1
    ax.axvspan(1.0, 1.3, color=ps.LIGHT_GREY, lw=0)
    ax.fill_between(bb[inside], np.minimum(dc[inside], top), top,
                    color=ps.UNSTABLE_COLOR, alpha=0.35, lw=0)
    ax.plot(bb[inside], dc[inside], color=ps.UNSTABLE_COLOR, lw=1.0)
    ax.plot([0.5], [30], "o", color="black", ms=4)
    ax.annotate(r"$(0.5,\,30)$", xy=(0.5, 30), xytext=(0.62, 40), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.plot([0.5], [gm_delta_c(0, 0.5)], "o", mfc="white", color="black", ms=4)
    ax.annotate(r"$\delta_c\approx11.66$", xy=(0.5, gm_delta_c(0, 0.5)),
                xytext=(0.62, 17), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.text(0.2, 50, "DDI", fontsize=8, color=ps.UNSTABLE_COLOR)
    ax.text(1.03, 52, "kinetics\nunstable", fontsize=7.5, color=ps.GREY)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, top)
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(r"$\delta=d_v/d_u$")
    ax.set_title(r"(b) Gierer--Meinhardt, $a=0$".replace("--", "–"))
    print(f"GM a=0: delta_c(b) = {(1 + np.sqrt(2))**2:.4f}/b, delta_c(0.5) = {gm_delta_c(0, 0.5):.4f}")

    fig.tight_layout(w_pad=2.0)
    ps.save(fig, "ch06-turing-space")


if __name__ == "__main__":
    main()
