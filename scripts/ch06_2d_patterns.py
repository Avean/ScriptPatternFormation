"""Figure for Chapter 6: patterns on a square (fig:ch06-2d-patterns).

Schnakenberg on (0, L)^2, L = 80, Neumann conditions, d_u = 1, d_v = 20, a = 0.1.
Grid 128 x 128 (cell-centred, 5-point Laplacian), IMEX Euler with dt = 0.05,
initial data (ubar, vbar) plus uniform noise in [-1e-3, 1e-3] (seed 1), T = 2000.

(a) u for b = 0.9 (spots), (b) v for the same run, (c) u for b = 1.5 (stripes).
The value b = 1.5 was found by a small scan (a = 0.1, delta = 20, b = 0.8, 0.9, 1.1,
1.3, 1.4, 1.5; see the text).

Also prints the number of connected components of {u > (max u + min u)/2} at
several times, and the number of unstable modes cos(m pi x/L) cos(n pi y/L).

Produces figures/ch06-2d-patterns.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

import plotstyle as ps
from ch06_solver import schnakenberg, imex_2d, band

N, L = 128, 80.0
DU, DV = 1.0, 20.0
DT, T_END, T_CHUNK = 0.05, 2000.0, 500.0
SEED = 1
U_CMAP = LinearSegmentedColormap.from_list("u", ["#f4f7fb", ps.U_COLOR])
V_CMAP = LinearSegmentedColormap.from_list("v", ["#fbf4f3", ps.V_COLOR])


def schnakenberg_jacobian(a, b):
    s = a + b
    return np.array([[(b - a) / s, s**2], [-2 * b / s, -s**2]])


def components(u):
    lab, n = ndimage.label(u > 0.5 * (u.max() + u.min()))
    return n


def run(b):
    f, g, (ub, vb) = schnakenberg(0.1, b)
    jac = schnakenberg_jacobian(0.1, b)
    lm, lp = band(jac, DU, DV)
    s = 0.1 + b
    dc = s**2 * (s + np.sqrt(2 * b * s)) ** 2 / (b - 0.1) ** 2
    mn = np.arange(0, 40)
    M, Nn = np.meshgrid(mn, mn)
    lam = (np.pi / L) ** 2 * (M**2 + Nn**2)
    n_unstable = int(np.sum((lam > lm) & (lam < lp)))
    print(f"b = {b}: delta_c = {dc:.2f}, band = ({lm:.4f}, {lp:.4f}), "
          f"unstable modes (m, n) on the square: {n_unstable}")
    rng = np.random.default_rng(SEED)
    u = ub + 1e-3 * (2 * rng.random((N, N)) - 1)
    v = vb + 1e-3 * (2 * rng.random((N, N)) - 1)
    t = 0.0
    while t < T_END - 1e-9:
        u_old = u
        u, v = imex_2d(f, g, u, v, DU, DV, L, DT, T_CHUNK)
        t += T_CHUNK
        print(f"   t = {t:6.0f}: u in [{u.min():.3f}, {u.max():.3f}], "
              f"components of {{u > mid}} = {components(u)}, "
              f"max|u(t) - u(t-{T_CHUNK:g})| = {np.abs(u - u_old).max():.2e}")
    return u, v


def show(ax, w, cmap, title, label):
    im = ax.imshow(w, origin="lower", extent=[0, L, 0, L], cmap=cmap,
                   interpolation="nearest", rasterized=True)
    ax.set_title(title)
    ax.set_xticks([0, 40, 80])
    ax.set_yticks([0, 40, 80])
    ax.tick_params(length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(label, rotation=0, labelpad=5)


def main():
    ps.apply()
    u1, v1 = run(0.9)
    u2, v2 = run(1.5)
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.15))
    show(axes[0], u1, U_CMAP, r"(a) $u$, $b=0.9$", r"$u$")
    show(axes[1], v1, V_CMAP, r"(b) $v$, $b=0.9$", r"$v$")
    show(axes[2], u2, U_CMAP, r"(c) $u$, $b=1.5$", r"$u$")
    fig.tight_layout(w_pad=0.8)
    ps.save(fig, "ch06-2d-patterns")


if __name__ == "__main__":
    main()
