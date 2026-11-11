"""Figure for Chapter 4: phase plane of the stationary problem d U'' + f(U) = 0.

Bistable kinetics f(u) = u(1-u)(u-1/2), d = 1, interval length L = 8.

The stationary equation is conservative: E(U, V) = (d/2) V^2 + F(U), with
F(u) = int_0^u f = -u^4/4 + u^3/2 - u^2/4, is constant along orbits (V = U').
A Neumann stationary solution is an orbit piece that starts and ends on the axis
V = 0 and needs "time" L. We compute the increasing one by shooting:
U(0) = U0, U'(0) = 0, and choose U0 in (0, 1/2) such that U' vanishes again
exactly at x = L.

Checks printed by the script:
  * U0 (about 0.147) and U(L) = 1 - U0 (symmetry),
  * the largest eigenvalue of the Neumann linearization d phi'' + f'(U) phi
    (finite differences), which is positive (about 0.081), in agreement with
    thm:nonconstant-unstable.

Produces figures/ch04-phase-plane.pdf (fig:ch04-phase-plane).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import plotstyle as ps

D = 1.0
L = 8.0


def f(u):
    return u * (1 - u) * (u - 0.5)


def fprime(u):
    return -3 * u**2 + 3 * u - 0.5


def F(u):
    return -u**4 / 4 + u**3 / 2 - u**2 / 4


def rhs(x, y):
    return [y[1], -f(y[0]) / D]


def first_return(U0):
    """Smallest x > 0 at which U' = 0 again, starting from (U0, 0)."""
    ev = lambda x, y: y[1]
    ev.terminal = True
    ev.direction = -1
    sol = solve_ivp(rhs, [1e-9, 500.0], [U0, 1e-15], events=ev,
                    rtol=1e-12, atol=1e-13, max_step=0.05)
    return sol.t_events[0][0]


def stationary_solution():
    U0 = brentq(lambda s: first_return(s) - L, 1e-3, 0.49, xtol=1e-14)
    sol = solve_ivp(rhs, [0.0, L], [U0, 0.0], dense_output=True,
                    rtol=1e-12, atol=1e-13)
    return U0, sol


def top_eigenvalue(U, x):
    """Largest eigenvalue of d phi'' + f'(U) phi with Neumann conditions (FD)."""
    n = len(x) - 1
    h = x[1] - x[0]
    M = np.diag(-2 * D / h**2 + fprime(U))
    M += np.diag(np.full(n, D / h**2), 1) + np.diag(np.full(n, D / h**2), -1)
    M[0, 1] = M[n, n - 1] = 2 * D / h**2
    return np.sort(np.linalg.eigvals(M).real)[-1]


def main():
    ps.apply()
    U0, sol = stationary_solution()
    x = np.linspace(0.0, L, 2001)
    U, V = sol.sol(x)
    mu = top_eigenvalue(U, x)
    print(f"U0 = {U0:.6f}, U(L) = {U[-1]:.6f}, max U' = {V.max():.4f}, "
          f"top eigenvalue of linearization = {mu:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL[0], 2.3),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # (a) phase plane
    ax = axes[0]
    Ug, Vg = np.meshgrid(np.linspace(-0.25, 1.25, 601), np.linspace(-0.3, 0.3, 601))
    E = 0.5 * D * Vg**2 + F(Ug)
    E_sep = 0.0                        # level of the saddles (0,0) and (1,0)
    E_c = F(0.5)                       # level of the center (1/2, 0): -1/64
    E_orb = F(U0)                      # level of the highlighted orbit
    inner = [E_c + 0.15 * (E_sep - E_c), E_c + 0.45 * (E_sep - E_c), E_orb]
    outer = [-0.004, 0.004, 0.012]
    ax.contour(Ug, Vg, E, levels=sorted(inner + outer), colors=ps.GREY,
               linewidths=0.6, linestyles="solid")
    ax.contour(Ug, Vg, E, levels=[E_sep], colors=ps.GREY, linewidths=0.9,
               linestyles="--")
    ax.plot(U, V, color=ps.U_COLOR, lw=2.0)
    i = len(x) // 2
    ax.annotate("", xy=(U[i + 40], V[i + 40]), xytext=(U[i], V[i]),
                arrowprops=dict(arrowstyle="-|>", color=ps.U_COLOR, lw=1.2))
    ax.plot([0, 1], [0, 0], "o", color="black", ms=3.5)
    ax.plot([0.5], [0], "o", mfc="white", mec="black", ms=3.5)
    ax.plot([U0, 1 - U0], [0, 0], "s", color=ps.U_COLOR, ms=3.5)
    box = dict(boxstyle="square,pad=0.1", fc="white", ec="none")
    ax.text(U0 - 0.02, 0.025, r"$x=0$", fontsize=8, color=ps.U_COLOR,
            ha="right", bbox=box)
    ax.text(1 - U0 + 0.02, 0.025, r"$x=L$", fontsize=8, color=ps.U_COLOR,
            ha="left", bbox=box)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$U'$")
    ax.set_title("(a) phase plane")

    # (b) profile
    ax = axes[1]
    for c in (0.0, 0.5, 1.0):
        ax.axhline(c, color=ps.GREY, lw=0.6, ls=":")
    ax.plot(x, U, color=ps.U_COLOR)
    ax.set_xlim(0, L)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$U(x)$")
    ax.set_title(rf"(b) stationary solution, $L={L:.0f}$")

    fig.tight_layout()
    ps.save(fig, "ch04-phase-plane")


if __name__ == "__main__":
    main()
