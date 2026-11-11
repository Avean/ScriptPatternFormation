"""Shared numerics for the Chapter 6 figure scripts (not a figure script itself).

Spatial discretization (Section "A numerical method"):
  cell-centred grid x_j = (j - 1/2) h, j = 1..N, h = L/N, Neumann condition by
  reflected ghost values u_0 = u_1, u_{N+1} = u_N, and the three-point Laplacian
      (Delta_h u)_j = (u_{j+1} - 2 u_j + u_{j-1}) / h^2 .
  The vectors cos(k pi x_j / L), k = 0..N-1, are eigenvectors of Delta_h with
  eigenvalues -(4/h^2) sin^2(k pi h / (2L)).  They are exactly the basis vectors
  of the type-II discrete cosine transform, so scipy.fft.dct diagonalizes Delta_h.
  In 2D (square, Neumann) the same holds for dctn and the 5-point Laplacian.

Time stepping:
  * euler_1d: explicit Euler (the scheme of the listing in the text);
  * imex_1d / imex_2d: semi-implicit (IMEX) Euler: diffusion implicit, reactions
    explicit,  (I - dt D Delta_h) w^{n+1} = w^n + dt F(w^n).  The linear solve is
    exact and cheap in the DCT basis.  Used for the long runs in the figures.
"""

import numpy as np
from scipy.fft import dct, idct, dctn, idctn


# ---------------------------------------------------------------- kinetics
def schnakenberg(a, b):
    """f = a - u + u^2 v,  g = b - u^2 v;  steady state (a+b, b/(a+b)^2)."""
    def f(u, v):
        return a - u + u * u * v

    def g(u, v):
        return b - u * u * v

    return f, g, (a + b, b / (a + b) ** 2)


def gierer_meinhardt(a, b):
    """f = a - b u + u^2/v,  g = u^2 - v;  steady state ((a+1)/b, ((a+1)/b)^2)."""
    def f(u, v):
        return a - b * u + u * u / v

    def g(u, v):
        return u * u - v

    ub = (a + 1) / b
    return f, g, (ub, ub ** 2)


def lengyel_epstein(a, b):
    """f = a - u - 4uv/(1+u^2),  g = b (u - uv/(1+u^2));  steady state (a/5, 1+(a/5)^2)."""
    def f(u, v):
        return a - u - 4 * u * v / (1 + u * u)

    def g(u, v):
        return b * (u - u * v / (1 + u * u))

    ub = a / 5
    return f, g, (ub, 1 + ub ** 2)


# ---------------------------------------------------------------- linear theory
def mode_rates(A, du, dv, lam):
    """Largest real part of the eigenvalues of A - lam*diag(du, dv), for an array lam."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    out = np.empty_like(lam)
    for i, l in enumerate(lam):
        out[i] = np.linalg.eigvals(A - l * np.diag([du, dv])).real.max()
    return out


def band(A, du, dv):
    """(lambda_-, lambda_+) of eq:band, or None if h has no negative values."""
    fu, fv, gu, gv = A.ravel()
    p = dv * fu + du * gv
    det = fu * gv - fv * gu
    disc = p * p - 4 * du * dv * det
    if p <= 0 or disc <= 0:
        return None
    s = np.sqrt(disc)
    return (p - s) / (2 * du * dv), (p + s) / (2 * du * dv)


# ---------------------------------------------------------------- grids
def grid_1d(L, N):
    h = L / N
    return (np.arange(1, N + 1) - 0.5) * h


def discrete_eigs_1d(L, N):
    """Eigenvalues -(4/h^2) sin^2(k pi h/(2L)), k = 0..N-1, of the Neumann Delta_h."""
    h = L / N
    k = np.arange(N)
    return -(4.0 / h ** 2) * np.sin(k * np.pi * h / (2 * L)) ** 2


def laplacian_1d(u, h):
    """Delta_h u with reflected ghost values (explicit, for checks)."""
    up = np.concatenate(([u[0]], u, [u[-1]]))
    return (up[2:] - 2 * up[1:-1] + up[:-2]) / h ** 2


def cosine_coefficients(u):
    """Discrete analogues of the cosine coefficients hat u_k (k = 0..N-1)."""
    N = len(u)
    c = dct(u, type=2) / N           # = (2/N) sum_j u_j cos(k pi x_j / L)
    c[0] /= 2                        # k = 0: (1/N) sum_j u_j
    return c


# ---------------------------------------------------------------- time stepping
def euler_1d(f, g, u, v, du, dv, L, dt, nsteps):
    """Explicit Euler for the method-of-lines system (the scheme of the listing)."""
    h = L / len(u)
    u, v = u.copy(), v.copy()
    for _ in range(nsteps):
        u, v = (u + dt * (du * laplacian_1d(u, h) + f(u, v)),
                v + dt * (dv * laplacian_1d(v, h) + g(u, v)))
    return u, v


def imex_1d(f, g, u, v, du, dv, L, dt, T, every=None):
    """IMEX Euler on (0,L). Returns final (u, v) and, if every is given, a list of
    (t, u, v) snapshots taken every `every` time units (including t = 0)."""
    N = len(u)
    lam = -discrete_eigs_1d(L, N)
    mu_u = 1.0 / (1.0 + dt * du * lam)
    mu_v = 1.0 / (1.0 + dt * dv * lam)
    u, v = u.copy(), v.copy()
    nsteps = int(round(T / dt))
    stride = int(round(every / dt)) if every else None
    snaps = [(0.0, u.copy(), v.copy())] if stride else None
    for n in range(1, nsteps + 1):
        ru = u + dt * f(u, v)
        rv = v + dt * g(u, v)
        u = idct(mu_u * dct(ru, type=2, norm="ortho"), type=2, norm="ortho")
        v = idct(mu_v * dct(rv, type=2, norm="ortho"), type=2, norm="ortho")
        if stride and n % stride == 0:
            snaps.append((n * dt, u.copy(), v.copy()))
    return (u, v, snaps) if stride else (u, v)


def imex_2d(f, g, u, v, du, dv, L, dt, T):
    """IMEX Euler on the square (0,L)^2 with Neumann conditions (5-point Laplacian)."""
    N = u.shape[0]
    lam1 = -discrete_eigs_1d(L, N)
    lam = lam1[:, None] + lam1[None, :]
    mu_u = 1.0 / (1.0 + dt * du * lam)
    mu_v = 1.0 / (1.0 + dt * dv * lam)
    u, v = u.copy(), v.copy()
    for _ in range(int(round(T / dt))):
        ru = u + dt * f(u, v)
        rv = v + dt * g(u, v)
        u = idctn(mu_u * dctn(ru, type=2, norm="ortho"), type=2, norm="ortho")
        v = idctn(mu_v * dctn(rv, type=2, norm="ortho"), type=2, norm="ortho")
    return u, v
