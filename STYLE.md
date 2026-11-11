# Style guide and notation for the script

This file is binding for every chapter. Chapter 5
(`chapters/05-turing-instability.tex`) is the reference chapter: when in doubt,
imitate its tone, structure, level of detail and LaTeX usage.

## Audience and level

* Mathematics students who know ODEs (linear systems, matrix exponential,
  linearization, phase plane) and real analysis, and who are **just starting PDEs**.
* Everything is done on the interval `(0,L)` with **homogeneous Neumann (zero-flux)
  boundary conditions**. General domains `Ω ⊂ ℝⁿ` appear only in remarks, with
  results stated without proof.
* No Sobolev spaces, no semigroups, no spectral theory of unbounded operators,
  no measure theory, before Chapter 7. "Square-integrable" and the L² norm are fine.
* Standing assumption (stated once in Chapter 3, `rem:well-posedness`): the problems
  we consider have unique smooth solutions for smooth initial data compatible with
  the boundary conditions. We never prove existence.
* Every theorem/proposition/lemma either has a full proof, or says explicitly
  "We omit the proof; see \cite{...}." Proofs should be complete at the level of
  a careful undergraduate text: no "it is easy to see" for non-trivial steps.

## Language and tone

* American English ("behavior", "stabilize", "modeling").
* Use "we". Friendly but precise. Short paragraphs. Motivate before defining.
* Each chapter opens with one short paragraph saying what the chapter does and
  why (no heading for it). Use 1–3 `keyidea` boxes per chapter for central
  take-aways (not more).
* Never copy text from the source slides or from books; paraphrase. At most one
  direct quotation per chapter, under 15 words, with citation.

## Notation (fixed for the whole text)

| Object | Notation | LaTeX |
|---|---|---|
| space, time, interval | `x ∈ (0,L)`, `t > 0` | `x\in(0,L)` |
| integral | `∫_0^L u dx` | `\int_0^L u\dd x` |
| L² norm on (0,L) | `‖u‖ = (∫_0^L u² dx)^{1/2}` | `\norm{u}` (no subscript) |
| Euclidean norm in ℝ² / ℂ² | `|ξ|` | `\abs{\bxi}` |
| scalar diffusion coefficient (Ch. 3, 4) | `d > 0` | `d` |
| scalar RD equation (Ch. 4) | `u_t = d u_xx + f(u)` | |
| two-component system | `u_t = d_u u_xx + f(u,v)`, `v_t = d_v v_xx + g(u,v)` | `d_u, d_v` |
| diffusion matrix | `D = diag(d_u, d_v)` | `D=\diag(d_u,d_v)` |
| diffusion ratio | `δ = d_v / d_u` | `\delta` |
| constant steady state | `(ū, v̄)`, scalar case `ū` | `(\ub,\vb)` |
| nonconstant stationary solution (Ch. 4) | `U(x)` | `U` |
| Jacobian of kinetics at `(ū,v̄)` | `A = [[f_u, f_v],[g_u, g_v]]` | partial derivatives as subscripts `f_u`; avoid `a_{ij}` |
| Neumann eigenvalues | `λ_k = (kπ/L)²`, `k = 0,1,2,…` (note `λ_0 = 0`) | `\lambda_k` |
| Neumann eigenfunctions | `φ_k(x) = cos(kπx/L)` (not normalized) | `\varphi_k` |
| wavenumber | `q_k = √λ_k = kπ/L` | `q_k` |
| cosine coefficients | `û_k = (1/L)∫u` for `k=0`, `û_k = (2/L)∫u cos(kπx/L)` for `k≥1` | `\hat u_k` |
| vector perturbation | `w = (w_1, w_2)^T` | `\bw` |
| its cosine coefficients | `ŵ_k(t) ∈ ℝ²` | `\hatbw_k` |
| mode matrices | `A_k = A − λ_k D` | `A_k` |
| growth rates (eigenvalues of A, A_k, …) | `μ` | `\mu` |
| determinant function | `h(λ) = det(A − λD)` | `h(\lambda)` |
| eigenvectors | `ξ ∈ ℂ²` | `\bxi` |
| real part | `Re μ` | `\re\mu` |

Further conventions:

* `N` means `{1,2,3,…}`; write `k = 0,1,2,…` or `k ≥ 0` explicitly when `k = 0` is included.
* Partial derivatives as subscripts: `u_t`, `u_{xx}`, `f_u`. Ordinary derivatives with primes.
* "steady state" for ODEs, "stationary solution" for PDEs; "constant steady state"
  for `(ū,v̄)` viewed as a solution of the PDE.
* "Diffusion-driven instability (DDI)" = "Turing instability". Spell out at first
  use in each chapter.
* Vocabulary: "linearly stable / linearly unstable" (Chapter 5, `def:linear-stability-rd`),
  "kinetic system" = the ODE without diffusion.
* `u` is always the first species; in examples with `f_u > 0` it is the
  self-enhancing species ("activator").

## Running examples (defined in Chapter 2, reused everywhere)

Use exactly these forms, parameters and labels.

1. **Schnakenberg** `\label{eq:schnakenberg}`:
   `f(u,v) = a − u + u²v`, `g(u,v) = b − u²v`, `a, b > 0`.
   Steady state `ū = a + b`, `v̄ = b/(a+b)²`.
   `f_u = (b−a)/(a+b)`, `f_v = (a+b)²`, `g_u = −2b/(a+b)`, `g_v = −(a+b)²`,
   `det A = (a+b)²`.
   **Standard parameters**: `a = 0.1, b = 0.9` ⇒ `ū = 1, v̄ = 0.9`,
   `A = [[0.8, 1], [−1.8, −1]]`, `tr A = −0.2`, `det A = 1`.
   Standard diffusion: `d_u = 1, d_v = 20`, interval `L = 40`
   (then exactly the modes `k = 4,…,10` are unstable, the fastest is `k = 6`; see `ex:schnakenberg-turing`).
2. **Gierer–Meinhardt** (simplest form, no saturation) `\label{eq:gierer-meinhardt}`:
   `f(u,v) = a − b u + u²/v`, `g(u,v) = u² − v`, `a ≥ 0, b > 0`.
   Steady state `ū = (a+1)/b`, `v̄ = ū²`.
   `f_u = b(1−a)/(1+a)`, `f_v = −b²/(a+1)²`, `g_u = 2(a+1)/b`, `g_v = −1`, `det A = b`.
   **Standard parameters**: `a = 0, b = 0.5` ⇒ `ū = 2, v̄ = 4`,
   `A = [[0.5, −0.25], [4, −1]]`, `tr A = −0.5`, `det A = 0.5`.
3. **Lengyel–Epstein (CIMA reaction)** `\label{eq:lengyel-epstein}`:
   `f(u,v) = a − u − 4uv/(1+u²)`, `g(u,v) = b (u − uv/(1+u²))`, `a, b > 0`.
   Steady state `ū = a/5`, `v̄ = 1 + ū²`.
   `f_u = (3ū² − 5)/(1+ū²)`, `f_v = −4ū/(1+ū²)`, `g_u = 2bū²/(1+ū²)`, `g_v = −bū/(1+ū²)`.

Schnakenberg is of substrate-depletion ("cross") type (`f_v > 0 > g_u`),
Gierer–Meinhardt of activator–inhibitor ("pure") type (`f_v < 0 < g_u`).

## Label table (use these; do not rename)

Prefixes: `ch:`, `sec:`, `eq:`, `thm:`, `prop:`, `lem:`, `cor:`, `def:`, `ex:`
(examples), `exer:` (exercises), `rem:`, `fig:`, `tab:`. Figure labels contain the
chapter: `fig:ch05-dispersion`. Refer with `\cref{...}` / `\Cref{...}` only.

You may `\cref` any label below even if its chapter is not written yet.
Create the labels your chapter exports, with exactly these names. Additional
internal labels are fine but must start with your chapter's topic, e.g.
`eq:heat-...`.

* Ch. 1 `ch:introduction`: `sec:roadmap`, `fig:ch01-activator-inhibitor`
* Ch. 2 `ch:ode-toolbox`: `def:stability-ode`, `thm:linear-ode-decay`,
  `lem:trace-det`, `thm:linearized-stability`, `eq:schnakenberg`, `ex:schnakenberg`,
  `eq:gierer-meinhardt`, `ex:gierer-meinhardt`, `eq:lengyel-epstein`,
  `ex:lengyel-epstein`, `tab:standard-parameters`
* Ch. 3 `ch:diffusion`: `eq:fick`, `eq:heat`, `eq:neumann`, `rem:well-posedness`,
  `prop:mass-conservation`, `lem:poincare-wirtinger`, `thm:heat-decay`,
  `eq:neumann-eigenproblem`, `prop:neumann-eigen`, `eq:cosine-coefficients`,
  `lem:coefficient-uxx`, `eq:heat-series`, `thm:neumann-eigen-general`,
  `ex:rectangle-eigen`, `rem:wavenumber`, `fig:ch03-heat-modes`
* Ch. 4 `ch:one-equation`: `eq:scalar-rd`, `prop:scalar-constant`,
  `cor:no-ddi-scalar`, `lem:rayleigh-growth`, `thm:nonconstant-unstable`,
  `rem:lyapunov`, `rem:higher-dim`, `rem:why-two`, `fig:ch04-phase-plane`
* Ch. 5 `ch:turing`: `eq:rd-system`, `eq:kinetic-system`, `eq:jacobian-A`,
  `eq:linearized-system`, `def:linear-stability-rd`, `def:ddi`, `prop:normal-modes`,
  `thm:modal-decomposition`, `thm:spectral-criterion`, `eq:h-def`,
  `thm:ddi-criterion`, `eq:turing-conditions`, `cor:turing-consequences`,
  `thm:critical-ratio`, `eq:band`, `cor:domain-size`, `thm:turing-small-du`,
  `ex:schnakenberg-turing`, `rem:general-domain`, `rem:turing-bifurcation`,
  `fig:ch05-dispersion`, `fig:ch05-band`
* Ch. 6 `ch:examples`: `sec:turing-spaces`, `sec:numerics`, `eq:discrete-laplacian`,
  `fig:ch06-turing-space`, `fig:ch06-1d-evolution`, `fig:ch06-phase-relation`,
  `fig:ch06-domain-size`, `fig:ch06-2d-patterns`
* Ch. 7 `ch:outlook`: `def:turing-pattern`, `thm:nonlinear-instability`
* App. A `app:fourier`: `thm:cosine-series`, `eq:parseval`
* App. B `app:linear-algebra`: `prop:2x2-eigenvalues`, `prop:2x2-exponential-bound`

## LaTeX conventions

* Shared preamble in `preamble.tex` (packages, environments, macros). Do not add
  packages or `\newcommand`s in chapter files. If you need a macro, write it out.
* Environments: `theorem, proposition, lemma, corollary` (plain), `definition,
  example, exercise` (definition style), `remark` (remark style), `proof`,
  `keyidea` (box). Shared counter per chapter (exercises have their own).
* Displayed equations: `equation`/`align` with `\label` only when referenced;
  otherwise `equation*`/`align*`. Systems with `\begin{aligned}` or `cases`.
* Each chapter ends with `\section{Exercises}\label{sec:<topic>-exercises}`,
  4–6 exercises; mark harder ones with "($\ast$)" at the start of the text.
* Figures: matplotlib scripts `scripts/chNN_<name>.py` using `scripts/plotstyle.py`
  (`ps.apply()`, `ps.save(fig, "chNN-name")`), output vector PDF to `figures/`.
  Include with `\includegraphics[width=...]{chNN-name}` (no path, no extension).
  Schematic diagrams: TikZ inline. No external images (copyright).
  Every figure gets a self-contained caption.
* Citations: only keys from `references.bib`, via `\cite{...}`. Do not add entries;
  if something is missing, add `% TODO cite: ...` in the chapter.
* Check your chapter compiles: `bash scripts/build_chapter.sh 03-diffusion`
  (builds in `build/`, never touches the main document). Undefined references to
  labels of other chapters are expected and fine.

## Page budgets (11pt, A4)

Ch. 1: 2–3 · Ch. 2: 4–5 · Ch. 3: 6–7 · Ch. 4: 4–5 · Ch. 5: 8–10 · Ch. 6: 5–6 ·
Ch. 7: 2–3 · App. A + B: 3 (exercises included in each chapter's budget).
