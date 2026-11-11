"""Shared plotting style for all figures in the script.

Usage in a figure script:

    import plotstyle as ps
    ps.apply()
    fig, ax = plt.subplots(figsize=ps.FULL)        # or ps.HALF, ps.WIDE
    ax.plot(x, u, color=ps.U_COLOR, label=r"$u$")
    ...
    ps.save(fig, "ch05-dispersion")                 # -> figures/ch05-dispersion.pdf

Conventions
-----------
* u (first species, usually the activator / self-enhancing one) is always U_COLOR,
  v (second species) is always V_COLOR.
* Unstable / growing things use UNSTABLE_COLOR, stable / decaying things STABLE_COLOR.
* Figures are vector PDFs, named  chNN-short-name.pdf, written to ../figures/.
* Include them in LaTeX with  \\includegraphics[width=\\textwidth]{chNN-short-name}
  (FULL), width=0.7\\textwidth (WIDE) or 0.48\\textwidth (HALF).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Sizes in inches. The text width of the notes is about 6.2 in.
FULL = (6.2, 2.6)
WIDE = (4.4, 2.8)
HALF = (3.0, 2.3)

U_COLOR = "#1f4e79"        # dark blue
V_COLOR = "#c0392b"        # dark red
STABLE_COLOR = "#2e7d32"   # green
UNSTABLE_COLOR = "#d35400" # orange
GREY = "#7f7f7f"
LIGHT_GREY = "#d9d9d9"


def apply():
    """Apply the shared rcParams."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })


def save(fig, name):
    """Save `fig` as figures/<name>.pdf and close it."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path
