"""
Shared matplotlib style for the paper figures.

Two series only, KNU and MIMIC-IV, distinguished by color and by marker shape so
that identity never rests on hue alone. Text and chrome use the ink tokens below
rather than a series color.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# series colors
KNU = "#B23A48"
MIMIC = "#1F6FB2"
SURFACE = "#FFFFFF"

# ink tokens, for text and chrome
INK = "#1A1A1A"
INK_SECONDARY = "#4A4A4A"
INK_MUTED = "#7A7A78"
GRID = "#E3E3E0"

# marks
MARKER = {"knu": "o", "mimic": "s"}
LABEL = {"knu": "KNU", "mimic": "MIMIC-IV"}
COLOR = {"knu": KNU, "mimic": MIMIC}
LW_DATA = 1.6          # data lines
LW_CHROME = 0.8        # spines / hairline grid
MS = 5.5               # marker size (pt)
RING = 1.2             # surface ring on markers
AREA_ALPHA = 0.10


def apply_style():
    """Global rcParams. Call once at the top of a figure script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": LW_CHROME,
        "xtick.major.width": LW_CHROME,
        "ytick.major.width": LW_CHROME,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.color": GRID,
        "grid.linewidth": LW_CHROME,
        "grid.linestyle": "-",
        "axes.grid": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,        # embed fonts as TrueType
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axes(ax, grid_axis=None):
    """Drop the top/right spines; optionally add a recessive hairline grid."""
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
        ax.set_axisbelow(True)
    return ax


def series_kw(domain, **over):
    """Standard mark spec for a domain's points: colored fill + surface ring."""
    kw = dict(color=COLOR[domain], marker=MARKER[domain], markersize=MS,
              markeredgecolor=SURFACE, markeredgewidth=RING,
              label=LABEL[domain], linewidth=LW_DATA)
    kw.update(over)
    return kw


def legend_handles(domains=("knu", "mimic")):
    """Marker-only legend handles. Passing errorbar artists straight to legend()
    drags the error caps in with them, which reads as noise in the key."""
    import matplotlib.pyplot as _plt
    return [_plt.Line2D([], [], color=COLOR[d], marker=MARKER[d], markersize=MS,
                        markeredgecolor=SURFACE, markeredgewidth=RING,
                        linestyle="none", label=LABEL[d]) for d in domains]


def save(fig, path, dpi=600):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  saved -> {path}")
