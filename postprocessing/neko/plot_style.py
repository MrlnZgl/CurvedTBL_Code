"""Central plot styling.

Import this module once near the top of any plotting script:

    import plot_style          # applies the settings on import

Every matplotlib figure created afterwards uses these settings automatically.
Change a value here and it takes effect in every script that imports it.
"""
import os
import shutil
import matplotlib as mpl
import numpy as np

# Make sure the TinyTeX binaries are findable, even when the Python process was
# launched from an environment (e.g. the IDE, a batch job, or an MPI rank) that
# didn't inherit the shell PATH. Without this, HAS_PDFLATEX below comes out False
# on machines that DO have TinyTeX and every script silently drops to pdf-only.
if shutil.which('pdflatex') is None:
    for _tinytex_bin in (os.path.expanduser('~/.TinyTeX/bin/x86_64-linux'),
                         os.path.expanduser('~/TinyTeX/bin/x86_64-linux')):
        if os.path.isdir(_tinytex_bin):
            os.environ['PATH'] = _tinytex_bin + os.pathsep + os.environ.get('PATH', '')
            break

# The pgf backend renders every piece of text through a real LaTeX engine, so it
# only works where pdflatex is actually installed. Fall back to matplotlib's
# built-in mathtext rendering (no external TeX) on machines without one.
HAS_PDFLATEX = shutil.which('pdflatex') is not None

if HAS_PDFLATEX:
    mpl.use('pgf')  # write LaTeX/pgf output; needs a TeX engine (pdflatex) on PATH at save time

    # also render plain '.pdf' saves through the pgf/LaTeX canvas (not matplotlib's
    # default Agg-based pdf backend), so pdf and pgf output stay visually identical
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

dpi = 100

def set_size(width_pts=455.24408, fraction=1., subplots=(1, 1)):
    """Figure size (inches) that fits `width_pts` LaTeX points without scaling.

    width_pts : document text width in points (\\the\\textwidth in LaTeX)
    fraction  : portion of the width the figure should occupy
    subplots  : (rows, cols) grid, used to scale the height
    """
    fig_width_pt = width_pts * fraction
    inches_per_pt = 1 / 72.27  # TeX points -> inches (fixed constant; NOT dpi)
    golden_ratio = 0.8#(5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


if HAS_PDFLATEX:
    # output format: pgf (LaTeX). savefig() without an explicit extension uses this.
    mpl.rcParams.update({
        'savefig.format':   'pgf',
        'pgf.texsystem':    'pdflatex',
        'pgf.rcfonts':      False,    # let the LaTeX document control the fonts
    })
else:
    # no TeX engine on this machine: fall back to matplotlib's built-in mathtext
    # rendering and plain pdf output instead of pgf/LaTeX.
    mpl.rcParams.update({
        'savefig.format':   'pdf',
        'text.usetex':      False,
    })

mpl.rcParams.update({
    'font.family':      'serif',

    # figure
    'figure.figsize':   set_size(),
    'figure.dpi':       dpi,
    'savefig.dpi':      200,
    # NOTE: do NOT set savefig.bbox='tight' for the pgf workflow. It re-crops the
    # canvas at save time, so the saved figure is no longer figsize and the exact
    # width from set_size() (matched to the LaTeX subfigure) is lost. Use
    # constrained_layout below to fill a *fixed* canvas instead.
    'savefig.bbox':     None,
    'savefig.pad_inches': 0.0,

    # let matplotlib pack axes/labels with minimal internal padding
    'figure.constrained_layout.use':   True,
    'figure.constrained_layout.h_pad': 0.02,
    'figure.constrained_layout.w_pad': 0.02,
    'figure.constrained_layout.hspace': 0.02,
    'figure.constrained_layout.wspace': 0.02,

    # fonts / text
    'font.size':        11,
    'axes.titlesize':   11,
    'axes.labelsize':   12,
    'axes.labelpad':    2.0,   # default 4.0 -- pulls axis labels in closer to the tick labels
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  8,

    # lines
    'lines.linewidth':  1.0,
    'lines.markersize': 6,

    # axes / grid
    'axes.grid':        False,
    'grid.linestyle':   ':',
    'grid.alpha':       0.5,
    'legend.frameon':   False,

    # ticks: inward, on all four sides of the axes box
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'xtick.bottom':     True,
    'ytick.left':       True,
    'ytick.right':      True,
    'xtick.major.size': 3.0,   # default 3.5
    'xtick.minor.size': 1.5,   # default 2.0
    'ytick.major.size': 3.0,
    'ytick.minor.size': 1.5,
    'xtick.minor.visible': True,   # show minor ticks between the major ones
    'ytick.minor.visible': True,
    'xtick.major.width':   0.5,    # default 0.8
    'ytick.major.width':   0.5,
    'xtick.minor.width':   0.4,    # default 0.6
    'ytick.minor.width':   0.4,

})


def colorbrewer_cmap(hex_colors, name='custom'):
    """Discrete colormap from a list of hex strings, e.g. pasted from a
    colorbrewer2.org palette's 'Export' hex list:

        FLUCTUATIONS_SET2 = colorbrewer_cmap(
            ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'], name='set2_5')

    cmap(i) for i >= len(hex_colors) clamps to the last color instead of
    erroring -- if a plot indexes further than the list is long (e.g.
    cmap(2*k) for k up to 3 needs index 6, so >= 7 colors), pass more colors
    or adjust the indices used at the call site.
    """
    from matplotlib.colors import ListedColormap
    return ListedColormap(hex_colors, name=name)


# --- colour schemes: config 1 (primary/solid) vs config 2 (dashed overlay) --
# One colormap pair per TBL_pp_plots.py plot kind. Edit the colormap *names*
# here to change how every TBL_pp_* script's config1/config2 comparison
# looks; plot_style.get_colors() is what the plotting functions call. Entries
# can be either a built-in matplotlib name (str, looked up via plt.get_cmap)
# or a Colormap object -- e.g. the output of colorbrewer_cmap() above.

# budgets/config1: one color per term (Production/Convection/Dissipation/
# Vel-Press./Turb. Trans./Visc. Diff/Residual), indexed sequentially
# cmap(0..6) -- see _plot_budget_component_Re() in TBL_plots_txt_Re.py.
_BUDGETS_SET1 = ['#e41a1c',  "#25b0b0", '#4daf4a', '#984ea3', "#ffd900",'#377eb8', '#a65628']  # 7 of colorbrewer2.org Set1 n=8
_BUDGETS_SET2 = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17']
PLOT_COLOR_SCHEMES = {
    'budgets':      {'config1': colorbrewer_cmap(_BUDGETS_SET1, name='set1_8_budgets'),
                     'config2': colorbrewer_cmap(_BUDGETS_SET1, name='set1_8_budgets')},
    'velocity':     {'config1': colorbrewer_cmap(
                        ['#377eb8', "#a89008", '#f781bf'], name='set1_8'),  'config2': 'tab20b'},
    'rms':          {'config1': colorbrewer_cmap(
                        ['#1b9e77', '#e6ab02', '#7570b3', '#e7298a'], name='Dark2_4'),
                     'config2': colorbrewer_cmap(
                        ['#1b9e77', '#e6ab02', '#7570b3', '#e7298a'], name='Dark2_4'),},
    'fluctuations': {'config1': colorbrewer_cmap(
                        ['#1b9e77', '#e6ab02', '#7570b3', '#e7298a'], name='Dark2_4'),
                     'config2': colorbrewer_cmap(
                        ['#1b9e77', '#e6ab02', '#7570b3', '#e7298a'], name='Dark2_4'),},
}


def get_colors(kind, role='config1'):
    """Colormap for `role` ('config1' or 'config2') in a `kind` plot
    ('budgets', 'velocity', 'rms', 'fluctuations'). Edit PLOT_COLOR_SCHEMES
    above to change any script's colours."""
    import matplotlib.pyplot as plt
    cmap = PLOT_COLOR_SCHEMES[kind][role]
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def dedup_legend(ax=None, **kwargs):
    """Draw a legend with duplicate labels collapsed to a single entry.

    Entries are keyed by their label *text*: several lines that share a label
    (e.g. the same dataset drawn with different linestyles) produce ONE legend
    entry, shown with the first matching line's style. Give lines the same
    label to merge them; give a label of None or one starting with '_' to keep
    a line out of the legend entirely.

    Any extra keyword arguments are forwarded to ``ax.legend`` (loc, ncol, ...).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:            # keep the first handle for each label
            unique[l] = h
    ax.legend(unique.values(), unique.keys(), **kwargs)
    return ax


def hide_origin_ticklabel(ax=None, axis='y'):
    """Hide the tick label at the plot's origin corner.

    When both axes start at the same value (e.g. xlim(0, ...) and
    ylim(0, ...)), their '0' tick labels sit on top of each other in the
    bottom-left corner. Call this after setting xlim/ylim to hide the
    lowest tick label on `axis` ('x' or 'y') so only the other axis' label
    remains visible there.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    ticks = ax.yaxis.get_major_ticks() if axis == 'y' else ax.xaxis.get_major_ticks()
    if ticks:
        ticks[0].label1.set_visible(False)
    return ax


def fixed_axes(fig=None, left=0.55, bottom=0.42, right=0.08, top=0.08):
    """Position the figure's axes at a fixed inch margin from every edge,
    instead of letting constrained_layout auto-size the plot area to fit
    whatever labels happen to be on it.

    Two figures created with the same set_size() figsize otherwise end up
    with *different* plot areas if their y-labels differ in rendered bbox
    size (e.g. a plain '$U^+$' vs a tall '$\\langle u_i u_j \\rangle^+$') --
    constrained_layout shrinks the axes to make room for whichever label is
    bigger. Fixed inch margins make the axes size depend only on figsize, so
    same figsize -> same plot area, always.

    Call right after plt.figure(...), before plotting. Margins are in
    inches; make them wide enough for the widest label used anywhere in your
    script, or a narrower label just leaves extra whitespace instead of
    sizing down to fit.
    """
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.gcf()
    fig.set_layout_engine(None)  # stop constrained_layout from touching this figure
    ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
    w, h = fig.get_size_inches()
    ax.set_position([left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h])
    return ax


def save(name, formats=('pdf', 'pgf') if HAS_PDFLATEX else ('pdf',)):
    """Save the current figure once per format, e.g. name.pdf and name.pgf.

    Pass `name` without an extension. Change the default `formats` here to
    change what every script produces. '.pgf' is only produced when a TeX
    engine (pdflatex) is available, since matplotlib's pgf backend needs one
    to render text at all.
    """
    import matplotlib.pyplot as plt
    for fmt in formats:
        plt.savefig(f'{name}.{fmt}')

def drop_outliers(x, y, window=7, n_sigma=5.):
    """Remove isolated spikes from an ordered profile via a Hampel filter.

    For each point, compare it to the median of a sliding window. Points that
    deviate by more than ``n_sigma`` scaled MADs are treated as broken and
    dropped (not interpolated), so the plotted line simply connects across the
    gap. Increase ``n_sigma`` to remove fewer points, decrease to remove more.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < window:
        return x, y
    k = 1.4826  # MAD -> std for a normal distribution
    half = window // 2
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        med = np.median(y[lo:hi])
        mad = np.median(np.abs(y[lo:hi] - med))
        if mad > 0 and np.abs(y[i] - med) > n_sigma * k * mad:
            keep[i] = False
    return x[keep], y[keep]

