"""
plot_style.py
Single source of truth for the figure typography. Importing this module sets the
matplotlib text (axis labels, titles, legends, tick labels, annotations) to Times
New Roman, so every figure the code produces uses the same serif face as the paper.

Times New Roman is named first; on a machine where it is installed (e.g. Windows) it
is used directly. The fallbacks are metric-compatible Times clones (Liberation Serif,
TeX Gyre Termes, Nimbus Roman) so the figures still render in a Times-like face on a
machine without the exact font, without a hard failure. Math text uses the STIX set,
which is the Times-compatible math companion, so symbols such as omega and the bars in
the input-power labels match the surrounding serif text.

Import it once, near the top of any module that makes figures:

    import plot_style  # noqa: F401  (sets Times New Roman on import)

It only sets rcParams; it does not choose a backend, so drivers remain free to select
Agg or an interactive backend as before.
"""

import matplotlib as mpl
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.15

_SERIF = [
    "Times New Roman",   # installed on the author's machine
    "Times",
    "Nimbus Roman",      # ghostscript Times clone
    "Liberation Serif",  # metric-compatible Times clone (Linux)
    "TeX Gyre Termes",   # Times clone shipped with TeX
    "STIXGeneral",
    "DejaVu Serif",      # last-resort, always present with matplotlib
]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": _SERIF,
    "mathtext.fontset": "stix",          # Times-compatible math
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "axes.formatter.use_mathtext": True,  # scientific-notation offset in the same face
})


def apply():
    """Re-apply the style. Useful if some other code has since reset rcParams."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF,
        "mathtext.fontset": "stix",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        "axes.formatter.use_mathtext": True,
    })
