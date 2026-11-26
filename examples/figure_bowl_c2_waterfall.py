"""Waterfall of the harmonic spectra at the beam maximum of benchmark C.2 (IEC TS 63587), for any
number of source pressures.

Each run is one bar chart of |P_n| against the harmonic number n, placed at its own depth in a 3D
axes (a waterfall of bar charts, as MATLAB's ``waterfall``), ordered by source pressure, so that
the spread of energy from the fundamental into the harmonics as p0 rises can be read along the
depth axis. The bars of a run share one colour, from a single-hue sequential scale on log p0.

The spectra are read from the finest nonlinear run in each folder that
``examples/benchmark_c2_pywave3d.py`` wrote (``nonlinear_ppw*``): the harmonics on the axis at the
focal index of ``results_meta.json``. Only ``axis`` is read from ``results.npz``, so the planes are
not loaded. The normalisation is chosen with ``--norm``:

* ``p0`` (default): |P_n| / p0, as Figure C.2 (b);
* ``fundamental``: |P_n| / |P_1|, the shape of the distribution alone;
* ``energy``: |P_n|^2 / sum_m |P_m|^2, the fraction of the mean-square pressure in each harmonic.

Usage::

    python examples/figure_bowl_c2_waterfall.py outputs/c2_pywave3d_p0_*MPa
    python examples/figure_bowl_c2_waterfall.py outputs/c2_pywave3d_p0_* --harmonics 20 --norm fundamental

The figure is written to ``docs/figures/bowl_c2_waterfall`` as ``.png``, ``_transparent.png`` and
``.jpg``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from figure_bowl_c2 import finest_run, regime, save_three

log = logging.getLogger("figure_bowl_c2_waterfall")

NORMS = {
    "p0": r"$|P_n|\,/\,p_0$",
    "fundamental": r"$|P_n|\,/\,|P_1|$",
    "energy": r"$|P_n|^2\,/\,\sum_m |P_m|^2$",
}


def focal_spectrum(folder: Path) -> tuple[float, np.ndarray, Path]:
    """(p0, complex harmonics at the beam maximum, the run read) for the folder of a C.2 run."""
    p0 = float(json.loads((folder / "c2_summary.json").read_text())["inputs"]["p0"])
    run = finest_run(folder)
    meta = json.loads((run / "results_meta.json").read_text())
    with np.load(run / "results.npz", allow_pickle=False) as data:
        h = np.asarray(data["axis"][int(meta["focal_index"])])
    return p0, h, run


def normalise(h: np.ndarray, p0: float, norm: str) -> np.ndarray:
    a = np.abs(h)
    if norm == "p0":
        return a / p0
    if norm == "fundamental":
        return a / a[0]
    if norm == "energy":
        return a**2 / np.sum(a**2)
    raise ValueError(f"unknown normalisation {norm!r}")


def waterfall_figure(cases: list[tuple[float, np.ndarray]], n_bars: int, norm: str, text_color: str,
                     cmap: str = "Blues", elev: float = 24.0, azim: float = -58.0) -> Any:  # fmt: skip
    """The waterfall: one bar chart of the first ``n_bars`` harmonics per (p0, harmonics), ordered by p0."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.size": 9, "text.color": text_color, "axes.labelcolor": text_color, "axes.edgecolor": text_color,
        "xtick.color": text_color, "ytick.color": text_color, "axes.titlecolor": text_color,
    })  # fmt: skip
    cases = sorted(cases, key=lambda c: c[0])
    lp = np.log([p0 for p0, _ in cases])
    t = (lp - lp[0]) / (lp[-1] - lp[0]) if len(cases) > 1 else np.ones(1)
    colors = colormaps[cmap](0.45 + 0.5 * t)  # light to dark, never the near-white end

    fig = plt.figure(figsize=(8.4, 6.2), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    n = np.arange(1, n_bars + 1)
    handles = []
    top = 0.0
    for q, ((p0, h), color) in enumerate(zip(cases, colors, strict=True)):
        k = min(n_bars, h.size)
        amp = normalise(h, p0, norm)[:k]
        top = max(top, float(amp.max()))
        # back rows first, so the nearer (lower p0) rows draw over them
        ax.bar(n[:k], amp, zs=q, zdir="y", width=0.7, color=color, edgecolor=text_color, linewidth=0.3,
               alpha=0.95, zorder=len(cases) - q)  # fmt: skip
        name = regime(p0)[0]
        handles.append(Patch(facecolor=color, edgecolor=text_color, linewidth=0.3,
                             label=f"{p0 / 1e6:g} MPa, {name} ({h.size} harmonics)"))  # fmt: skip
    ax.set_xlabel(r"harmonic number $f\,/\,f_0$")
    ax.set_ylabel(r"source pressure $p_0$")
    ax.set_zlabel(NORMS[norm])
    ax.set_xticks(n if n_bars <= 16 else n[::2])
    ax.set_yticks(np.arange(len(cases)), [f"{p0 / 1e6:g} MPa" for p0, _ in cases])
    ax.set_xlim(0.4, n_bars + 0.6)
    ax.set_ylim(-0.5, len(cases) - 0.5)
    ax.set_zlim(0, 1.05 * top)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.6, max(0.6, 0.3 * len(cases)), 0.9))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("none")
        pane.set_edgecolor((0.5, 0.5, 0.5, 0.3))
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("none")
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, title=r"$p_0$",
              title_fontsize=7.5, alignment="left")  # fmt: skip
    fig.suptitle(f"Focused bowl in water, IEC TS 63587 benchmark C.2: spectra at the beam maximum, the first "
                 f"{n_bars} harmonics, against the source pressure", fontsize=9)  # fmt: skip
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="the folders of C.2 runs (benchmark_c2_pywave3d.py -o)")
    parser.add_argument("-o", "--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--name", default="bowl_c2_waterfall", help="file name stem")
    parser.add_argument("--harmonics", type=int, default=12, help="number of harmonics drawn")
    parser.add_argument("--norm", choices=sorted(NORMS), default="p0", help="normalisation of the bars")
    parser.add_argument("--cmap", default="Blues", help="single-hue sequential matplotlib colormap")
    parser.add_argument("--elev", type=float, default=24.0, help="view elevation [deg]")
    parser.add_argument("--azim", type=float, default=-58.0, help="view azimuth [deg]")
    parser.add_argument("--text-color", default="#303030")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import matplotlib

    matplotlib.use("Agg")

    cases = []
    for folder in args.runs:
        p0, h, run = focal_spectrum(folder)
        log.info("read %s: p0 = %g MPa, %d harmonics, |P_1| / p0 = %.2f", run, p0 / 1e6, h.size, abs(h[0]) / p0)
        cases.append((p0, h))
    args.output.mkdir(parents=True, exist_ok=True)
    fig = waterfall_figure(cases, args.harmonics, args.norm, args.text_color, args.cmap, args.elev, args.azim)
    for f in save_three(fig, args.output / args.name, args.dpi):
        log.info("wrote %s (%.0f kB)", f, f.stat().st_size / 1e3)


if __name__ == "__main__":
    main()
