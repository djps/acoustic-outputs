"""Figures of benchmark C.2 of IEC TS 63587 (the focused bowl in water, nonlinear), from a saved run.

It reads the run that ``examples/benchmark_c2_pywave3d.py`` saved (``Results.load``), and draws:

1. ``bowl_c2_field``: the figure of the linear field (``figure_bowl_field.py``) for the nonlinear
   one. The peak positive pressure p+ in dB on the plane through the axis, with its -3, -6, -12
   and -20 dB contours and an inset of the focal region; p+ and |p-| on the axis, with the focal
   lengths of p+, its beam maximum and its prefocal and postfocal maxima, and the linear field at
   the same source pressure (C.1) for comparison; p+ across the beam in the plane of the beam
   maximum, with its focal widths and side lobes.
2. ``bowl_c2_waveform``: as Figure C.2 of the TS, for this source pressure: (a) the pressure
   waveform at the beam maximum, over two periods, with p+ and p-, and the linear waveform at the
   same source pressure; (b) its spectrum, the amplitude of each harmonic; (c) the amplitudes of
   the first harmonics along the axis.

Every marked value comes from the package's outputs of the run (``Results.metrics``,
``Results.level_extents``, ``iec.secondary_maxima``, ``iec.side_lobe_peaks``, and
``waveform.waveform`` for the waveform from the recorded harmonics). Each figure is written as
``.png`` (white background), ``_transparent.png`` and ``.jpg``::

    python examples/figure_bowl_c2.py outputs/c2_pywave3d_p0_0.15MPa          # into docs/figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from figure_bowl_field import MINUS, features_from_results, figure, region_extent

import acoustic_outputs as ao
from acoustic_outputs.benchmarks import FocusedBowl
from acoustic_outputs.waveform import waveform

log = logging.getLogger("figure_bowl_c2")

P_PLUS = {"color": "#08519c", "lw": 1.4}
P_MINUS = {"color": "#525252", "lw": 1.1}
LINEAR = {"color": "#9e9e9e", "lw": 0.9, "ls": (0, (4, 2))}


def finest_run(run: Path) -> Path:
    """The nonlinear run with the most points per wavelength in the folder of a C.2 run."""
    runs = sorted(run.glob("nonlinear_ppw*"), key=lambda p: float(re.sub(r"[^0-9.]", "", p.name.split("ppw")[1])))
    if not runs:
        raise FileNotFoundError(f"no nonlinear_ppw* run in {run}")
    return runs[-1]


def field_figure(res: ao.Results, bowl: FocusedBowl, p0: float, text_color: str) -> Any:
    """Figure 1: the field of p+, its axis and its focal plane."""
    ix = res.axis_ij[0]
    x = res.grid.x - res.grid.x[ix]
    field = (x, res.z, res.xz_p_plus.T)
    z_lo, z_hi, x_in = region_extent(field, -20.0)
    lam = bowl.wavelength
    keep_z = (res.z >= z_lo - lam) & (res.z <= z_hi + lam)
    keep_x = np.abs(x) <= x_in + lam
    inset = (x[keep_x], res.z[keep_z], res.xz_p_plus.T[np.ix_(keep_x, keep_z)])
    i = res.focal_index
    za = np.linspace(res.z[0], res.z[-1], 4001)
    linear = f"linear (C.1), p0 = {p0 / 1e6:g} MPa"
    feat = features_from_results(
        res,
        axial_curves=[
            (res.z, res.axis_p_plus, {**P_PLUS, "label": r"$p_+$"}),
            (res.z, np.abs(res.axis_p_minus), {**P_MINUS, "label": r"$|p_-|$"}),
            (za, np.abs(bowl.axial_pressure(za)), {**LINEAR, "label": linear}),
        ],
        lateral_curves=[
            (x, res.xz_p_plus[i], {**P_PLUS, "label": r"$p_+$"}),
            (x, np.abs(res.xz_p_minus[i]), {**P_MINUS, "label": r"$|p_-|$"}),
        ],
    )
    title = (
        f"Focused bowl in water, IEC TS 63587 benchmark C.2: p0 = {p0 / 1e6:g} MPa, "
        f"aperture {bowl.aperture * 1e3:g} mm, radius of curvature {bowl.D * 1e3:g} mm, {bowl.frequency / 1e6:g} MHz"
    )
    return figure(bowl, field, inset, feat, text_color, title=title, quantity="p_+", z_view=(-2e-3, res.z[-1]),
                  source_z=float(res.z[0])), feat  # fmt: skip


def waveform_figure(res: ao.Results, bowl: FocusedBowl, p0: float, text_color: str) -> Any:
    """Figure 2, as Figure C.2: the focal waveform, its spectrum, and the harmonics along the axis."""
    import matplotlib.pyplot as plt

    m = res.metrics()
    i = res.focal_index
    h = res.axis[i]  # harmonics at the beam maximum (retarded time)
    k = h.size
    nt = 64 * k
    f0 = bowl.frequency
    period = 1 / f0
    p = waveform(h, nt)
    # two periods, the peak positive pressure at t = 0
    j = int(np.argmax(p))
    t = (np.arange(2 * nt) - nt) / nt * period
    wave = np.roll(np.tile(p, 2), nt - j)
    p_lin = np.abs(bowl.axial_pressure(res.z[i]))
    lin = p_lin * np.cos(2 * np.pi * f0 * t)  # the linear waveform at the same point, in phase at t = 0

    plt.rcParams.update({
        "font.size": 9, "text.color": text_color, "axes.labelcolor": text_color, "axes.edgecolor": text_color,
        "xtick.color": text_color, "ytick.color": text_color, "axes.titlecolor": text_color,
    })  # fmt: skip
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 9.0), layout="constrained", height_ratios=[1.2, 1.0, 1.0])
    a, b, c = axes
    a.plot(t * 1e6, wave / 1e6, **P_PLUS, label=f"p0 = {p0 / 1e6:g} MPa")
    a.plot(t * 1e6, lin / 1e6, **LINEAR, label=f"linear (C.1), {p_lin / 1e6:.2f} MPa")
    pp, pm = m["focus"]["p_positive"], m["focus"]["p_negative"]
    a.axhline(pp / 1e6, color="#b2182b", lw=0.5, ls=":")
    a.axhline(pm / 1e6, color="#b2182b", lw=0.5, ls=":")
    a.annotate(rf"$p_+$ = {pp / 1e6:.2f} MPa", (t[-1] * 1e6, pp / 1e6), xytext=(-4, 3), textcoords="offset points",
               ha="right", va="bottom", fontsize=8, color="#b2182b")  # fmt: skip
    a.annotate(rf"$p_-$ = {pm / 1e6:.2f} MPa".replace("-", MINUS), (t[-1] * 1e6, pm / 1e6), xytext=(-4, -3),
               textcoords="offset points", ha="right", va="top", fontsize=8, color="#b2182b")  # fmt: skip
    a.set_xlim(t[0] * 1e6, t[-1] * 1e6)
    a.set_ylim(1.35 * min(pm, -p_lin) / 1e6, 1.2 * max(pp, p_lin) / 1e6)  # room for the labels and the legend
    a.set_xlabel(r"time [$\mu$s]")
    a.set_ylabel("pressure [MPa]")
    a.set_title(f"(a) waveform at the beam maximum, z = {res.z[i] * 1e3:.2f} mm from the apex", loc="left",
                fontsize=9)  # fmt: skip
    a.legend(loc="lower left", fontsize=7.5, frameon=False)
    a.grid(alpha=0.25)

    n = np.arange(1, k + 1)
    amp = np.abs(h) / p0  # the harmonics normalised by the source pressure, against f / f0 = n
    b.vlines(n, 0, amp, color=P_PLUS["color"], lw=1.4)
    b.plot(n, amp, "o", ms=3.5, color=P_PLUS["color"])
    # up to the last harmonic above 0.1% of the fundamental: the rest are too small to see on a linear scale
    shown = int(np.nonzero(amp >= 1e-3 * amp[0])[0][-1]) + 1
    b.set_xlim(0, shown + 1)
    b.set_xticks(np.arange(0, shown + 2))
    b.set_ylim(0, 1.1 * amp[0])
    b.set_xlabel(r"$f\,/\,f_0$")
    b.set_ylabel(r"$|P_n|\,/\,p_0$")
    b.set_title(f"(b) spectrum at the beam maximum: harmonics 1 to {shown} of {k} (the others are below 0.1% of "
                f"$|P_1|$), $p_0$ = {p0 / 1e6:g} MPa", loc="left", fontsize=9)  # fmt: skip
    b.grid(alpha=0.25)

    colors = ["#08519c", "#e66101", "#7b3294", "#1b7837"]
    for q in range(min(4, k)):
        c.plot(res.z * 1e3, np.abs(res.axis[:, q]) / 1e6, color=colors[q], lw=1.2, label=f"$|P_{q + 1}|$")
    c.axvline(res.z[i] * 1e3, color="#b2182b", lw=0.8, ls=":", label="beam maximum")
    c.set_xlim(res.z[0] * 1e3, res.z[-1] * 1e3)
    c.set_xlabel("z from the bowl's apex [mm]")
    c.set_ylabel("amplitude on axis [MPa]")
    c.set_title("(c) the first harmonics along the axis", loc="left", fontsize=9)
    c.legend(loc="upper left", fontsize=7.5, frameon=False)
    c.grid(alpha=0.25)
    for ax in axes:
        ax.set_facecolor("none")
    fig.suptitle(f"Focused bowl in water, IEC TS 63587 benchmark C.2, p0 = {p0 / 1e6:g} MPa: focal waveform and "
                 "spectrum (as Figure C.2)", fontsize=9)  # fmt: skip
    return fig


def save_three(fig: Any, stem: Path, dpi: int) -> list[Path]:
    """The figure as .png, _transparent.png and .jpg."""
    files = [stem.with_suffix(".png"), stem.parent / f"{stem.name}_transparent.png", stem.with_suffix(".jpg")]
    trim = {"bbox_inches": "tight", "pad_inches": 0.1}
    fig.savefig(files[0], dpi=dpi, facecolor="white", **trim)
    fig.savefig(files[1], dpi=dpi, transparent=True, **trim)
    fig.savefig(files[2], dpi=dpi, facecolor="white", pil_kwargs={"quality": 92}, **trim)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the folder of a C.2 run (benchmark_c2_pywave3d.py -o)")
    parser.add_argument("-o", "--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--name", default="bowl_c2", help="file name stem")
    parser.add_argument("--text-color", default="#303030")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import matplotlib

    matplotlib.use("Agg")
    summary = json.loads((args.run / "c2_summary.json").read_text())
    inputs = summary["inputs"]
    p0 = float(inputs["p0"])
    bowl = FocusedBowl(p0=p0, **{k: inputs[k] for k in ("frequency", "sound_speed", "density", "aperture",
                                                        "radius_of_curvature")})  # fmt: skip
    run = finest_run(args.run)
    res = ao.Results.load(run)
    log.info("read %s: %d planes of %d x %d points, %d harmonics", run, res.z.size, res.grid.nx, res.grid.ny,
             res.axis.shape[1])  # fmt: skip
    args.output.mkdir(parents=True, exist_ok=True)
    fig, feat = field_figure(res, bowl, p0, args.text_color)
    files = save_three(fig, args.output / f"{args.name}_field", args.dpi)
    fig2 = waveform_figure(res, bowl, p0, args.text_color)
    files += save_three(fig2, args.output / f"{args.name}_waveform", args.dpi)
    w = feat["widths"]
    log.info("p+ focal widths: %s", ", ".join(f"{lv:g} dB {w[lv] * 1e3:.3f} mm" for lv in w))
    for f in files:
        log.info("wrote %s (%.0f kB)", f, f.stat().st_size / 1e3)


if __name__ == "__main__":
    main()
