"""Numerical schlieren images of benchmark C.2 (IEC TS 63587), for any number of saved runs.

As Figures 5 and 6 of Yuldashev et al., "HIFU beam", IEEE TUFFC 68(9) 2837 (2021): the pressure
waveforms p(tau) and their time derivatives |dp / d(omega tau)| (a "numerical schlieren image",
where the shock fronts are the dark lines), against the phase omega tau and

* (a, b) the depth z on the beam axis, from the harmonics recorded on the axis at every plane,
  with the depth where a shock forms on the axis (``Results.shock_formation``: the spectral
  exponent of the harmonics falls to 4/3) and the beam maximum;
* (c, d) the distance across the beam at one depth: from the x-z (or y-z) slice of harmonics when
  the run kept one (``benchmark_c2_pywave3d.py --slices``), at the depth ``--z`` (default: where
  the shock forms, else the beam maximum); otherwise from the window of the x-y plane it kept
  (``--xy-plane``), or from the focal plane, which every run keeps.

The waveforms are rebuilt from the K harmonics with a spectral filter (``--filter``: Lanczos sigma
by default) that damps the Gibbs ringing at a shock; the derivative is that of the filtered series,
sum i n sigma_n P_n exp(i n omega tau), not a finite difference. The derivative is shown in units of
|P_1| at the beam maximum on the axis, so a sinusoid of that amplitude has a peak of 1 (the
``steepness`` of the metrics). The phase is shifted so that the steepest rise of the waveform at
the beam maximum is at pi, as Figure C.2. The schlieren images use a power scale (``--gamma``, 0.5
by default; 1 is linear) so that the weaker fronts (edge waves, the front before it steepens) show
beside the shock.

Only the arrays needed are read from ``results.npz``. Each run gives one figure, named with the
regime's suffix as ``figure_bowl_c2.py`` (none for 0.15 MPa, ``_linear``, ``_shocked`` ...)::

    python examples/figure_bowl_c2_schlieren.py outputs/c2_pywave3d_p0_{0.015,0.15,0.45}MPa
    python examples/figure_bowl_c2_schlieren.py outputs/c2_pywave3d_p0_0.15MPa_slices --z 0.088
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from figure_bowl_c2 import finest_run, phase_axis, regime, save_three

from acoustic_outputs import iec
from acoustic_outputs.waveform import waveform

log = logging.getLogger("figure_bowl_c2_schlieren")

#: samples per period per harmonic of the rebuilt waveforms
SAMPLES_PER_HARMONIC = 16
FILTERS = ("sigma", "sigma2", "none")
KEYS = ("z", "axis", "x", "y", "slice_z", "xz_harmonics", "yz_harmonics", "xz_coordinate", "yz_coordinate",
        "xy_harmonics", "xy_x", "xy_y")  # fmt: skip


def spectral_filter(k: int, kind: str) -> np.ndarray:
    """The factors sigma_n, n = 1 .. K: Lanczos sigma = sinc(n / (K + 1)), its square, or none."""
    n = np.arange(1, k + 1)
    s = np.sinc(n / (k + 1))
    return {"sigma": s, "sigma2": s**2, "none": np.ones(k)}[kind]


def rebuild(h: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """The filtered waveform p and its derivative dp / d(omega tau), (nt, ...), from harmonics (K, ...)."""
    k = h.shape[0]
    shape = (k,) + (1,) * (h.ndim - 1)
    f = (spectral_filter(k, kind)).reshape(shape) * h.astype(complex)
    nt = SAMPLES_PER_HARMONIC * k
    n = np.arange(1, k + 1).reshape(shape)
    return waveform(f, nt), waveform(1j * n * f, nt)


def load(run: Path) -> dict[str, Any]:
    """The arrays of a saved run that the figure needs, and its meta data."""
    meta = json.loads((run / "results_meta.json").read_text())
    data: dict[str, Any] = {"meta": meta, "run": run}
    with np.load(run / "results.npz", allow_pickle=False) as npz:
        for k in KEYS:
            if k in npz.files:
                data[k] = npz[k]
    return data


def focal_plane_line(run: Path, direction: str, meta: dict[str, Any]) -> np.ndarray:
    """The harmonics on the line through the axis of the saved focal plane, (K, n)."""
    ix, iy = meta["axis_index"]
    with np.load(run / "results.npz", allow_pickle=False) as npz:
        plane = npz["focal_plane"]
    return plane[:, :, iy] if direction == "x" else plane[:, ix, :]


def transverse(d: dict[str, Any], direction: str, z_want: float,
               asked: bool) -> tuple[np.ndarray, np.ndarray, float, str]:  # fmt: skip
    """Harmonics (K, n) across the beam along ``direction``, their coordinate, their depth and their source."""
    meta = d["meta"]
    ix, iy = meta["axis_index"]
    sl = f"{direction}z_harmonics"
    if sl in d:
        j = int(np.argmin(np.abs(d["slice_z"] - z_want)))
        return d[sl][:, j, :], d[f"{direction}z_coordinate"], float(d["slice_z"][j]), f"{direction}-z slice"
    if asked:
        log.warning("%s kept no %s-z slice: --z is ignored, the plane it kept is used", d["run"], direction)
    if "xy_harmonics" in d:
        xa, ya = d["x"][ix], d["y"][iy]
        if direction == "x":
            j = int(np.argmin(np.abs(d["xy_y"] - ya)))
            return d["xy_harmonics"][:, :, j], d["xy_x"], float(meta["xy_z"]), "x-y window"
        j = int(np.argmin(np.abs(d["xy_x"] - xa)))
        return d["xy_harmonics"][:, j, :], d["xy_y"], float(meta["xy_z"]), "x-y window"
    coord = d["x"] if direction == "x" else d["y"]
    return focal_plane_line(d["run"], direction, meta), coord, float(d["z"][meta["focal_index"]]), "focal plane"


def figure(d: dict[str, Any], p0: float, direction: str, z_arg: float | None, half_width: float, kind: str,
           text_color: str, z_range: tuple[float, float] = (-12e-3, 8e-3),
           gamma: float = 0.5) -> tuple[Any, dict[str, Any]]:  # fmt: skip
    """The four panels for one run; also the shock formation and the depth across the beam. The axis
    panels span ``z_range`` about the beam maximum [m]; the schlieren images are on a power scale of
    exponent ``gamma`` (1: linear), so that weaker fronts show beside the shock."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm, TwoSlopeNorm

    meta = d["meta"]
    z, axis = d["z"], d["axis"]
    fi = meta["focal_index"]
    ix, iy = meta["axis_index"]
    k = axis.shape[1]
    sf = iec.shock_formation(z, axis, axis=1)
    z_sf = sf["z"]
    z_want = z_arg if z_arg is not None else (z_sf if math.isfinite(z_sf) else float(z[fi]))
    h_tr, coord, z_tr, source = transverse(d, direction, z_want, asked=z_arg is not None)
    a1 = abs(axis[fi, 0])

    # time frame: the steepest rise at the beam maximum on the axis at pi
    p_f, d_f = rebuild(axis[fi][:, None], kind)
    nt = p_f.shape[0]
    shift = nt // 2 - int(np.argmax(d_f[:, 0]))
    theta = 2 * np.pi * np.arange(nt) / nt

    zk = (z >= z[fi] + z_range[0]) & (z <= z[fi] + z_range[1])
    p_ax, d_ax = (np.roll(a, shift, axis=0) for a in rebuild(axis[zk].T, kind))
    c0 = d["x"][ix] if direction == "x" else d["y"][iy]
    keep = np.abs(coord - c0) <= half_width + 1e-12
    p_tr, d_tr = (np.roll(a, shift, axis=0) for a in rebuild(h_tr[:, keep], kind))
    s_tr = (coord[keep] - c0) * 1e3

    plt.rcParams.update({
        "font.size": 9, "text.color": text_color, "axes.labelcolor": text_color, "axes.edgecolor": text_color,
        "xtick.color": text_color, "ytick.color": text_color, "axes.titlecolor": text_color,
    })  # fmt: skip
    fig, ax = plt.subplots(2, 2, figsize=(10.0, 7.6), layout="constrained", height_ratios=[1.4, 1.0])

    def pressure(a: Any, y: np.ndarray, p: np.ndarray) -> None:
        lo, hi = float(p.min()) / 1e6, float(p.max()) / 1e6
        norm = TwoSlopeNorm(vmin=min(lo, -1e-9), vcenter=0.0, vmax=max(hi, 1e-9))
        im = a.pcolormesh(theta, y, p.T / 1e6, cmap="RdBu_r", norm=norm, shading="nearest", rasterized=True)
        fig.colorbar(im, ax=a, label="p [MPa]", pad=0.01)

    def schlieren(a: Any, y: np.ndarray, dp: np.ndarray) -> None:
        v = np.abs(dp.T) / a1
        im = a.pcolormesh(theta, y, v, cmap="Greys", norm=PowerNorm(gamma, vmin=0, vmax=float(v.max())),
                          shading="nearest", rasterized=True)  # fmt: skip
        fig.colorbar(im, ax=a, label=r"$|\partial p/\partial(\omega\tau)|\,/\,|P_1|_{\rm focus}$", pad=0.01)

    zmm = z[zk] * 1e3
    pressure(ax[0, 0], zmm, p_ax)
    schlieren(ax[0, 1], zmm, d_ax)
    for a in ax[0]:
        a.axhline(z[fi] * 1e3, color="#b2182b", lw=0.8, ls=":", label=f"beam maximum, {z[fi] * 1e3:.2f} mm")
        if math.isfinite(z_sf) and zmm[0] <= z_sf * 1e3 <= zmm[-1]:
            a.axhline(z_sf * 1e3, color="#e66101", lw=0.9, ls="--", label=f"shock forms, {z_sf * 1e3:.2f} mm")
        if source.endswith("slice") and zmm[0] <= z_tr * 1e3 <= zmm[-1]:
            a.axhline(z_tr * 1e3, color="#1b7837", lw=0.8, ls="-.", label=f"(c, d) at {z_tr * 1e3:.2f} mm")
        a.set_ylabel("z on the axis, from the apex [mm]")
        phase_axis(a)
        a.set_xlabel("")
    ax[0, 1].legend(loc="lower right", fontsize=7.5, framealpha=0.85)
    shock = f"shock forms at z = {z_sf * 1e3:.2f} mm" if math.isfinite(z_sf) else "no shock forms on the axis"
    ax[0, 0].set_title(f"(a) waveforms on the axis ({shock})", loc="left", fontsize=9)
    ax[0, 1].set_title("(b) their time derivative: numerical schlieren", loc="left", fontsize=9)

    pressure(ax[1, 0], s_tr, p_tr)
    schlieren(ax[1, 1], s_tr, d_tr)
    for a in ax[1]:
        a.set_ylabel(f"{direction} from the axis [mm]")
        phase_axis(a)
    where = f"z = {z_tr * 1e3:.2f} mm, {source}"
    ax[1, 0].set_title(f"(c) waveforms across the beam ({where})", loc="left", fontsize=9)
    ax[1, 1].set_title(f"(d) their time derivative ({where})", loc="left", fontsize=9)
    for a in ax.flat:
        a.set_facecolor("none")
    filt = {"sigma": "Lanczos sigma filter", "sigma2": "sigma-squared filter", "none": "no filter"}[kind]
    fig.suptitle(f"Focused bowl in water, IEC TS 63587 benchmark C.2, p0 = {p0 / 1e6:g} MPa: numerical schlieren "
                 f"({k} harmonics, {filt}; as Figs. 5 and 6 of Yuldashev et al. 2021)", fontsize=9)  # fmt: skip
    return fig, {"shock_formation_z": z_sf, "transverse_z": z_tr, "transverse_source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="the folders of C.2 runs (benchmark_c2_pywave3d.py -o)")
    parser.add_argument("--run", help="the run inside each folder (default: the finest nonlinear_ppw*)")
    parser.add_argument("--z", type=float, help="depth of panels (c, d) from the apex [m], when slices were kept "
                        "(default: where the shock forms, else the beam maximum)")  # fmt: skip
    parser.add_argument("--direction", choices=["x", "y"], default="x", help="the line across the beam")
    parser.add_argument("--half-width", type=float, default=1e-3, help="half the span across the beam [m]")
    parser.add_argument("--filter", choices=FILTERS, default="sigma", help="spectral filter of the waveforms")
    parser.add_argument("--z-range", type=float, nargs=2, default=[-12e-3, 8e-3], metavar=("BEFORE", "AFTER"),
                        help="span of the axis panels about the beam maximum [m]")  # fmt: skip
    parser.add_argument("--gamma", type=float, default=0.5, help="power scale of the schlieren images (1: linear)")
    parser.add_argument("-o", "--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--name", default="bowl_c2_schlieren", help="file name stem")
    parser.add_argument("--text-color", default="#303030")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    for folder in args.runs:
        p0 = float(json.loads((folder / "c2_summary.json").read_text())["inputs"]["p0"])
        run = folder / args.run if args.run else finest_run(folder)
        d = load(run)
        fig, info = figure(d, p0, args.direction, args.z, args.half_width, args.filter, args.text_color,
                           tuple(args.z_range), args.gamma)  # fmt: skip
        log.info("%s: p0 = %g MPa, shock forms on the axis at z = %.2f mm; across the beam at z = %.2f mm (%s)", run,
                 p0 / 1e6, info["shock_formation_z"] * 1e3, info["transverse_z"] * 1e3,
                 info["transverse_source"])  # fmt: skip
        for f in save_three(fig, args.output / f"{args.name}{regime(p0)[2]}", args.dpi):
            log.info("wrote %s (%.0f kB)", f, f.stat().st_size / 1e3)
        plt.close(fig)


if __name__ == "__main__":
    main()
