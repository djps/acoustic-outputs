"""``acoustic-outputs``: the field parameters of a solver's output file.

    acoustic-outputs DATA [--kind KIND] [-o DIR] [--frequency F] [--sound-speed C] [--density RHO]
                     [--set KEY=VALUE ...] [--point NAME X Y Z ...] [-q | -v]
    acoustic-outputs --example KIND FILE      # write a synthetic dataset of KIND to try it

``DATA`` is a ``.npz``, ``.mat``, HDF5 or DOLFINx ``.xdmf`` file (see :mod:`acoustic_outputs.datasets`); the outputs
(``metrics.json``, ``axial.txt``, ``planes.txt``, ``focal_plane.txt``, ``results.npz`` and the
point waveforms) are written to ``DIR`` (default ``./outputs``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_outputs import __version__, synthetic
from acoustic_outputs.datasets import KINDS, from_dataset, save_dataset

log = logging.getLogger("acoustic_outputs")


def _parse_value(text: str) -> Any:
    """A ``--set`` value: JSON (numbers, booleans, lists) or else the string itself."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _example(kind: str, path: Path) -> Path:
    """Write a synthetic Gaussian-beam dataset of ``kind``."""
    beam = synthetic.GaussianBeam()
    small = beam.coarse()
    flat = synthetic.GaussianBeam(y=None)
    data = {
        "harmonics": lambda: synthetic.harmonics_dataset(beam),
        "time-series": lambda: synthetic.time_series_dataset(small),
        "kwave": lambda: synthetic.kwave_dataset(small),
        "fdtd": lambda: synthetic.fdtd_dataset(flat),
        "fullwave": lambda: synthetic.fullwave_dataset(small),
        "babelbrain": lambda: synthetic.babelbrain_dataset(beam),
        "msound": lambda: synthetic.msound_dataset(beam),
        "focus": lambda: synthetic.focus_dataset(synthetic.GaussianBeam(z=np.linspace(15e-3, 45e-3, 121))),
        "axisymmetric": lambda: synthetic.axisymmetric_dataset(beam),
        "unstructured": lambda: synthetic.unstructured_mesh(
            synthetic.GaussianBeam(
                x=np.linspace(-5e-3, 5e-3, 21), y=np.linspace(-5e-3, 5e-3, 21), z=np.linspace(20e-3, 40e-3, 41)
            )
        ),
    }[kind]()
    return save_dataset(path, kind, **data)


def summary(metrics: dict[str, Any]) -> list[str]:
    """A few lines on the focus, the power and the main widths."""
    f = metrics["focus"]

    def mm(v: float) -> str:
        return "n/a" if v is None or not math.isfinite(v) else f"{v * 1e3:.3f} mm"

    lvl = next((k.removeprefix("focal_width_x_") for k in metrics if k.startswith("focal_width_x_")), "6dB")
    return [
        f"focus ({metrics['focal_quantity']}) at z = {mm(f['z'])}{' (on the edge)' if f['on_edge'] else ''}: "
        f"p+ = {f['p_positive'] / 1e6:.4g} MPa, p- = {f['p_negative'] / 1e6:.4g} MPa, "
        f"I = {f['intensity'] / 1e4:.4g} W/cm^2",
        f"power: first plane {metrics['power_source']:.4g} W, focal plane {metrics['power_focal_plane']:.4g} W",
        f"-{lvl}: length {mm(metrics.get(f'focal_length_{lvl}', math.nan))}, "
        f"width x {mm(metrics.get(f'focal_width_x_{lvl}', math.nan))}, "
        f"y {mm(metrics.get(f'focal_width_y_{lvl}', math.nan))}",
        f"MI {metrics['iec']['derated']['MI']:.3g}, TIS {metrics['iec']['derated']['TIS']:.3g}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acoustic-outputs",
        description="Acoustic field parameters (IEC and beam metrics) of a solver's output.",
    )
    parser.add_argument("data", nargs="?", type=Path, help="dataset file (.npz, .mat, .h5, .xdmf)")
    parser.add_argument("--kind", choices=KINDS, help="kind of solver output (default: the file's 'kind')")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--frequency", type=float, help="base frequency [Hz]")
    parser.add_argument("--sound-speed", type=float, help="sound speed c0 [m/s]")
    parser.add_argument("--density", type=float, help="density rho0 [kg/m^3]")
    parser.add_argument("--n-harmonics", type=int, help="harmonics fitted to time series")
    parser.add_argument("--set", metavar="KEY=VALUE", action="append", default=[], help="set a dataset key")
    parser.add_argument("--point", nargs=4, metavar=("NAME", "X", "Y", "Z"), action="append", default=[],
                        help="write the waveform at the point (x, y, z) [m] to NAME")  # fmt: skip
    parser.add_argument("--harmonics", type=int, default=4, help="harmonics written to the files")
    parser.add_argument("--example", metavar="KIND", choices=KINDS, help="write a synthetic dataset to DATA")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s" if args.quiet else "%(message)s",
    )
    if args.data is None:
        parser.error("a dataset file is needed")
    if args.example:
        path = _example(args.example, args.data)
        log.info("wrote a synthetic %s dataset to %s", args.example, path)
        return 0
    overrides: dict[str, Any] = {
        "frequency": args.frequency,
        "sound_speed": args.sound_speed,
        "density": args.density,
        "n_harmonics": args.n_harmonics,
    }
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--set {item!r}: expected KEY=VALUE")
        overrides[key.strip()] = _parse_value(value)
    points = [(name, (float(x), float(y), float(z))) for name, x, y, z in args.point]
    try:
        res = from_dataset(args.data, args.kind, points=points, **overrides)
    except (KeyError, ValueError, NotImplementedError, OSError) as err:
        msg = err.args[0] if isinstance(err, KeyError) and err.args else err
        print(f"acoustic-outputs: {msg}", file=sys.stderr)
        return 1
    written = res.save(args.output, harmonics=args.harmonics)
    for line in summary(res.metrics(args.harmonics)):
        log.info(line)
    log.info("wrote %d files to %s", len(written), args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
