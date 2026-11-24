"""IEC TS 63587, benchmark C.1: the linear field of a uniformly vibrating spherically focused bowl.

Two uses:

1. The built-in run (default): the transducer of benchmark C.2 (aperture 100 mm, radius of
   curvature 90 mm, 1 MHz) in water, at its linear level, simulated by an angular spectrum
   model from a Rayleigh source plane at two grid spacings, compared with the analytic axis and
   with the Rayleigh integral on the longitudinal plane. It writes the report Annex C asks for
   (inputs, how they were put into the model, discrepancies, convergence)::

       python examples/benchmark_c1_bowl.py -o outputs/c1
       python examples/benchmark_c1_bowl.py -o outputs/c1_lossy --attenuation 5 --plot

2. Your own solver (``--results``): simulate the same bowl with any solver (k-Wave, FOCUS,
   OptimUS, a finite-element model, ...), read its output with ``acoustic-outputs`` (any dataset
   this package reads), and compare its axis with the analytic solution. ``--apex-z`` is the
   position of the bowl's apex in your planes' z::

       python examples/benchmark_c1_bowl.py --results my_kwave_bowl.npz --apex-z 0.0 \\
           --frequency 1e6 --sound-speed 1485 --density 998

The same is available as ``acoustic-outputs --benchmark bowl``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from acoustic_outputs import from_dataset
from acoustic_outputs.benchmarks import FocusedBowl, compare_axis, run_bowl_benchmark, save_report, summary


def plot(report: dict, out: Path) -> None:
    """Axis |p| against the analytic solution, and the error along the axis (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = report["_results"]
    bowl = FocusedBowl(**{k: v for k, v in report["inputs"].items() if k != "derived"})
    z = res.z
    ref = np.abs(bowl.axial_pressure(z))
    sim = np.abs(res.axis[:, 0])
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    a.plot(z * 1e3, ref / 1e6, "k-", label="analytic (C.1)")
    a.plot(z * 1e3, sim / 1e6, "r--", label=f"{report['model']['method']} model")
    a.axvline(bowl.D * 1e3, color="0.6", lw=0.8)
    a.set_ylabel("|p| on axis [MPa]")
    a.legend()
    b.plot(z * 1e3, (sim - ref) / ref.max(), "r-")
    b.set_xlabel("z from the apex [mm]")
    b.set_ylabel("error / max |p|")
    fig.tight_layout()
    fig.savefig(out / "benchmark_axis.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/c1_bowl"))
    parser.add_argument("--aperture", type=float, default=100e-3, help="aperture diameter [m]")
    parser.add_argument("--radius-of-curvature", type=float, default=90e-3, help="[m]")
    parser.add_argument("--frequency", type=float, default=1e6)
    parser.add_argument("--sound-speed", type=float, default=1485.0)
    parser.add_argument("--density", type=float, default=998.0)
    parser.add_argument("--p0", type=float, default=15e3, help="rho c v0 [Pa]")
    parser.add_argument("--attenuation", type=float, default=0.0, help="alpha at f [Np/m]")
    parser.add_argument("--method", choices=["angular_spectrum", "rayleigh"], default="angular_spectrum")
    parser.add_argument("--ppw", type=float, nargs="+", default=[2.0, 4.0], help="grid points per wavelength")
    parser.add_argument("--results", type=Path, help="compare a dataset of your own solver instead")
    parser.add_argument("--apex-z", type=float, default=0.0, help="the apex's z in that dataset [m]")
    parser.add_argument("--kind", help="the dataset's kind, if the file does not say (see acoustic-outputs --help)")
    parser.add_argument("--plot", action="store_true", help="also write benchmark_axis.png (matplotlib)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bowl = FocusedBowl(frequency=args.frequency, sound_speed=args.sound_speed, density=args.density,
                       aperture=args.aperture, radius_of_curvature=args.radius_of_curvature, p0=args.p0,
                       attenuation=args.attenuation)  # fmt: skip
    if args.results is not None:
        res = from_dataset(args.results, args.kind, frequency=args.frequency, sound_speed=args.sound_speed,
                           density=args.density)  # fmt: skip
        cmp = compare_axis(res, bowl, apex_z=args.apex_z)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "benchmark_comparison.json").write_text(json.dumps(cmp, indent=2, default=float) + "\n")
        print(json.dumps(cmp, indent=2, default=float))
        return
    report = run_bowl_benchmark(bowl, spacings=tuple(bowl.wavelength / n for n in args.ppw), method=args.method)
    save_report(report, args.output)
    if args.plot:
        plot(report, args.output)
    print("\n".join(summary(report)))


if __name__ == "__main__":
    main()
