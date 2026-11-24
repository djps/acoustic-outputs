"""IEC TS 63587, benchmark C.2: the nonlinear field of the focused bowl in water, by pyWave3D.

The benchmark inputs of C.2.2: an axisymmetric spherically curved bowl, aperture 100 mm, radius of
curvature 90 mm, vibrating uniformly (velocity v0) at 1 MHz in the continuous-wave regime, in water
(c = 1485 m/s, beta = 3.5, rho = 998 kg/m^3, diffusivity delta = 4.33e-6 m^2/s), at a source
pressure p0 = rho c v0. The default level is the quasi-linear one of Figure C.2, p0 = 0.15 MPa.

The model is pyWave3D (one-way Westervelt equation, spatial marching of the harmonics in retarded
time, exact angular-spectrum diffraction and thermoviscous absorption, spectral / shock-capturing
nonlinear step; ``~/Wave3D/pyWave3D``). The source is the bowl's own field on a plane a few
wavelengths beyond its aperture plane, from the Rayleigh integral over the bowl
(:class:`acoustic_outputs.benchmarks.FocusedBowl`): nonlinear distortion between the bowl and that
plane is negligible at these levels. This is how the Westervelt results of C.2 set the boundary
condition on a curved source, rather than with the flat equivalent source of the KZK model (C.2.2,
last item), which is made for the parabolic approximation. A PML at the lateral edges keeps the
periodic angular spectrum from wrapping the field round.

pyWave3D's march feeds this package's :class:`~acoustic_outputs.recorder.PlaneRecorder` plane by
plane (its own recorder is swapped for it), so every metric is computed by ``acoustic_outputs``.
The planes' z are distances from the bowl's apex. The run is repeated linearly on the same grid,
which must reproduce the analytic axis of C.1, and nonlinearly at two resolutions, for the
convergence that TS 63587 (5.3.1) asks for.

Outputs (``-o``, default ``outputs/c2_pywave3d``): ``c2_summary.json`` (inputs, model, the
quantities of Table C.1 and their convergence, the linear check), and each run's ``metrics.json``,
``axial.txt`` ... in a subfolder::

    PYTHONPATH=~/Wave3D/pyWave3D/src python examples/benchmark_c2_pywave3d.py --workers 16
    PYTHONPATH=~/Wave3D/pyWave3D/src python examples/benchmark_c2_pywave3d.py --p0 0.45e6 --harmonics 64
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# pyWave3D checks its annotations at run time; its run() is annotated to return its own Results,
# and here it returns acoustic_outputs' (the recorder is swapped), so pyWave3D's checks are turned
# off (its documented switch, read when pywave3d is imported)
os.environ.setdefault("PYWAVE3D_NO_BEARTYPE", "1")

import acoustic_outputs as ao
from acoustic_outputs.benchmarks import FocusedBowl
from acoustic_outputs.compare import error_norms
from acoustic_outputs.results import main_lobe_crossings

log = logging.getLogger("benchmark_c2")

#: benchmark C.2.2: the transducer and the water
C2 = {"frequency": 1.0e6, "sound_speed": 1485.0, "density": 998.0, "aperture": 0.100, "radius_of_curvature": 0.090}
BETA = 3.5
DIFFUSIVITY = 4.33e-6


def recorder_for_acoustic_outputs(grid: Any, n_harmonics: int, **kwargs: Any) -> ao.PlaneRecorder:
    """An ``acoustic_outputs`` recorder in place of pyWave3D's (same arguments and methods)."""
    g = ao.Grid(nx=grid.nx, ny=grid.ny, nz=grid.nz, nt=grid.nt, dx=grid.dx, dy=grid.dy, dz=grid.dz,
                x0=grid.x0, y0=grid.y0, z0=grid.z0)  # fmt: skip
    settings = kwargs.pop("settings")
    kwargs["settings"] = ao.RecordSettings(**dataclasses.asdict(settings))
    return ao.PlaneRecorder(g, n_harmonics, **kwargs)


def run(bowl: FocusedBowl, *, ppw: float, harmonics: int, half_width: float, z_start: float, z_end: float,
        layer_width: float, nonlinear: bool, workers: int) -> ao.Results:  # fmt: skip
    """March the bowl's field with pyWave3D, recorded by acoustic_outputs."""
    import pywave3d as pw
    import pywave3d.solver as pw_solver
    from pywave3d.linear import AbsorbingLayer
    from pywave3d.model import Source

    lam = bowl.wavelength
    medium = pw.Medium(sound_speed=bowl.sound_speed, density=bowl.density, nonlinearity=BETA, diffusivity=DIFFUSIVITY)
    lx = 2 * half_width
    grid = pw.Grid.from_lengths(lx, lx, z_end - z_start, wavelength=lam, points_per_wavelength=(ppw, ppw, ppw),
                                nt=2 * harmonics, origin=(-half_width, -half_width, z_start))  # fmt: skip
    # the source plane: the bowl's field at z_start (Rayleigh integral, exp(+i omega t) as pyWave3D),
    # axisymmetric about the grid's beam axis (the point nx/2, ny/2)
    xc, yc = grid.x[grid.nx // 2], grid.y[grid.ny // 2]
    rho = np.hypot(grid.x[:, None] - xc, grid.y[None, :] - yc)
    dr = lam / 16
    radii = dr * np.arange(int(np.ceil(rho.max() / dr)) + 2)
    t0 = time.perf_counter()
    profile = bowl.rayleigh(radii, [z_start], points_per_wavelength=8.0, time_sign=1)[:, 0]
    log.info("source plane at z = %.2f mm from the apex: Rayleigh integral at %d radii, %.1f s", z_start * 1e3,
             radii.size, time.perf_counter() - t0)  # fmt: skip
    plane = np.interp(rho, radii, profile.real) + 1j * np.interp(rho, radii, profile.imag)
    source = Source(x=grid.x, y=grid.y, harmonics=plane[None])
    model = pw.WaveModel(medium=medium, frequency=bowl.frequency, lengths=(lx, lx, z_end - z_start),
                         origin=(-half_width, -half_width, z_start), source=source,
                         layer=AbsorbingLayer.for_reflection(layer_width, reflection=1e-4, kind="pml"))  # fmt: skip
    settings = pw.SolverSettings(nonlinearity=nonlinear, absorbing_layer=True, workers=workers)
    original = pw_solver.PlaneRecorder
    pw_solver.PlaneRecorder = recorder_for_acoustic_outputs  # the march feeds acoustic_outputs
    try:
        res = pw.WaveSolver(model, grid, settings).run()
    finally:
        pw_solver.PlaneRecorder = original
    assert isinstance(res, ao.Results)
    return dataclasses.replace(res, focusing_depth=bowl.D - z_start)  # focal shift from the first plane


def table_c1(res: ao.Results) -> dict[str, Any]:
    """The quantities of Table C.1: focal p+, p-, shock amplitude, and the -6 dB focal length and
    width of p+ and of p- (each about its own maximum on the axis), from the package's outputs."""
    m = res.metrics()
    nl = m["iec"]["nonlinear"]
    out: dict[str, Any] = {
        "beam_maximum_z": m["focus"]["z"],
        "p_plus": m["focus"]["p_positive"],
        "p_minus": m["focus"]["p_negative"],
        "shock_amplitude": nl.get("shock_amplitude"),
        "focal_length_6dB_p_plus": m["focal_length_6dB"],
        "focal_width_6dB_p_plus": m["focal_width_x_6dB"],
        "harmonics_at_focus": m["focus"]["harmonics"][:6],
    }
    # p-: its own maximum on the axis, and the -6 dB extents about it (package functions)
    pm = np.abs(res.axial("p_negative"))
    i = int(np.argmax(pm))
    lo, hi = main_lobe_crossings(res.z, pm, i, 10 ** (-6 / 20) * pm[i])
    ix = res.axis_ij[0]
    line = np.abs(res.xz_p_minus[i])
    xlo, xhi = main_lobe_crossings(res.grid.x, line, ix, 10 ** (-6 / 20) * line[ix])
    out.update(p_minus_max_z=float(res.z[i]), p_minus_max=float(-pm[i]), focal_length_6dB_p_minus=hi - lo,
               focal_width_6dB_p_minus=xhi - xlo)  # fmt: skip
    out["side_lobes_x"] = m["iec"]["side_lobes"]["x"]
    out["prefocal_postfocal"] = m["iec"]["side_lobes"]["axial"]
    out["steepness"] = nl.get("steepness")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/c2_pywave3d"))
    parser.add_argument("--p0", type=float, default=0.15e6, help="source pressure rho c v0 [Pa]")
    parser.add_argument("--ppw", type=float, nargs="+", default=[6.0, 9.0], help="points per wavelength (coarse, fine)")
    parser.add_argument("--harmonics", type=int, default=32, help="harmonics kept (nt = 2 K)")
    parser.add_argument("--half-width", type=float, default=0.065, help="half the transverse domain [m]")
    parser.add_argument("--layer", type=float, default=0.008, help="PML width [m]")
    parser.add_argument("--z-end", type=float, default=0.120, help="last plane, from the apex [m]")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("pywave3d").setLevel(logging.WARNING)

    bowl = FocusedBowl(p0=args.p0, **C2)
    z_start = bowl.h + 3 * bowl.wavelength  # three wavelengths beyond the aperture plane
    common = {"harmonics": args.harmonics, "half_width": args.half_width, "z_start": z_start, "z_end": args.z_end,
              "layer_width": args.layer, "workers": args.workers}  # fmt: skip
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "benchmark": "IEC TS 63587 C.2: nonlinear focused field in water",
        "inputs": {
            **C2,
            "p0": args.p0,
            "velocity_v0": bowl.velocity,
            "beta": BETA,
            "diffusivity": DIFFUSIVITY,
            "regime": "CW",
        },
        "model": {
            "solver": "pyWave3D: one-way Westervelt, spatial marching of harmonics in retarded time, exact "
            "angular-spectrum diffraction, thermoviscous absorption, spectral/MUSCL-Godunov nonlinear step",
            "source": f"the bowl's Rayleigh field on the plane z = {z_start * 1e3:.3f} mm from the apex",
            "transverse_domain": 2 * args.half_width,
            "pml_width": args.layer,
            "harmonics": args.harmonics,
            "recording": "acoustic_outputs.PlaneRecorder, fed by the march",
        },
        "runs": {},
    }

    # the linear check: the same grid without nonlinearity, against the analytic axis (C.1)
    t0 = time.perf_counter()
    lin = run(bowl, ppw=args.ppw[0], nonlinear=False, **common)
    analytic = np.abs(bowl.axial_pressure(lin.z))
    summary["linear_check"] = {
        "points_per_wavelength": args.ppw[0],
        "axis_vs_analytic_C1": error_norms(np.abs(lin.axis[:, 0]), analytic),
        "focal_pressure_analytic": bowl.focal_magnitude(),
        "seconds": time.perf_counter() - t0,
    }
    lin.save(args.output / "linear")
    log.info("linear check: axis |p| relative L2 %.2e vs the analytic solution",
             summary["linear_check"]["axis_vs_analytic_C1"]["L2_rel"])  # fmt: skip

    for ppw in args.ppw:
        t0 = time.perf_counter()
        res = run(bowl, ppw=ppw, nonlinear=True, **common)
        name = f"nonlinear_ppw{ppw:g}"
        res.save(args.output / name, harmonics=6)
        summary["runs"][name] = {"points_per_wavelength": ppw, "grid": dataclasses.asdict(res.grid),
                                 "table_C1": table_c1(res), "seconds": time.perf_counter() - t0}  # fmt: skip
        t = summary["runs"][name]["table_C1"]
        log.info("%s: p+ %.3f MPa, p- %.3f MPa at z = %.2f mm, shock amplitude %s, %.0f s", name, t["p_plus"] / 1e6,
                 t["p_minus"] / 1e6, t["beam_maximum_z"] * 1e3, t["shock_amplitude"],
                 summary["runs"][name]["seconds"])  # fmt: skip
    if len(args.ppw) > 1:
        a, b = (summary["runs"][f"nonlinear_ppw{p:g}"]["table_C1"] for p in args.ppw[-2:])
        summary["convergence"] = {
            k: (b[k] - a[k]) / abs(b[k]) if isinstance(a[k], float) and b[k] and math.isfinite(b[k]) else None
            for k in (
                "p_plus",
                "p_minus",
                "focal_length_6dB_p_plus",
                "focal_width_6dB_p_plus",
                "focal_length_6dB_p_minus",
                "focal_width_6dB_p_minus",
            )
        }
    (args.output / "c2_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    log.info("wrote %s", args.output / "c2_summary.json")


if __name__ == "__main__":
    main()
