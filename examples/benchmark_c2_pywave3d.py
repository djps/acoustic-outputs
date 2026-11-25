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
software and its versions, the quantities of Table C.1 and their convergence, the linear check),
and for each run, in a subfolder, its ``metrics.json``, ``axial.txt`` ... and ``numerics.json``:
the grid and its steps (dx, dy, dz, points per wavelength of the fundamental and of the highest
harmonic), the time sampling (samples per period, dt), the march (Strang splitting, step and
number of steps), the nonlinear step (method, thresholds, Godunov time step and CFL, where the
shock-capturing scheme took over, the number of steps of each kind and their sub-steps), the PML
(width, strength, profile, reflection), the source plane (position, Rayleigh quadrature, radial
step), the precision, threads and timings::

    PYTHONPATH=~/Wave3D/pyWave3D/src python examples/benchmark_c2_pywave3d.py --workers 16
    PYTHONPATH=~/Wave3D/pyWave3D/src python examples/benchmark_c2_pywave3d.py --p0 0.45e6 --harmonics 64
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import math
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
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
#: the PML's round-trip reflection at normal incidence
PML_REFLECTION = 1e-4
#: the source plane: Rayleigh quadrature points per wavelength on the bowl, and its radial step
RAYLEIGH_PPW = 8.0
RADIAL_STEP_PER_WAVELENGTH = 1 / 16


def recorder_for_acoustic_outputs(margin: int, plane_precision: str) -> Any:
    """A factory of ``acoustic_outputs`` recorders to put in place of pyWave3D's (same arguments
    and methods). They leave out the ``margin`` points at each edge of the plane (the PML), so
    nothing recorded or stored includes the absorbing layer, and keep the whole planes (source,
    face, focal, peak, final) at ``plane_precision``; the march itself stays in double precision."""

    def make(grid: Any, n_harmonics: int, **kwargs: Any) -> ao.PlaneRecorder:
        g = ao.Grid(nx=grid.nx, ny=grid.ny, nz=grid.nz, nt=grid.nt, dx=grid.dx, dy=grid.dy, dz=grid.dz,
                    x0=grid.x0, y0=grid.y0, z0=grid.z0)  # fmt: skip
        settings = kwargs.pop("settings")
        kwargs["settings"] = ao.RecordSettings(**dataclasses.asdict(settings), margin=margin,
                                               plane_precision=plane_precision)  # fmt: skip
        return ao.PlaneRecorder(g, n_harmonics, **kwargs)

    return make


def pml_margin(layer: Any, grid: Any) -> int:
    """The number of grid points of the PML at each edge of the plane (the most of the four)."""
    sx, sy = layer.profiles(grid)
    ends = [int(np.argmax(s == 0)) for s in (sx, sy)] + [int(np.argmax(s[::-1] == 0)) for s in (sx, sy)]
    return max(ends)


def git_version(path: Path) -> str | None:
    """The short commit of the repository holding ``path``, marked if it has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no", "."],
                               capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit} with uncommitted changes" if dirty else commit


def software() -> dict[str, Any]:
    """The versions of the codes that ran, for the report."""
    import pywave3d as pw
    import scipy

    return {
        "acoustic_outputs": {"version": ao.__version__, "git": git_version(Path(ao.__file__).parent)},
        "pywave3d": {"version": pw.__version__, "git": git_version(Path(pw.__file__).parent)},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cpus": os.cpu_count(),
    }


def setup(bowl: FocusedBowl, *, ppw: float, harmonics: int, half_width: float, z_start: float, z_end: float,
          layer_width: float, nonlinear: bool, workers: int) -> tuple[Any, Any, Any, Any]:  # fmt: skip
    """pyWave3D's medium, grid, absorbing layer and settings of a run."""
    import pywave3d as pw
    from pywave3d.linear import AbsorbingLayer

    medium = pw.Medium(sound_speed=bowl.sound_speed, density=bowl.density, nonlinearity=BETA, diffusivity=DIFFUSIVITY)
    lx = 2 * half_width
    grid = pw.Grid.from_lengths(lx, lx, z_end - z_start, wavelength=bowl.wavelength,
                                points_per_wavelength=(ppw, ppw, ppw), nt=2 * harmonics,
                                origin=(-half_width, -half_width, z_start))  # fmt: skip
    layer = AbsorbingLayer.for_reflection(layer_width, reflection=PML_REFLECTION, kind="pml")
    settings = pw.SolverSettings(nonlinearity=nonlinear, absorbing_layer=True, workers=workers)
    return medium, grid, layer, settings


def numerics(bowl: FocusedBowl, grid: Any, layer: Any, settings: Any, operator: Any, *, ppw: float, z_start: float,
             res: ao.Results, stats: dict[str, Any] | None, seconds: float, margin: int,
             plane_precision: str) -> dict[str, Any]:  # fmt: skip
    """The discretisation and the numerical parameters of a run, as the report of TS 63587 asks."""
    lam = bowl.wavelength
    period = 1 / bowl.frequency
    k = grid.n_harmonics
    out: dict[str, Any] = {
        "points_per_wavelength_requested": ppw,
        "grid": {
            "nx": grid.nx, "ny": grid.ny, "planes": grid.nz, "dx": grid.dx, "dy": grid.dy, "dz": grid.dz,
            "x0": grid.x0, "y0": grid.y0, "z0": grid.z0, "z_last": grid.z0 + grid.n_steps * grid.dz,
            "lengths": list(grid.lengths),
        },
        "points_per_wavelength": {"x": lam / grid.dx, "y": lam / grid.dy, "z": lam / grid.dz},
        "points_per_wavelength_highest_harmonic": {"x": lam / (k * grid.dx), "z": lam / (k * grid.dz)},
        "time": {"samples_per_period": grid.nt, "dt": period / grid.nt, "harmonics": k,
                 "highest_frequency": k * bowl.frequency},
        "march": {
            "scheme": "Strang splitting in z: half linear step, nonlinear step, half linear step (consecutive half "
                      "steps merged into one between planes that are not recorded)",
            "step_dz": grid.dz,
            "steps": grid.n_steps,
            "record_every": settings.record_every,
        },
        "linear_step": {
            "diffraction": "exact angular spectrum (one-way, non-paraxial)" if not settings.paraxial
            else "paraxial (KZK)",
            "absorption": "thermoviscous, diffusivity delta" if settings.absorption else "off",
            "harmonic_tolerance": settings.harmonic_tolerance,
        },
        "absorbing_layer": {
            "kind": layer.kind, "width": layer.width, "points_in_layer": layer.width / grid.dx,
            "strength_Np_per_m": layer.strength, "profile_power": layer.power, "pml_levels": layer.levels,
            "reflection_normal_incidence": PML_REFLECTION,
        },
        "source": {
            "plane_z_from_apex": z_start,
            "field": "Rayleigh integral over the bowl (Gauss-Legendre), exp(+i omega t)",
            "rayleigh_points_per_wavelength": RAYLEIGH_PPW,
            "radial_step": RADIAL_STEP_PER_WAVELENGTH * lam,
            "onto_grid": "linear interpolation in the radius from the beam axis",
        },
        "computation_precision": "complex64" if settings.single_precision else "complex128",
        "stored": {
            "region": "the interior of the plane: the PML is left out of everything recorded and stored",
            "pml_points_removed_per_edge": margin,
            "nx": res.grid.nx, "ny": res.grid.ny,
            "x_range": [float(res.grid.x[0]), float(res.grid.x[-1])],
            "y_range": [float(res.grid.y[0]), float(res.grid.y[-1])],
            "whole_planes_precision": "complex64" if plane_precision == "single" else "complex128",
            "whole_planes": "source, face, focal, spatial-peak and final planes (all harmonics); the axis, the "
                            "lines through it and every derived quantity are computed and kept in double precision",
        },
        "workers": settings.workers,
        "timings": {**res.timings, "wall": seconds},
    }  # fmt: skip
    if operator is None:
        out["nonlinear_step"] = None
    else:
        out["nonlinear_step"] = {
            "method": settings.method,
            "switch": "spectral (Runge-Kutta on the harmonics) until the waveform steepens, then Godunov "
                      "(MUSCL, shock-capturing) on a time grid",
            "switch_threshold": settings.switch_threshold,
            "amplitude_threshold": settings.amplitude_threshold,
            "rk_safety": operator.rk_safety,
            "godunov_samples_per_period": operator.n_godunov,
            "godunov_dtau": period / operator.n_godunov,
            "cfl": settings.cfl,
            "godunov_from_z": None if math.isnan(res.godunov_from) else res.godunov_from,
            "substeps": stats if stats is not None else "not recorded (the run predates this record)",
        }  # fmt: skip
    return out


def run(bowl: FocusedBowl, *, ppw: float, harmonics: int, half_width: float, z_start: float, z_end: float,
        layer_width: float, nonlinear: bool, workers: int,
        plane_precision: str = "single") -> tuple[ao.Results, dict[str, Any]]:  # fmt: skip
    """March the bowl's field with pyWave3D, recorded by acoustic_outputs; the results and the numerics."""
    import pywave3d as pw
    import pywave3d.solver as pw_solver
    from pywave3d.model import Source

    t_run = time.perf_counter()
    lam = bowl.wavelength
    medium, grid, layer, settings = setup(bowl, ppw=ppw, harmonics=harmonics, half_width=half_width,
                                          z_start=z_start, z_end=z_end, layer_width=layer_width,
                                          nonlinear=nonlinear, workers=workers)  # fmt: skip
    # the source plane: the bowl's field at z_start (Rayleigh integral, exp(+i omega t) as pyWave3D),
    # axisymmetric about the grid's beam axis (the point nx/2, ny/2)
    xc, yc = grid.x[grid.nx // 2], grid.y[grid.ny // 2]
    rho = np.hypot(grid.x[:, None] - xc, grid.y[None, :] - yc)
    dr = RADIAL_STEP_PER_WAVELENGTH * lam
    radii = dr * np.arange(int(np.ceil(rho.max() / dr)) + 2)
    t0 = time.perf_counter()
    profile = bowl.rayleigh(radii, [z_start], points_per_wavelength=RAYLEIGH_PPW, time_sign=1)[:, 0]
    log.info("source plane at z = %.2f mm from the apex: Rayleigh integral at %d radii, %.1f s", z_start * 1e3,
             radii.size, time.perf_counter() - t0)  # fmt: skip
    plane = np.interp(rho, radii, profile.real) + 1j * np.interp(rho, radii, profile.imag)
    source = Source(x=grid.x, y=grid.y, harmonics=plane[None])
    lx = 2 * half_width
    model = pw.WaveModel(medium=medium, frequency=bowl.frequency, lengths=(lx, lx, z_end - z_start),
                         origin=(-half_width, -half_width, z_start), source=source, layer=layer)  # fmt: skip
    margin = pml_margin(layer, grid)
    original = pw_solver.PlaneRecorder
    pw_solver.PlaneRecorder = recorder_for_acoustic_outputs(margin, plane_precision)  # the march feeds acoustic_outputs
    try:
        solver = pw.WaveSolver(model, grid, settings)
        log.info("grid: %s", solver.grid.summary())
        stats = None
        op = solver.nonlinear
        if op is not None:  # count the nonlinear steps by method, and their sub-steps
            stats = {"steps": 0, "spectral_steps": 0, "godunov_steps": 0, "max": 0, "total": 0}
            apply = op.apply

            def counted(*args: Any, **kwargs: Any) -> Any:
                method = apply(*args, **kwargs)
                stats["steps"] += 1
                stats[f"{method}_steps"] = stats.get(f"{method}_steps", 0) + 1
                stats["max"] = max(stats["max"], op.last_substeps)
                stats["total"] += op.last_substeps
                return method

            op.apply = counted
        res = solver.run()
    finally:
        pw_solver.PlaneRecorder = original
    assert isinstance(res, ao.Results)
    res = dataclasses.replace(res, focusing_depth=bowl.D - z_start)  # focal shift from the first plane
    info = numerics(bowl, solver.grid, layer, settings, op, ppw=ppw, z_start=z_start, res=res, stats=stats,
                    seconds=time.perf_counter() - t_run, margin=margin, plane_precision=plane_precision)  # fmt: skip
    return res, info


def save_run(res: ao.Results, info: dict[str, Any], folder: Path, **kwargs: Any) -> None:
    """Save a run's results and, beside them, its numerics (``numerics.json``)."""
    res.save(folder, **kwargs)
    (folder / "numerics.json").write_text(json.dumps(info, indent=2, default=float) + "\n")


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
    parser.add_argument("--plane-precision", choices=["single", "double"], default="single",
                        help="precision of the whole planes stored (the march is double precision)")  # fmt: skip
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("pywave3d").setLevel(logging.WARNING)

    bowl = FocusedBowl(p0=args.p0, **C2)
    z_start = bowl.h + 3 * bowl.wavelength  # three wavelengths beyond the aperture plane
    common = {"harmonics": args.harmonics, "half_width": args.half_width, "z_start": z_start, "z_end": args.z_end,
              "layer_width": args.layer, "workers": args.workers,
              "plane_precision": args.plane_precision}  # fmt: skip
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
            "pml_reflection": PML_REFLECTION,
            "harmonics": args.harmonics,
            "samples_per_period": 2 * args.harmonics,
            "z_range_from_apex": [z_start, args.z_end],
            "recording": "acoustic_outputs.PlaneRecorder, fed by the march; the PML is left out of what is "
            f"recorded and stored, and whole planes are stored in {args.plane_precision} precision",
            "software": software(),
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "runs": {},
    }

    # the linear check: the same grid without nonlinearity, against the analytic axis (C.1)
    t0 = time.perf_counter()
    lin, info = run(bowl, ppw=args.ppw[0], nonlinear=False, **common)
    analytic = np.abs(bowl.axial_pressure(lin.z))
    summary["linear_check"] = {
        "points_per_wavelength": args.ppw[0],
        "axis_vs_analytic_C1": error_norms(np.abs(lin.axis[:, 0]), analytic),
        "focal_pressure_analytic": bowl.focal_magnitude(),
        "seconds": time.perf_counter() - t0,
        "numerics": info,
    }
    save_run(lin, info, args.output / "linear")
    log.info("linear check: axis |p| relative L2 %.2e vs the analytic solution",
             summary["linear_check"]["axis_vs_analytic_C1"]["L2_rel"])  # fmt: skip
    del lin  # saved: free its planes before the next march
    gc.collect()

    for ppw in args.ppw:
        t0 = time.perf_counter()
        res, info = run(bowl, ppw=ppw, nonlinear=True, **common)
        name = f"nonlinear_ppw{ppw:g}"
        save_run(res, info, args.output / name, harmonics=6)
        summary["runs"][name] = {"points_per_wavelength": ppw, "grid": dataclasses.asdict(res.grid),
                                 "numerics": info, "table_C1": table_c1(res),
                                 "seconds": time.perf_counter() - t0}  # fmt: skip
        t = summary["runs"][name]["table_C1"]
        log.info("%s: p+ %.3f MPa, p- %.3f MPa at z = %.2f mm, shock amplitude %s, %.0f s", name, t["p_plus"] / 1e6,
                 t["p_minus"] / 1e6, t["beam_maximum_z"] * 1e3, t["shock_amplitude"],
                 summary["runs"][name]["seconds"])  # fmt: skip
        del res  # saved: free its planes before the next march
        gc.collect()
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
