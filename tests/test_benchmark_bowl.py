"""Benchmark C.1 of IEC TS 63587 (the focused bowl), the angular spectrum projector and the error norms."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_outputs import compare
from acoustic_outputs.benchmarks import FocusedBowl, compare_axis, run_bowl_benchmark, save_report, simulate
from acoustic_outputs.cli import main
from acoustic_outputs.propagation import angular_spectrum


@pytest.fixture(scope="module")
def small() -> FocusedBowl:
    """A small, strongly focused bowl (20 mm aperture, 30 mm radius of curvature) that runs fast."""
    return FocusedBowl(aperture=20e-3, radius_of_curvature=30e-3)


# -- the analytic solution -------------------------------------------------------------------------


def test_geometry_and_focal_values() -> None:
    b = FocusedBowl()  # C.2's transducer
    assert b.a == 50e-3 and b.phi == pytest.approx(math.asin(50 / 90))
    assert b.h == pytest.approx(90e-3 * (1 - math.cos(b.phi)))
    assert b.a**2 + b.h**2 == pytest.approx(2 * b.D * b.h)  # the identity behind the regular form
    # C.2 and C.3, and the axial formula at the focus, which C.1 leaves singular
    assert abs(b.focal_pressure()) == pytest.approx(b.focal_magnitude(), rel=1e-14)
    assert complex(b.axial_pressure(b.D)) == pytest.approx(b.focal_pressure(), rel=1e-14)
    assert b.focal_magnitude() == pytest.approx(b.p0 * b.k * b.h, rel=1e-14)  # lossless: |p(D)| = p0 k h


def test_axial_formula_matches_C1_away_from_the_focus() -> None:
    b = FocusedBowl(attenuation=3.0)
    z = np.array([20e-3, 50e-3, 89e-3, 91e-3, 150e-3])
    kh = b.k_hat
    rm = np.sqrt(b.a**2 + (z - b.h) ** 2)
    c1 = b.p0 * (np.exp(1j * kh * z) - np.exp(1j * kh * rm)) / (1 - z / b.D)
    assert np.allclose(b.axial_pressure(z), c1, rtol=1e-12)
    # continuous through the focus, and conjugated for exp(+i omega t)
    near = b.axial_pressure(b.D + np.array([-1e-9, 0.0, 1e-9]))  # the phase turns by k 1e-9 = 4e-6
    assert np.allclose(near, near[1], rtol=1e-5)
    assert np.allclose(b.axial_pressure(z, time_sign=1), c1.conj(), rtol=1e-12)


def test_flat_piston_limit() -> None:
    """A very weakly curved bowl tends to the piston's axis, p0 (exp(ikz) - exp(ik sqrt(a^2 + z^2)))."""
    b = FocusedBowl(aperture=20e-3, radius_of_curvature=1e3)
    z = np.linspace(5e-3, 80e-3, 50)
    piston = b.p0 * (np.exp(1j * b.k * z) - np.exp(1j * b.k * np.sqrt(b.a**2 + z**2)))
    assert np.allclose(b.axial_pressure(z), piston, atol=1e-3 * b.p0)


def test_beam_maximum_before_the_focus(small: FocusedBowl) -> None:
    z_bm, p_bm = small.beam_maximum()
    assert z_bm < small.D and p_bm >= small.focal_magnitude()
    z = np.linspace(z_bm - 1e-3, z_bm + 1e-3, 2001)
    assert np.max(np.abs(small.axial_pressure(z))) <= p_bm * (1 + 1e-12)


def test_rayleigh_quadrature(small: FocusedBowl) -> None:
    """The quadrature of the Rayleigh integral gives the analytic axis, and has converged off it."""
    z = np.linspace(small.h + 1e-3, 1.5 * small.D, 200)
    on_axis = small.rayleigh(0.0, z)[0]
    assert compare.relative_l2(on_axis, small.axial_pressure(z)) < 1e-12
    r = np.linspace(0, 6e-3, 13)
    assert compare.relative_l2(small.rayleigh(r, [small.D])[:, 0],
                               small.rayleigh(r, [small.D], points_per_wavelength=12)[:, 0]) < 1e-10  # fmt: skip
    lossy = FocusedBowl(aperture=20e-3, radius_of_curvature=30e-3, attenuation=30.0)
    assert compare.relative_l2(lossy.rayleigh(0.0, z)[0], lossy.axial_pressure(z)) < 1e-12


# -- the benchmark -----------------------------------------------------------------------------------


def test_rayleigh_model_checks_the_processing(small: FocusedBowl) -> None:
    """The reference field, revolved and recorded, gives the analytic axis back to round-off."""
    res = simulate(small, spacing=small.wavelength / 4, r_max=4e-3, method="rayleigh")
    cmp = compare_axis(res, small)
    assert cmp["axis"]["magnitude"]["L2_rel"] < 1e-12 and cmp["axis"]["complex"]["L2_rel"] < 1e-12
    assert cmp["geometric_focus"]["relative_error"] == pytest.approx(0.0, abs=1e-4)  # interpolated between planes
    assert abs(cmp["beam_maximum"]["position_error"]) < cmp["beam_maximum"]["plane_spacing"] / 10
    m = res.metrics()
    assert m["focus"]["z"] == pytest.approx(small.beam_maximum()[0], abs=res.grid.dz)


def test_angular_spectrum_model_converges(small: FocusedBowl) -> None:
    lam = small.wavelength
    report = run_bowl_benchmark(small, spacings=(lam / 2, lam / 4), r_max=4e-3)
    conv = report["convergence"]
    assert conv["axis_magnitude_L2_rel"][1] < conv["axis_magnitude_L2_rel"][0] < 1e-3
    assert conv["longitudinal_magnitude_L2_rel"][1] < conv["longitudinal_magnitude_L2_rel"][0] < 2e-3
    fine = report["runs"][-1]
    assert abs(fine["beam_maximum"]["relative_error"]) < 1e-3
    assert abs(fine["beam_maximum"]["position_error"]) < lam / 10
    assert report["model"]["reference_axis_vs_analytic"]["L2_rel"] < 1e-12
    assert report["metrics_finest"]["focal_shift"] < 0  # the beam maximum is before the geometric focus


def test_absorbing_medium(small: FocusedBowl) -> None:
    lossy = FocusedBowl(aperture=20e-3, radius_of_curvature=30e-3, attenuation=20.0)
    res = simulate(lossy, spacing=lossy.wavelength / 4, r_max=4e-3)
    cmp = compare_axis(res, lossy)
    assert cmp["axis"]["magnitude"]["L2_rel"] < 1e-3
    assert cmp["geometric_focus"]["analytic"] < small.focal_magnitude() * math.exp(-20 * 0.03) * 1.0001


def test_report_files_and_cli(tmp_path: Path, small: FocusedBowl) -> None:
    lam = small.wavelength
    report = run_bowl_benchmark(small, spacings=(lam / 2,), r_max=3e-3, longitudinal=False)
    written = save_report(report, tmp_path / "api")
    names = {p.name for p in written}
    assert {"benchmark.json", "benchmark_axis.txt", "metrics.json"} <= names
    saved = json.loads((tmp_path / "api" / "benchmark.json").read_text())
    assert saved["inputs"]["aperture"] == 20e-3 and "longitudinal_plane" not in saved["runs"][0]
    table = np.loadtxt(tmp_path / "api" / "benchmark_axis.txt")
    assert np.allclose(table[:, 1], np.abs(small.axial_pressure(table[:, 0])))
    out = tmp_path / "cli"
    args = ["--benchmark", "bowl", "-o", str(out), "-q", "--set", "aperture=0.02", "--set",
            "radius_of_curvature=0.03", "--set", f"spacings=[{lam / 2}]", "--set", "r_max=0.003",
            "--set", "longitudinal=false"]  # fmt: skip
    assert main(args) == 0
    assert json.loads((out / "benchmark.json").read_text())["inputs"]["radius_of_curvature"] == 0.03
    with pytest.raises(SystemExit):
        main(["--benchmark", "bowl", "--set", "colour=blue"])


# -- the angular spectrum projector and the norms ------------------------------------------------------


def test_angular_spectrum_plane_wave_and_beam() -> None:
    k = 2 * np.pi / 1.5e-3
    x = 0.1e-3 * np.arange(-64, 64)
    ky = 2 * np.pi * 5 / (x.size * 0.1e-3)  # a periodic oblique plane wave: exact
    plane = np.exp(1j * ky * x)[None, :] * np.ones((x.size, 1))
    out = angular_spectrum(plane, 0.1e-3, 0.1e-3, [0.0, 2e-3], k, pad=1.0)
    assert np.allclose(out[1], plane * np.exp(1j * np.sqrt(k**2 - ky**2) * 2e-3), atol=1e-12)
    # exp(+i omega t): the conjugate; and back again (propagating content only)
    g = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2e-3) ** 2).astype(complex)  # no evanescent part
    fwd = angular_spectrum(g, 0.1e-3, 0.1e-3, [3e-3], k)[0]
    assert np.allclose(angular_spectrum(g.conj(), 0.1e-3, 0.1e-3, [3e-3], k, time_sign=1)[0], fwd.conj())
    back = angular_spectrum(fwd, 0.1e-3, 0.1e-3, [-3e-3], k)[0]
    assert compare.relative_l2(back, g) < 1e-4  # what spreads beyond the 12.8 mm window is cropped


def test_error_norms() -> None:
    ref = np.array([1.0, 2.0, -2.0])
    x = np.array([1.0, 2.5, -2.0])
    n = compare.error_norms(x, ref)
    assert n["L2"] == pytest.approx(0.5) and n["L2_rel"] == pytest.approx(0.5 / 3)
    assert n["Linf"] == pytest.approx(0.5) and n["Linf_rel"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="do not match"):
        compare.l2(x, ref[:2])
    s, v, r = compare.on_common_points([0, 1, 2], [0.0, 1.0, 2.0], [-1, 0.5, 1.5, 3], [9, 9, 9, 9])
    assert np.allclose(s, [0.5, 1.5]) and np.allclose(v, [0.5, 1.5]) and np.allclose(r, [9, 9])
