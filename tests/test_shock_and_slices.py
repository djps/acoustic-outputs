"""The spectral shock criterion (Fubini solution, sawtooth) and the harmonic slices and x-y window
that the recorder keeps."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import jv

import acoustic_outputs as ao
from acoustic_outputs import iec
from acoustic_outputs.synthetic import GaussianBeam


def fubini(sigma: float, k: int) -> np.ndarray:
    """Harmonics of the Fubini solution at distance sigma (in shock-formation distances), unit amplitude."""
    n = np.arange(1, k + 1)
    return 2 * jv(n, n * sigma) / (n * sigma) + 0j


def sawtooth(k: int, jump: float = 2.0) -> np.ndarray:
    n = np.arange(1, k + 1)
    return jump * 1j / (math.pi * n) * (-1.0) ** (n + 1)


@pytest.mark.parametrize("k", [32, 64, 128])
def test_fubini_exponent_is_four_thirds_where_the_shock_forms(k: int) -> None:
    s = iec.spectral_shock(fubini(1.0, k))
    assert float(s["exponent"]) == pytest.approx(4 / 3, abs=0.03)
    before = iec.spectral_shock(fubini(0.8, k))
    assert float(before["exponent"]) > 2 and not bool(before["shocked"])


def test_sawtooth_jump_from_the_spectrum() -> None:
    s = iec.spectral_shock(sawtooth(64, jump=3.0))
    assert float(s["exponent"]) == pytest.approx(1.0, abs=1e-12)
    assert bool(s["shocked"])
    assert float(s["shock_amplitude"]) == pytest.approx(3.0, rel=1e-12)


def test_no_shock_for_a_sine_or_a_noise_floor() -> None:
    h = np.zeros(64, dtype=complex)
    h[0] = 1.0
    h[1:] = 1e-12  # a flat numerical floor: s = 0 but far below the fundamental
    s = iec.spectral_shock(h)
    assert not bool(s["shocked"]) and math.isnan(float(s["shock_amplitude"]))
    few = iec.spectral_shock(sawtooth(4))
    assert math.isnan(float(few["exponent"])) and not bool(few["shocked"])


def test_shock_formation_along_a_line() -> None:
    """Fubini harmonics at sigma = z / z_s: the shock forms at z_s."""
    z_s = 0.05
    z = np.linspace(0.01, 0.06, 101)
    h = np.stack([fubini(zz / z_s, 64) for zz in z])
    sf = iec.shock_formation(z, h, axis=1)
    assert sf["z"] == pytest.approx(z_s, abs=z[1] - z[0])
    assert sf["exponent"].shape == z.shape


@pytest.fixture(scope="module")
def beam() -> GaussianBeam:
    return GaussianBeam().coarse()


def record(beam: GaussianBeam, **settings: object) -> ao.Results:
    return ao.from_harmonics(beam.harmonics(), x=beam.x, y=beam.y, z=beam.z, settings=ao.RecordSettings(**settings),
                             **beam.medium())  # fmt: skip


def test_slices_over_an_extent(beam: GaussianBeam) -> None:
    extent = ((-2e-3, 20e-3), (3e-3, 35e-3))
    res = record(beam, slices="both", slice_extent=extent)
    h = beam.harmonics()  # (nz, K, nx, ny)
    keep_z = (beam.z >= 20e-3 - 1e-12) & (beam.z <= 35e-3 + 1e-12)
    keep_x = (beam.x >= -2e-3 - 1e-12) & (beam.x <= 3e-3 + 1e-12)
    ix, iy = res.axis_ij
    np.testing.assert_allclose(res.slice_z, beam.z[keep_z])
    np.testing.assert_allclose(res.xz_coordinate, beam.x[keep_x], atol=1e-15)
    np.testing.assert_allclose(res.yz_coordinate, beam.y[keep_x], atol=1e-15)  # the same extent in y
    assert res.xz_harmonics.shape == (len(beam.amplitudes), keep_z.sum(), keep_x.sum())
    np.testing.assert_allclose(res.xz_harmonics, h[keep_z][:, :, keep_x, iy].transpose(1, 0, 2))
    np.testing.assert_allclose(res.yz_harmonics, h[keep_z][:, :, ix, keep_x].transpose(1, 0, 2))
    # the axis is in the slice
    j = int(np.argmin(np.abs(res.xz_coordinate - res.grid.x[ix])))
    np.testing.assert_allclose(res.xz_harmonics[:, :, j].T, res.axis[keep_z])


def test_xy_window_of_the_focal_plane_and_at_a_depth(beam: GaussianBeam) -> None:
    window = ((-1e-3, -2e-3), (2e-3, 1e-3))  # bottom left, top right
    res = record(beam, xy_plane=True, xy_extent=window)
    cx = (beam.x >= -1e-3 - 1e-12) & (beam.x <= 2e-3 + 1e-12)
    cy = (beam.y >= -2e-3 - 1e-12) & (beam.y <= 1e-3 + 1e-12)
    assert res.xy_z == pytest.approx(res.z[res.focal_index])
    np.testing.assert_allclose(res.xy_x, beam.x[cx], atol=1e-15)
    np.testing.assert_allclose(res.xy_y, beam.y[cy], atol=1e-15)
    np.testing.assert_allclose(res.xy_harmonics, res.focal_plane[:, cx][:, :, cy])
    at = record(beam, xy_plane=True, xy_extent=window, xy_z=26.4e-3)
    iz = int(np.argmin(np.abs(beam.z - 26.4e-3)))
    assert at.xy_z == pytest.approx(beam.z[iz])
    np.testing.assert_allclose(at.xy_harmonics, beam.harmonics()[iz][:, cx][:, :, cy])


def test_slices_saved_and_loaded(beam: GaussianBeam, tmp_path: Path) -> None:
    res = record(beam, slices="xz", slice_extent=((-1e-3, 0.0), (1e-3, 1.0)), xy_plane=True,
                 plane_precision="single")  # fmt: skip
    assert res.yz_harmonics is None and res.xz_harmonics.dtype == np.complex64
    res.save(tmp_path)
    back = ao.Results.load(tmp_path)
    np.testing.assert_array_equal(back.xz_harmonics, res.xz_harmonics)
    np.testing.assert_array_equal(back.xy_harmonics, res.xy_harmonics)
    assert back.xy_z == res.xy_z


def test_bad_extents(beam: GaussianBeam) -> None:
    with pytest.raises(ValueError, match="no grid point"):
        record(beam, slices="xz", slice_extent=((1.0, 0.0), (2.0, 1.0)))
    with pytest.raises(ValueError, match="backwards"):
        record(beam, xy_plane=True, xy_extent=((1e-3, 0.0), (-1e-3, 1e-3)))
    with pytest.raises(Exception, match="xy"):  # beartype rejects it first when it is on
        record(beam, slices="xy")


def test_metrics_report_the_shock_formation(beam: GaussianBeam) -> None:
    nl = record(beam).metrics()["iec"]["nonlinear"]
    assert nl["shocked_spectral"] is False  # four harmonics: no band to test
    assert math.isnan(nl["shock_formation"]["z"])
