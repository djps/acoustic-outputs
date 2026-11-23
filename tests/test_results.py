"""The recorded metrics of a focused Gaussian beam against its analytic values."""

from __future__ import annotations

import math

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs.synthetic import GaussianBeam


@pytest.fixture(scope="module")
def res(beam: GaussianBeam) -> ao.Results:
    return ao.from_harmonics(beam.harmonics(), x=beam.x, y=beam.y, z=beam.z, **beam.medium())


def test_focus(beam: GaussianBeam, res: ao.Results) -> None:
    m = res.metrics()
    f = m["focus"]
    assert f["z"] == pytest.approx(beam.focus)
    assert f["x"] == pytest.approx(0.0, abs=1e-15) and f["y"] == pytest.approx(0.0, abs=1e-15)
    assert not f["on_edge"]
    # every harmonic is in phase at the focus: p+ is the sum of the amplitudes
    assert f["p_positive"] == pytest.approx(sum(beam.amplitudes), rel=1e-12)
    np.testing.assert_allclose(f["harmonics"], beam.amplitudes, rtol=1e-12)
    w = ao.waveform(np.asarray(beam.amplitudes, dtype=complex).reshape(-1, 1), 4096)
    assert f["p_negative"] == pytest.approx(w.min(), rel=1e-6)
    assert f["intensity"] == pytest.approx(sum(a**2 for a in beam.amplitudes) / (2 * beam.impedance))
    assert m["spatial_peak_p_positive"]["off_axis"] is False


@pytest.mark.parametrize("level", [-3.0, -6.0, -12.0])
def test_widths_lengths_and_areas(beam: GaussianBeam, res: ao.Results, level: float) -> None:
    m = res.metrics(level_db=level)
    tag = f"{abs(level):g}dB"
    assert m[f"focal_width_x_{tag}"] == pytest.approx(beam.focal_width(level), rel=3e-3)
    assert m[f"focal_width_y_{tag}"] == pytest.approx(beam.focal_width(level), rel=3e-3)
    if level > -12:  # at -12 dB the focal zone is longer than the planes
        assert m[f"focal_length_{tag}"] == pytest.approx(beam.focal_length(level), rel=5e-3)
    else:
        assert math.isnan(m[f"focal_length_{tag}"])
    r = beam.focal_width(level) / 2
    assert m[f"focal_area_{tag}"] == pytest.approx(math.pi * r**2, rel=0.05)  # counted in 0.25 mm cells


def test_power_is_the_same_on_every_plane(beam: GaussianBeam, res: ao.Results) -> None:
    np.testing.assert_allclose(res.power, beam.power, rtol=1e-5)  # the end planes lose 1e-6 past the edges
    exact = res.metrics()["iec"]["power"]["exact"]["focal_plane"]
    assert 0.98 * beam.power < exact <= beam.power  # the obliquity of the angular spectrum


def test_axis_profiles(beam: GaussianBeam, res: ao.Results) -> None:
    g_axis = beam.waist / beam.width(beam.z)
    np.testing.assert_allclose(res.axis_p_plus, sum(beam.amplitudes) * g_axis, rtol=1e-9)
    np.testing.assert_allclose(np.abs(res.axis[:, 1]), beam.amplitudes[1] * g_axis, rtol=1e-12)
    np.testing.assert_allclose(res.centroid_x, 0.0, atol=1e-12)


def test_beam_axis_elsewhere(beam: GaussianBeam) -> None:
    """A beam off the plane centre, with the axis given."""
    shifted = np.roll(beam.harmonics(), 8, axis=2)  # +2 mm in x
    res = ao.from_harmonics(shifted, x=beam.x, y=beam.y, z=beam.z, axis_xy=(2e-3, 0.0), **beam.medium())
    assert res.axis_xy == (pytest.approx(2e-3), pytest.approx(0.0, abs=1e-15))
    m = res.metrics(level_db=-6)
    assert m["focus"]["x"] == pytest.approx(2e-3)
    assert m["focal_width_x_6dB"] == pytest.approx(beam.focal_width(-6), rel=3e-3)
    assert m["focus"]["p_positive"] == pytest.approx(sum(beam.amplitudes))


def test_two_dimensional_beam(flat: GaussianBeam) -> None:
    res = ao.from_harmonics(flat.harmonics()[..., 0], layout="zkx", x=flat.x, z=flat.z, **flat.medium())
    assert res.grid.ny == 1 and res.grid.dy == 1.0
    m = res.metrics(level_db=-6)
    assert m["focus"]["z"] == pytest.approx(flat.focus)
    assert m["focal_width_x_6dB"] == pytest.approx(flat.focal_width(-6), rel=3e-3)
    assert m["focal_length_6dB"] == pytest.approx(flat.focal_length(-6), rel=5e-3)
    assert math.isnan(m["focal_width_y_6dB"])
    np.testing.assert_allclose(res.power, flat.power, rtol=1e-4)  # W per metre in y


def test_points_and_files(beam: GaussianBeam, tmp_path) -> None:
    settings = ao.RecordSettings(points=[("focus.txt", (0.0, 0.0, beam.focus)), ("off.txt", (1e-3, 0.0, 0.02))])
    res = ao.from_harmonics(beam.harmonics(), x=beam.x, y=beam.y, z=beam.z, settings=settings, **beam.medium())
    assert set(res.points) == {"focus.txt", "off.txt"}
    names = {f.name for f in res.save(tmp_path)}
    assert {"axial.txt", "planes.txt", "focal_plane.txt", "metrics.json", "results.npz", "focus.txt"} <= names
    ax = np.loadtxt(tmp_path / "axial.txt")
    assert ax.shape == (beam.z.size, 8 + 4)
    np.testing.assert_allclose(ax[:, 1], res.axis_p_plus, rtol=1e-8)
    _t, p, _ = np.loadtxt(tmp_path / "focus.txt").T
    assert p.max() == pytest.approx(sum(beam.amplitudes), rel=1e-3)


def test_recorder_checks(beam: GaussianBeam) -> None:
    g = ao.Grid(nx=4, ny=4, nz=2, nt=2, dx=1.0, dy=1.0, dz=1.0)
    rec = ao.PlaneRecorder(g, 1, frequency=1e6, sound_speed=1500.0, impedance=1.5e6)
    with pytest.raises(ValueError, match="shape"):
        rec.add(0.0, np.zeros((2, 4, 4), dtype=complex))
    rec.add(0.0, np.ones((1, 4, 4), dtype=complex))
    with pytest.raises(ValueError, match="increasing"):
        rec.add(0.0, np.ones((1, 4, 4), dtype=complex))
    with pytest.raises(ValueError, match="outside"):
        ao.PlaneRecorder(g, 1, frequency=1e6, sound_speed=1500.0, impedance=1.5e6, axis_index=(4, 0))
    with pytest.raises(ValueError, match="no plane"):
        ao.PlaneRecorder(g, 1, frequency=1e6, sound_speed=1500.0, impedance=1.5e6).finish()
