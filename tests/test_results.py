"""The recorded metrics of a focused Gaussian beam against its analytic values."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal


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


def test_beamwidth_is_the_greatest_distance() -> None:
    """Across the beam the width is the greatest distance between level crossings (IEC 62127-1,
    61828): side lobes above the level are inside it. The main lobe's width is kept beside it."""
    from acoustic_outputs.results import area_above, beamwidth_at_level, connected_area, width_at_level

    x = np.linspace(-10, 10, 2001)
    v = np.exp(-(x**2)) + 0.2 * np.exp(-((np.abs(x) - 4) ** 2) / 0.2)  # a main lobe and side lobes at 0.2
    i = 1000
    assert beamwidth_at_level(x, v, 0.5) == pytest.approx(width_at_level(x, v, i, 0.5))  # above the lobes
    main = width_at_level(x, v, i, 0.1)
    full = beamwidth_at_level(x, v, 0.1)
    assert main == pytest.approx(2 * np.sqrt(np.log(10)), rel=1e-4)
    assert full == pytest.approx(2 * (4 + np.sqrt(0.2 * np.log(2))), rel=1e-3)  # to the side lobes' outer edges
    assert math.isnan(beamwidth_at_level(x, v, 0.0))  # at or above the level at the ends: not resolved
    plane = np.add.outer(v, v) / 2
    assert area_above(plane, 0.15, 1.0) > connected_area(plane, (i, i), 0.15, 1.0)


def test_focal_widths_include_side_lobes() -> None:
    """A focused bowl's first side lobes (-16.8 dB) are inside its -20 dB width, not its -12 dB one."""
    from acoustic_outputs.benchmarks import FocusedBowl, simulate

    bowl = FocusedBowl()  # C.2's transducer: deep nulls between the lobes in the focal plane
    res = simulate(bowl, spacing=bowl.wavelength / 6, r_max=5e-3, z_range=(85e-3, 95e-3), method="rayleigh")
    m = res.metrics()
    assert m["focal_width_x_12dB"] == pytest.approx(m["focal_width_x_12dB_main_lobe"], rel=1e-12)
    assert m["focal_width_x_20dB"] > 1.5 * m["focal_width_x_20dB_main_lobe"]
    assert m["focal_width_y_20dB"] == pytest.approx(m["focal_width_x_20dB"], rel=1e-9)
    assert m["focal_area_20dB"] > m["focal_area_20dB_main_lobe"]
    assert m["focal_area_6dB"] == pytest.approx(m["focal_area_6dB_main_lobe"])
    lobes = m["iec"]["side_lobes"]["x"]
    assert -20 < lobes["after_level_dB"] < -12


def test_side_lobe_peaks_and_level_extents() -> None:
    """All the side lobes, of which secondary_maxima are the largest; and level_extents behind the metrics."""
    from acoustic_outputs import iec
    from acoustic_outputs.benchmarks import FocusedBowl, simulate

    x = np.linspace(-10, 10, 2001)
    v = np.exp(-(x**2)) + 0.2 * np.exp(-((np.abs(x) - 4) ** 2) / 0.2) + 0.1 * np.exp(-((np.abs(x) - 7) ** 2) / 0.2)
    peaks = iec.side_lobe_peaks(x, v, 1000)
    assert [round(p["position"], 6) for p in peaks] == [-7.0, -4.0, 4.0, 7.0]
    assert peaks[1]["level_dB"] == pytest.approx(20 * np.log10(0.2 + np.exp(-16)), abs=1e-6)
    sec = iec.secondary_maxima(x, v, 1000)
    assert sec["before_position"] == peaks[1]["position"] and sec["after_position"] == peaks[2]["position"]

    bowl = FocusedBowl()
    res = simulate(bowl, spacing=bowl.wavelength / 6, r_max=5e-3, z_range=(60e-3, 115e-3), method="rayleigh")
    m = res.metrics()
    for lv in (-6.0, -20.0):
        ext = res.level_extents(lv)
        tag = f"{abs(lv):g}dB"
        assert m[f"focal_length_{tag}"] == pytest.approx(np.diff(ext["axis"]["main_lobe"])[0])
        assert m[f"focal_width_x_{tag}"] == pytest.approx(np.diff(ext["x"]["all_lobes"])[0])
        assert m[f"focal_width_x_{tag}_main_lobe"] == pytest.approx(np.diff(ext["x"]["main_lobe"])[0])
    # along the axis at -20 dB the prefocal and postfocal maxima (about -12 and -15 dB) lie outside the
    # focal length but inside the outermost crossings
    ext = res.level_extents(-20.0)["axis"]
    axial = m["iec"]["side_lobes"]["axial"]
    assert ext["all_lobes"][0] < axial["before_position"] < ext["main_lobe"][0]
    assert ext["main_lobe"][1] < axial["after_position"] < ext["all_lobes"][1]
    _, line_x, _ = res.focal_lines()
    lobes = iec.side_lobe_peaks(res.grid.x, line_x, res.axis_ij[0])
    first = min((p for p in lobes if p["position"] > 0), key=lambda p: p["position"])
    assert first["level_dB"] == pytest.approx(m["iec"]["side_lobes"]["x"]["after_level_dB"])
    # the lateral side lobes are of the focal quantity, as the axial ones are
    res.focal_quantity = "first_harmonic"
    _, line_x, _ = res.focal_lines()
    assert res.metrics()["iec"]["side_lobes"]["x"] == iec.secondary_maxima(res.grid.x, line_x, res.axis_ij[0])


def test_save_and_load_round_trip(tmp_path: Path, beam: GaussianBeam) -> None:
    """Results.load gives back what save wrote: the same arrays and the same metrics."""
    res = ao.from_harmonics(beam.harmonics(), x=beam.x, y=beam.y, z=beam.z, **beam.medium(), focusing_depth=0.03)
    res.save(tmp_path / "run")
    back = ao.Results.load(tmp_path / "run")
    assert back.grid == res.grid and back.focal_index == res.focal_index and back.axis_ij == res.axis_ij
    assert np.array_equal(back.axis, res.axis) and np.array_equal(back.xz_p_plus, res.xz_p_plus)
    assert_metrics_equal(res.metrics(), back.metrics(), 1e-12)
    with pytest.raises(FileNotFoundError, match="older version"):
        ao.Results.load(tmp_path)


def test_margin_and_single_precision_planes(beam: GaussianBeam, res: ao.Results, tmp_path: Path) -> None:
    """A margin (an absorbing layer) is left out of everything recorded, and the planes kept can
    be single precision; what is computed while recording is unchanged."""
    m = 4
    settings = ao.RecordSettings(margin=m, plane_precision="single")
    cut = ao.from_harmonics(beam.harmonics(), x=beam.x, y=beam.y, z=beam.z, settings=settings, **beam.medium())
    assert (cut.grid.nx, cut.grid.ny) == (res.grid.nx - 2 * m, res.grid.ny - 2 * m)
    np.testing.assert_allclose(cut.grid.x, res.grid.x[m:-m], atol=1e-15)
    np.testing.assert_allclose(cut.grid.y, res.grid.y[m:-m], atol=1e-15)
    assert cut.axis_xy == pytest.approx(res.axis_xy, abs=1e-15)
    inner = (slice(None), slice(m, -m), slice(m, -m))
    for name in ("focal_plane", "peak_plane", "source_plane", "face_plane", "final"):
        plane = getattr(cut, name)
        assert plane.dtype == np.complex64, name
        np.testing.assert_allclose(plane, getattr(res, name)[inner], rtol=1e-6, atol=1e-6 * np.abs(plane).max())
    assert cut.axis.dtype == np.complex128  # the axis is not a whole plane: it stays double
    np.testing.assert_array_equal(cut.xz_p_plus, res.xz_p_plus[:, m:-m])
    np.testing.assert_array_equal(cut.axis_p_plus, res.axis_p_plus)
    assert np.all(cut.power <= res.power * (1 + 1e-12))  # the power through the interior only
    full, part = res.metrics(level_db=-6), cut.metrics(level_db=-6)
    assert part["focus"]["p_positive"] == full["focus"]["p_positive"]
    assert part["focal_width_x_6dB"] == pytest.approx(full["focal_width_x_6dB"], rel=1e-12)
    cut.save(tmp_path)
    assert ao.Results.load(tmp_path).focal_plane.dtype == np.complex64
    with pytest.raises(ValueError, match="margin"):
        ao.PlaneRecorder(res.grid, 1, frequency=1e6, sound_speed=1500.0, impedance=1.5e6,
                         settings=ao.RecordSettings(margin=res.grid.nx // 2))  # fmt: skip
