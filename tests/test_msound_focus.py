"""mSOUND (frequency-specific and time-domain) and FOCUS (CW, KZK, transient) output."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

import acoustic_outputs as ao
from acoustic_outputs import synthetic
from acoustic_outputs.datasets import from_dataset
from acoustic_outputs.preprocess._common import phase_convention
from acoustic_outputs.preprocess.focus import grid_vectors
from acoustic_outputs.preprocess.msound import centred_coordinates
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal


def reference(beam: GaussianBeam, k: int | None = None) -> dict:
    """Metrics of the first ``k`` harmonics of the beam, given as retarded harmonics."""
    return ao.from_harmonics(beam.harmonics()[:, :k], x=beam.x, y=beam.y, z=beam.z, **beam.medium()).metrics()


@pytest.fixture(scope="module")
def fine() -> GaussianBeam:
    """Planes 0.25 mm (a sixth of a wavelength) apart, close enough to read the phase convention."""
    return GaussianBeam(z=np.linspace(15e-3, 45e-3, 121))


# -- mSOUND --------------------------------------------------------------------------------------


def test_msound_centred_grid() -> None:
    assert np.allclose(centred_coordinates(5, 1.0), [-2, -1, 0, 1, 2])
    assert np.allclose(centred_coordinates(4, 1.0), [-2, -1, 0, 1])  # set_grid's makeDim


def test_msound_fundamental_and_second(beam: GaussianBeam) -> None:
    data = synthetic.msound_dataset(beam)
    ref = reference(beam, 2)
    assert_metrics_equal(ref, from_dataset(data).metrics(), 1e-9)  # recognised without a kind
    # the transverse grid from the spacings alone, as set_grid makes it
    bare = {k: v for k, v in data.items() if k not in ("x", "y")}
    assert_metrics_equal(ref, from_dataset(bare, "msound").metrics(), 1e-9)
    # the fundamental alone
    one = ao.from_msound(data["P_fundamental"], dz=data["dz"], dx=data["dx"], dy=data["dy"], z0=data["z0"],
                         frequency=1e6, sound_speed=1500.0, density=1000.0)  # fmt: skip
    assert one.grid.n_harmonics == 1
    assert one.metrics()["focus"]["p_positive"] == pytest.approx(1e6, rel=1e-9)


def test_msound_2d(flat: GaussianBeam) -> None:
    """Forward2D_fund: (num_x, num_y + 1), mgrid.y the beam axis and dy the plane spacing."""
    data = synthetic.msound_dataset(flat)
    assert data["P_fundamental"].shape == (flat.x.size, flat.z.size)
    res = from_dataset(data)
    assert res.grid.ny == 1
    assert_metrics_equal(reference(flat, 2), res.metrics(), 1e-9)


def test_msound_time_domain(coarse: GaussianBeam, coarse_ref: ao.Results) -> None:
    data = synthetic.msound_time_series_dataset(coarse)
    assert data["p_time"].shape[1] == coarse.x.size * coarse.y.size * coarse.z.size
    assert_metrics_equal(coarse_ref.metrics(), from_dataset(data).metrics(), 1e-8)
    with pytest.raises(ValueError, match="sensors"):
        ao.from_msound_time_series(data["p_time"][:, :-1], data["sensor_mask"], dt=data["dt"], dz=data["dz"],
                                   dx=data["dx"], dy=data["dy"], frequency=1e6, sound_speed=1500.0,
                                   density=1000.0)  # fmt: skip


def test_msound_mat_file(beam: GaussianBeam, tmp_path: Path) -> None:
    """As saved from MATLAB: struct(mgrid), the medium struct (heterogeneous rho) and omega_c."""
    d = synthetic.msound_dataset(beam)
    mgrid = {k: d[k] for k in ("dx", "dy", "dz", "x", "y", "num_x", "dim")}
    mgrid.update(z=centred_coordinates(beam.z.size - 1, d["dz"]), dt=0.0, t=0.0)
    medium = {"c0": 1500.0, "rho0": 1000.0, "c": 1500.0, "rho": np.full(d["P_fundamental"].shape, 1000.0),
              "beta": 3.6, "ca": 0.0, "cb": 2.0}  # fmt: skip
    path = tmp_path / "msound.mat"
    savemat(path, {"P_fundamental": d["P_fundamental"], "P_second": d["P_second"], "mgrid_s": mgrid,
                   "medium": medium, "omega_c": 2 * math.pi * 1e6, "z0": d["z0"]})  # fmt: skip
    assert_metrics_equal(reference(beam, 2), from_dataset(path).metrics(), 1e-9)


# -- FOCUS ---------------------------------------------------------------------------------------


def test_phase_convention(fine: GaussianBeam, beam: GaussianBeam) -> None:
    h = fine.harmonics()[:, 0]
    absolute = h * np.exp(-1j * fine.k * (fine.z - fine.z[0]))[:, None, None]
    args = (fine.frequency, fine.sound_speed)
    assert phase_convention(absolute, fine.z, *args) == ("absolute", 1)
    assert phase_convention(absolute.conj(), fine.z, *args) == ("absolute", -1)
    assert phase_convention(h, fine.z, *args) == ("retarded", 0)
    assert phase_convention(beam.harmonics()[:, 0], beam.z, *args) is None  # 1 mm: 2/3 of a wavelength


@pytest.mark.parametrize("time_sign", [1, -1])
def test_focus_cw(fine: GaussianBeam, time_sign: int, caplog: pytest.LogCaptureFixture) -> None:
    ref = reference(fine, 1)
    with caplog.at_level(logging.INFO):
        res = from_dataset(synthetic.focus_dataset(fine, time_sign=time_sign), "focus")
    assert f"exp({'+' if time_sign > 0 else '-'}i omega t)" in caplog.text
    assert_metrics_equal(ref, res.metrics(), 1e-9)
    # one frequency: the convention changes nothing
    wrong = from_dataset(synthetic.focus_dataset(fine, time_sign=time_sign), "focus", phase="retarded",
                         time_sign=-time_sign)  # fmt: skip
    assert_metrics_equal(ref, wrong.metrics(), 1e-9)


def test_focus_kzk(beam: GaussianBeam, fine: GaussianBeam, caplog: pytest.LogCaptureFixture) -> None:
    # read as retarded from the field
    assert_metrics_equal(reference(fine), from_dataset(synthetic.focus_dataset(fine, "kzk"), "focus").metrics(), 1e-9)
    # planes too far apart: taken as retarded, with a warning
    with caplog.at_level(logging.WARNING):
        res = from_dataset(synthetic.focus_dataset(beam, "kzk"), "focus")
    assert "too far apart" in caplog.text
    assert_metrics_equal(reference(beam), res.metrics(), 1e-9)


def test_focus_transient(coarse: GaussianBeam, coarse_ref: ao.Results) -> None:
    data = synthetic.focus_dataset(coarse, "transient")
    assert data["pressure"].shape[:3] == (coarse.x.size, coarse.y.size, coarse.z.size)
    assert_metrics_equal(coarse_ref.metrics(), from_dataset(data, "focus").metrics(), 1e-8)


def test_focus_grid_limits_and_2d(fine: GaussianBeam, flat: GaussianBeam) -> None:
    x, y, z = grid_vectors([0.5e-3, 0.5e-3, 1e-3], -1e-3, 1e-3, 0.0, 0.0, 10e-3, 20e-3)
    assert np.allclose(x, [-1e-3, -0.5e-3, 0, 0.5e-3, 1e-3]) and y.size == 1 and z.size == 11
    # set_coordinate_grid(delta, xmin, ..., zmax) instead of the vectors
    d = synthetic.focus_dataset(fine)
    dx, dz = fine.x[1] - fine.x[0], fine.z[1] - fine.z[0]
    lim = {"delta": np.array([dx, dx, dz]), "xmin": fine.x[0], "xmax": fine.x[-1], "ymin": fine.y[0],  # type: ignore[index]
           "ymax": fine.y[-1], "zmin": fine.z[0], "zmax": fine.z[-1]}  # type: ignore[index]  # fmt: skip
    res = from_dataset({k: v for k, v in d.items() if k not in "xyz"} | lim, "focus")
    assert_metrics_equal(reference(fine, 1), res.metrics(), 1e-9)
    # a 2D (xz) grid, also as MATLAB saves it, without the singleton y
    flat_fine = GaussianBeam(waist=flat.waist, x=flat.x, y=None, z=np.linspace(5e-3, 55e-3, 201))
    d2 = synthetic.focus_dataset(flat_fine)
    assert d2["pressure"].shape == (flat_fine.x.size, 1, flat_fine.z.size)
    ref2 = reference(flat_fine, 1)
    assert_metrics_equal(ref2, from_dataset(d2, "focus").metrics(), 1e-9)
    assert_metrics_equal(ref2, from_dataset(d2 | {"pressure": d2["pressure"][:, 0]}, "focus").metrics(), 1e-9)
    # a yz grid: y is the transverse axis
    yz = d2 | {"pressure": np.swapaxes(d2["pressure"], 0, 1), "x": np.zeros(1), "y": flat_fine.x}
    assert_metrics_equal(ref2, from_dataset(yz, "focus").metrics(), 1e-9)


def test_focus_errors(coarse: GaussianBeam) -> None:
    d = synthetic.focus_dataset(coarse)
    with pytest.raises(ValueError, match="does not fit"):
        from_dataset(d | {"z": coarse.z[:-1]}, "focus")
    t = synthetic.focus_dataset(coarse, "transient")
    with pytest.raises(ValueError, match="time step"):
        ao.from_focus(t["pressure"], x=coarse.x, y=coarse.y, z=coarse.z, frequency=1e6, sound_speed=1500.0,
                      density=1000.0)  # fmt: skip
    with pytest.raises(ValueError, match="must be a number"):
        from_dataset(d | {"density": np.full(3, 1000.0)}, "focus")
