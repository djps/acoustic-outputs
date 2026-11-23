"""The preprocessors of each kind of solver output give back the field they were given.

The coarse Gaussian beam is given as harmonics in another axis order and phase convention, and
as time series of the kind a time-domain pseudo-spectral solver (k-wave-python) and an FDTD
solver (fullwave25) record: after a transient, at a time step that does not divide the period,
decimated, with drift. Each must give the metrics of the beam's harmonics.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs import synthetic
from acoustic_outputs.preprocess import HarmonicFit, fdtd, spectral, time_domain
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal

F0 = 1e6


def test_layouts_and_conventions(coarse: GaussianBeam, coarse_ref: ao.Results) -> None:
    """Axes (x, y, z, k), absolute phase, exp(-i omega t): the same field."""
    d = synthetic.harmonics_dataset(coarse, absolute=True)
    out = spectral.from_harmonics(
        d["harmonics"].conj(), layout="xyzk", x=coarse.x, y=coarse.y, z=coarse.z, phase="absolute", time_sign=-1,
        **coarse.medium(),
    )  # fmt: skip
    assert_metrics_equal(coarse_ref.metrics(), out.metrics(), 1e-9)


def test_single_harmonic(coarse: GaussianBeam) -> None:
    """A CW field without a harmonic axis."""
    h = coarse.harmonics()[:, 0]
    out = spectral.from_harmonics(h, layout="zxy", x=coarse.x, y=coarse.y, z=coarse.z, **coarse.medium())
    assert out.axis.shape == (coarse.z.size, 1)
    np.testing.assert_allclose(out.axis[:, 0], h[:, 16, 16])


def test_time_series(coarse: GaussianBeam, coarse_ref: ao.Results) -> None:
    d = synthetic.time_series_dataset(coarse)
    out = time_domain.from_time_series(
        d["p"], d["dt"], coords={"x": coarse.x, "y": coarse.y, "z": coarse.z}, n_harmonics=4, **coarse.medium()
    )
    assert_metrics_equal(coarse_ref.metrics(), out.metrics(), 1e-8)


def test_kwave_sensor_data(coarse: GaussianBeam, coarse_ref: ao.Results) -> None:
    """(n_sensor, Nt) in C order of a block mask inside a larger grid, propagation along x."""
    d = synthetic.kwave_dataset(coarse)
    kgrid = SimpleNamespace(dim=3, dt=d["dt"], **{f"{c}_vec": d[f"{c}_vec"][:, None] for c in "xyz"})
    out = time_domain.from_kwave(
        {"p": d["p"]}, kgrid, sensor_mask=d["sensor_mask"], record_start_index=d["record_start_index"],
        n_harmonics=4, **coarse.medium(),
    )  # fmt: skip
    assert_metrics_equal(coarse_ref.metrics(), out.metrics(), 1e-8)


def test_kwave_matlab_order(coarse: GaussianBeam, coarse_ref: ao.Results) -> None:
    """MATLAB k-Wave orders the sensors column-major."""
    d = synthetic.kwave_dataset(coarse, pad=(0, 0, 0))
    sub = d["sensor_mask"].shape
    p = d["p"].reshape(*sub, -1).reshape(-1, d["p"].shape[1], order="F")
    kgrid = SimpleNamespace(dim=3, dt=d["dt"], **{f"{c}_vec": d[f"{c}_vec"] for c in "xyz"})
    out = time_domain.from_kwave(
        p, kgrid, record_start_index=d["record_start_index"], order="F", n_harmonics=4, **coarse.medium()
    )
    assert_metrics_equal(coarse_ref.metrics(), out.metrics(), 1e-8)


def test_fullwave_sensor_data(coarse: GaussianBeam, coarse_ref: ao.Results) -> None:
    """Sensors by grid index in any order, decimated in time, with absorbing-layer drift."""
    d = synthetic.fullwave_dataset(coarse)
    sensor = SimpleNamespace(outcoords=d["outcoords"], sampling_modulus_time=d["sampling_modulus_time"],
                             is_sparse_grid=False)  # fmt: skip
    grid = SimpleNamespace(dx=d["dx"], dy=d["dy"], dz=d["dz"], dt=d["dt"], f0=d["f0"], c0=d["c0"])
    origin = tuple(d["origin"])
    out = fdtd.from_fullwave(d["sensor_data"], sensor, grid, density=d["density"], origin=origin, n_harmonics=4)
    assert_metrics_equal(coarse_ref.metrics(), out.metrics(), 1e-8)
    # without removing the drift, the fit leaves part of it in the harmonics
    bad = fdtd.from_fullwave(
        d["sensor_data"], sensor, grid, density=d["density"], origin=origin, n_harmonics=4, detrend=False
    )
    assert abs(bad.metrics()["focus"]["p_positive"] / coarse_ref.metrics()["focus"]["p_positive"] - 1) > 1e-3


def test_fdtd_2d(flat: GaussianBeam) -> None:
    """A 2D FDTD record (x axial, y lateral) against the same field given as harmonics."""
    beam = GaussianBeam(waist=flat.waist, x=flat.x[::2], y=None, z=flat.z[::5])
    d = synthetic.fdtd_dataset(beam)
    ref = spectral.from_harmonics(beam.harmonics()[..., 0], layout="zkx", x=beam.x, z=beam.z, **beam.medium())
    out = fdtd.from_time_series(d["p"], d["dt"], coords={"x": d["x"], "y": d["y"]}, n_harmonics=4, **beam.medium())
    assert out.grid.ny == 1
    assert_metrics_equal(ref.metrics(), out.metrics(), 1e-8)


def test_harmonic_fit() -> None:
    dt = 1 / (F0 * 10.3)
    with pytest.raises(ValueError, match="Nyquist"):
        HarmonicFit(500, dt, F0, 6)
    with pytest.raises(ValueError, match="too short"):
        HarmonicFit(10, dt, F0, 2, periods=2.0)
    fit = HarmonicFit(500, dt, F0, 3, periods=3.0, start_time=1e-7)
    t = 1e-7 + dt * np.arange(500)
    p = np.real((2 - 1j) * np.exp(2j * np.pi * F0 * t)) + np.real(0.5j * np.exp(6j * np.pi * F0 * t))
    h, rel = fit(p)
    np.testing.assert_allclose(h, [2 - 1j, 0, 0.5j], atol=1e-12)
    assert rel < 1e-12
    # window_start instead of periods
    fit = HarmonicFit(500, dt, F0, 3, start_time=1e-7, window_start=20e-6)
    assert fit.first == int(np.searchsorted(t, 20e-6 - 1e-9 * dt))
    np.testing.assert_allclose(fit(p)[0], [2 - 1j, 0, 0.5j], atol=1e-12)


def test_not_steady_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    dt = 1 / (F0 * 12.1)
    t = dt * np.arange(400)
    growing = (t / t[-1])[None, :] * np.sin(2 * np.pi * F0 * t)[None, :] * np.ones((3, 4, 1, 1))  # x, y, z, t
    with caplog.at_level(logging.WARNING, logger="acoustic_outputs.preprocess"):
        time_domain.from_time_series(
            growing, dt, coords={"x": [0.0, 1e-3, 2e-3], "y": [0, 1e-3, 2e-3, 3e-3], "z": [0.0]},
            frequency=F0, sound_speed=1500.0, density=1000.0, n_harmonics=2, periods=4.0,
        )  # fmt: skip
    assert "not a steady periodic field" in caplog.text


def test_layout_and_sensor_errors() -> None:
    with pytest.raises(ValueError, match="layout"):
        spectral.standard_layout(np.zeros((2, 3, 4)), "zkxy")
    with pytest.raises(ValueError, match="layout"):
        spectral.standard_layout(np.zeros((2, 3)), "kx")
    with pytest.raises(ValueError, match="uniformly"):
        spectral.from_harmonics(
            np.zeros((2, 3)), layout="zx", x=[0, 1, 3], z=[0, 1], frequency=F0, sound_speed=1.0, density=1.0
        )
    with pytest.raises(ValueError, match="increase"):
        spectral.from_harmonics(
            np.zeros((2, 3)), layout="zx", x=[0, 1, 2], z=[1, 0], frequency=F0, sound_speed=1.0, density=1.0
        )
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1:3] = True
    assert time_domain.sensor_block(mask, (4, 4)) == (slice(1, 2), slice(1, 3))
    mask[2, 1] = True
    with pytest.raises(ValueError, match="rectangular"):
        time_domain.sensor_block(mask, (4, 4))
    with pytest.raises(ValueError, match="block"):
        fdtd.gather(np.zeros((3, 10)), [[0, 0], [0, 1], [1, 1]])
    with pytest.raises(ValueError, match="repeated"):
        fdtd.gather(np.zeros((4, 10)), [[0, 0], [0, 0], [1, 1], [0, 1]])
    sparse = SimpleNamespace(is_sparse_grid=True)
    with pytest.raises(NotImplementedError):
        fdtd.from_fullwave(np.zeros((1, 1)), sparse, SimpleNamespace(f0=F0, c0=1500.0), density=1000.0)
