"""A real k-wave-python time-domain run, reduced by the k-Wave preprocessor.

A 3.5 mm piston drives a 40 x 24 x 24 grid at 4 points per wavelength with a CW tone. The
harmonic fitted on k-Wave's first recorded plane is projected by an angular spectrum, zero-padded
to three times the width (k-Wave has a PML, so its field has no periodic images), and compared
with k-Wave's own beam axis: within 6% of the peak (pyWave3D's march measured 4.1%).

Skipped if k-wave-python (``pip install 'acoustic-outputs[kwave]'``) is not installed; about 15 s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs.datasets import from_dataset, save_dataset

pytestmark = pytest.mark.slow


def angular_spectrum(plane: np.ndarray, dx: float, dz: np.ndarray, k: float, pad: int = 3) -> np.ndarray:
    """|P| of a CW plane (exp(+i omega t)) projected by dz, on a domain ``pad`` times as wide."""
    nx, ny = plane.shape
    big = np.zeros((pad * nx, pad * ny), complex)
    big[nx : 2 * nx, ny : 2 * ny] = plane
    kx = 2 * np.pi * np.fft.fftfreq(pad * nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(pad * ny, dx)
    kz = np.conj(np.sqrt(k**2 - kx[:, None] ** 2 - ky[None, :] ** 2 + 0j))  # evanescent: decaying
    spec = np.fft.fft2(big)
    return np.array([np.fft.ifft2(spec * np.exp(-1j * kz * d))[nx : 2 * nx, ny : 2 * ny] for d in dz])


@pytest.fixture(scope="module")
def kwave_run() -> tuple[dict, float]:
    kfo = pytest.importorskip("kwave.kspaceFirstOrder")
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource

    c0, rho, f0 = 1500.0, 1000.0, 1e6
    nx, nyz = 40, 24
    dx = c0 / f0 / 4
    kgrid = kWaveGrid([nx, nyz, nyz], [dx, dx, dx])
    dt = 0.3 * dx / c0
    kgrid.setTime(int((nx * dx / c0 + 6 / f0) / dt), dt)
    source = kSource()
    mask = np.zeros((nx, nyz, nyz), bool)
    yy, zz = np.meshgrid(kgrid.y_vec.ravel(), kgrid.z_vec.ravel(), indexing="ij")
    mask[1] = np.hypot(yy, zz) <= 3.5e-3
    source.p_mask = mask
    t = np.arange(kgrid.Nt) * dt
    source.p = (1e5 * np.clip(t * f0 / 2, 0, 1) * np.sin(2 * np.pi * f0 * t))[None, :]
    sensor = kSensor(mask=np.ones((nx, nyz, nyz), bool))
    sensor.mask[:3] = False
    sensor.record = ["p"]
    data = kfo.kspaceFirstOrder(
        kgrid, kWaveMedium(sound_speed=c0, density=rho), source, sensor, pml_size=10, quiet=True
    )
    dataset = {
        "p": np.asarray(data["p"]),
        "dt": dt,
        "x_vec": np.asarray(kgrid.x_vec).ravel(),
        "y_vec": np.asarray(kgrid.y_vec).ravel(),
        "z_vec": np.asarray(kgrid.z_vec).ravel(),
        "sensor_mask": sensor.mask,
        "frequency": f0,
        "sound_speed": c0,
        "density": rho,
        "n_harmonics": 1,
    }
    return dataset, dx


def test_kwave_beam_against_an_angular_spectrum(kwave_run: tuple[dict, float], tmp_path) -> None:
    dataset, _dx = kwave_run
    path = save_dataset(tmp_path / "kwave.npz", "kwave", **dataset)
    res = from_dataset(path)
    assert res.z.size == 40 - 3 and res.grid.nx == 24
    ix, iy = res.axis_ij
    k = 2 * math.pi * dataset["frequency"] / dataset["sound_speed"]
    ours = np.abs(angular_spectrum(res.source_plane[0], res.grid.dx, res.z - res.z[0], k)[:, ix, iy])
    kwave_axis = np.abs(res.axis[:, 0])
    assert np.max(np.abs(ours - kwave_axis)) < 0.06 * kwave_axis.max()
    m = res.metrics()
    assert m["iec"]["power"]["exact"]["source"] > 0 and m["focus"]["p_positive"] > 1e5
    # the beam spreads into k-Wave's PML: the power through the recorded planes only falls
    assert np.all(np.diff(res.power[2:]) < 0.01 * res.power[2])


def test_kwave_through_the_functions(kwave_run: tuple[dict, float]) -> None:
    """The same data through :func:`from_kwave` with a kgrid-like object."""
    from types import SimpleNamespace

    d, _ = kwave_run
    kgrid = SimpleNamespace(dim=3, dt=d["dt"], x_vec=d["x_vec"], y_vec=d["y_vec"], z_vec=d["z_vec"])
    res = ao.from_kwave(
        {"p": d["p"]}, kgrid, frequency=d["frequency"], sound_speed=d["sound_speed"], density=d["density"],
        sensor_mask=d["sensor_mask"], n_harmonics=1,
    )  # fmt: skip
    assert res.metrics()["focus"]["p_positive"] == pytest.approx(
        from_dataset(d, "kwave").metrics()["focus"]["p_positive"]
    )
