"""Real output of FOCUS 0.981 and mSOUND v0.1, run in GNU Octave 10 (``tests/data/make_fixtures.m``).

The checks are against physics, not against this package: FOCUS's piston field against the exact
on-axis solution, and mSOUND's marching against an angular-spectrum propagation of its own
source plane. They pin down each program's conventions (FOCUS: exp(+j omega t), pressure in Pa
for a surface velocity of 1 m/s; mSOUND: exp(+i omega t), absolute phase, plane i at (i - 1) dz)
and that the readers apply them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs.datasets import from_dataset, load_dataset

DATA = Path(__file__).parent / "data"
RHO_C = 1000.0 * 1500.0
K = 2 * np.pi * 1e6 / 1500.0


def piston_axis(z: np.ndarray, a: float = 3e-3) -> np.ndarray:
    """|p| on the axis of a baffled piston of radius a, surface velocity 1 m/s."""
    return RHO_C * np.abs(np.exp(-1j * K * z) - np.exp(-1j * K * np.sqrt(z**2 + a**2)))


def test_focus_cw(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        res = from_dataset(DATA / "focus_cw.mat", "focus")  # its coordinate_grid and medium structs, as saved
    assert "absolute, exp(+i omega t)" in caplog.text
    assert res.grid.ny == 1 and res.impedance == pytest.approx(RHO_C)
    assert np.allclose(res.axis_p_plus, piston_axis(res.z), rtol=1e-9, atol=1e-6 * RHO_C)
    # the last axial maximum, a^2 / lambda - lambda / 4 = 5.625 mm, is halfway between two planes
    assert res.metrics()["focus"]["z"] in (pytest.approx(5.5e-3), pytest.approx(5.75e-3))


def test_focus_transient() -> None:
    """fnm_tsd's 25-cycle tone burst, fitted over a steady window, gives the CW field."""
    res = from_dataset(DATA / "focus_tsd.mat", "focus")
    assert np.allclose(res.axis_p_plus, piston_axis(res.z), rtol=1e-4)
    assert np.allclose(res.axis_p_minus, -piston_axis(res.z), rtol=1e-4)


def test_msound_fsmdm_matches_the_angular_spectrum() -> None:
    d = load_dataset(DATA / "msound_fsmdm_2d.mat")
    p, dx, dz = d["P_fundamental"].astype(complex), d["dx"], d["dy"]  # 2D: the beam runs along mgrid.y
    kx = 2 * np.pi * np.fft.fftfreq(p.shape[0], dx)
    kz = np.sqrt((K**2 - kx**2).astype(complex))
    keep = K**2 - 1.05 * kx**2 > 0  # mSOUND's evanescent filter
    planes = np.arange(p.shape[1])
    exact = np.fft.ifft(np.fft.fft(p[:, 0])[:, None] * np.exp(-1j * kz[:, None] * planes * dz) * keep[:, None], axis=0)
    exact[:, 0] = p[:, 0]  # the source plane is not filtered
    assert np.linalg.norm(exact - p) < 1e-6 * np.linalg.norm(p)  # single precision
    res = from_dataset(DATA / "msound_fsmdm_2d.mat")  # recognised as mSOUND
    ref = ao.from_harmonics(np.stack([exact, d["P_second"]]), layout="kxz", x=d["x"], z=planes * dz,
                            phase="absolute", frequency=1e6, sound_speed=1500.0, density=1000.0)  # fmt: skip
    assert res.grid.n_harmonics == 2
    assert np.allclose(res.axis_p_plus, ref.axis_p_plus, rtol=1e-5)
    # the second harmonic grows from nothing at the source
    assert np.abs(d["P_second"][:, 0]).max() == 0 and np.abs(d["P_second"][:, -1]).max() > 1e3


def test_msound_tmdm_matches_the_fsmdm() -> None:
    """Forward2D's linear CW record, fitted, gives Forward2D_fund's field on the sensor block."""
    tm = from_dataset(DATA / "msound_tmdm_2d.mat", axis_xy=(0.0, 0.0))
    d = load_dataset(DATA / "msound_fsmdm_2d.mat")
    fs = ao.from_msound(d["P_fundamental"], dz=d["dy"], x=d["x"], frequency=1e6, sound_speed=1500.0,
                        density=1000.0, axis_xy=(0.0, 0.0))  # fmt: skip
    sel = np.isin(np.round(fs.z, 9), np.round(tm.z, 9))
    assert sel.sum() == tm.z.size == 20
    assert np.allclose(tm.axis_p_plus, fs.axis_p_plus[sel], rtol=1e-5)
