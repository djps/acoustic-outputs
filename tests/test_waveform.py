"""Waveforms from harmonics, and their peak pressures."""

from __future__ import annotations

import math

import numpy as np

from acoustic_outputs.waveform import harmonics_of, peak_pressures, waveform


def dense_peaks(h: np.ndarray, nt: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    w = waveform(h, nt)
    return w.max(axis=0), w.min(axis=0)


def test_waveform_and_back() -> None:
    rng = np.random.default_rng(1)
    h = rng.standard_normal((5, 7)) + 1j * rng.standard_normal((5, 7))
    np.testing.assert_allclose(harmonics_of(waveform(h, 32), 5), h, atol=1e-12)
    np.testing.assert_allclose(harmonics_of(waveform(h.T, 32, axis=-1), 5, axis=-1), h.T, atol=1e-12)
    # p = Re P exp(+i omega tau): P = -i is sin(omega tau)
    w = waveform(np.array([[-1j]]), 8)[:, 0]
    np.testing.assert_allclose(w, np.sin(2 * np.pi * np.arange(8) / 8), atol=1e-15)


def test_peaks_between_samples() -> None:
    """A sine whose peak falls half way between samples: refined exactly, unlike the samples."""
    h = np.array([np.exp(-1j * np.pi / 4) * np.ones(3)])  # peak at omega tau = pi / 4
    pp, pm = peak_pressures(h)
    np.testing.assert_allclose(pp, 1.0, rtol=1e-12)
    np.testing.assert_allclose(pm, -1.0, rtol=1e-12)
    assert waveform(h, 4).max() < 0.75  # 4 samples miss it by 29%


def test_peaks_of_random_smooth_waveforms() -> None:
    rng = np.random.default_rng(7)
    k = 6
    h = (rng.standard_normal((k, 400)) + 1j * rng.standard_normal((k, 400))) / np.arange(1, k + 1)[:, None] ** 2
    pp, pm = peak_pressures(h)
    dp, dm = dense_peaks(h)
    np.testing.assert_allclose(pp, dp, rtol=1e-6)
    np.testing.assert_allclose(pm, dm, rtol=1e-6)
    assert pp.shape == (400,)


def test_peaks_of_a_shocked_waveform_are_no_worse_than_samples() -> None:
    k = 64
    n = np.arange(1, k + 1)
    saw = (2j / (math.pi * n) * (-1.0) ** (n + 1)).reshape(-1, 1)
    pp, pm = peak_pressures(saw)
    w = waveform(saw, 4 * k)
    dp, dm = dense_peaks(saw, 65536)
    assert w.max() <= pp[0] <= dp[0] * (1 + 1e-6)
    assert dm[0] * (1 + 1e-6) <= pm[0] <= w.min()


def test_shapes() -> None:
    h = np.zeros((3, 4, 5), dtype=complex)
    h[0] = 2.0
    pp, pm = peak_pressures(h)
    assert pp.shape == pm.shape == (4, 5)
    np.testing.assert_allclose(pp, 2.0)
    np.testing.assert_allclose(pm, -2.0)
    empty = peak_pressures(np.zeros((3, 0), dtype=complex))
    assert empty[0].shape == (0,)
