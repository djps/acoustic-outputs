"""Waveforms from harmonics and back: p(tau) = Re sum_n P_n exp(+i n omega tau)."""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from acoustic_outputs._types import ComplexArray, IntArray, RealArray


def waveform(harmonics: ComplexArray, nt: int, workers: int = 1, axis: int = 0) -> RealArray:
    """Return the waveform p(tau_j), tau_j = j T / nt, from harmonics P_n, n = 1 .. K.

    ``p = Re sum_n P_n exp(+i n omega tau)``; the harmonics run along ``axis`` (0 or -1) and
    the samples replace them there. Harmonics above nt/2 are dropped.
    """
    h = np.moveaxis(harmonics, axis, 0)
    m = min(h.shape[0], nt // 2)
    spec = np.zeros((nt // 2 + 1, *h.shape[1:]), dtype=np.result_type(harmonics.dtype, np.complex64))
    np.multiply(h[:m], nt / 2, out=spec[1 : m + 1])
    spec = np.moveaxis(spec, 0, axis)
    return sfft.irfft(spec, n=nt, axis=axis, workers=workers, overwrite_x=True)


def harmonics_of(wave: RealArray, k: int, workers: int = 1, axis: int = 0) -> ComplexArray:
    """Return harmonics P_n, n = 1 .. k, of a waveform sampled over one period along ``axis``."""
    nt = wave.shape[axis]
    spec = np.moveaxis(sfft.rfft(wave, axis=axis, workers=workers), axis, 0)
    out = np.zeros((k, *spec.shape[1:]), dtype=spec.dtype)
    m = min(k, nt // 2)
    np.multiply(spec[1 : m + 1], 2 / nt, out=out[:m])
    return np.moveaxis(out, 0, axis)


def peak_pressures(harmonics: ComplexArray, workers: int = 1, iterations: int = 4) -> tuple[RealArray, RealArray]:
    """Peak positive and negative pressure (p+, p-) of the waveforms of harmonics (K, ...).

    The waveform is sampled at 4 K points per period. The two largest local maxima of the
    samples (of -p for p-) are refined by Newton steps on the trigonometric series itself, each
    kept within one sample of where it started, where a local maximum of samples must have its
    peak. Sampling alone misses a peak between samples by up to 1 - cos(pi / (4 K)) of the
    fundamental (2% for K = 4); refined, the error is at round-off for smooth waveforms. A
    refined value never falls below the sampled one, so steep (shocked) waveforms are at least
    as accurate as the samples.
    """
    h = np.asarray(harmonics)
    k = h.shape[0]
    rest = h.shape[1:]
    flat = h.reshape(k, -1)
    nt = 4 * k
    wave = waveform(flat, nt, workers)  # (nt, npts)
    n = np.arange(1, k + 1)[:, None]
    sample = 2 * np.pi / nt  # one sample, in omega tau
    cols = np.arange(flat.shape[1])

    def refine(j: IntArray, s: RealArray, sign: float) -> RealArray:
        phi0 = sample * j
        phi = phi0.copy()
        for _ in range(iterations):
            e = flat * np.exp(1j * n * phi[None, :])
            d1 = -(n * e).imag.sum(axis=0)  # d/dphi of Re sum P_n e^{i n phi}
            d2 = -((n**2) * e).real.sum(axis=0)
            ok = sign * d2 < 0  # curving the right way: a maximum of sign * p
            step = np.where(ok, -d1 / np.where(d2 == 0, 1.0, d2), 0.0)
            phi = np.clip(phi + np.clip(step, -sample / 2, sample / 2), phi0 - sample, phi0 + sample)
        value = sign * (flat * np.exp(1j * n * phi[None, :])).real.sum(axis=0)
        return np.maximum(value, s[j, cols])

    def extreme(sign: float) -> RealArray:
        s = sign * wave
        local = (s >= np.roll(s, 1, axis=0)) & (s >= np.roll(s, -1, axis=0))
        ranked = np.where(local, s, -np.inf)
        if nt > 2:
            top = np.argpartition(-ranked, 1, axis=0)[:2]
            best = np.maximum(refine(top[0], s, sign), refine(top[1], s, sign))
        else:
            best = refine(np.argmax(ranked, axis=0), s, sign)
        return sign * best

    return extreme(1.0).reshape(rest), extreme(-1.0).reshape(rest)
