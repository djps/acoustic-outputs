"""Linear field projection between parallel planes by the angular spectrum method.

In a homogeneous medium, the complex pressure on a plane z = z0 determines it on every
parallel plane: each plane wave of the plane's spatial spectrum, exp(i (kx x + ky y)), travels as
exp(i kz (z - z0)) with kz = sqrt(k̂^2 - kx^2 - ky^2), k̂ = k + i alpha (TS 63587, A.3; the
forward projection of 7.4.2 and the back-projection of Annex D are both of this kind).

:func:`angular_spectrum` projects a plane over a set of distances. The plane is zero-padded
(``pad``) so that the periodic FFT does not wrap the field round, and the result is cropped back
to the original grid. Projecting backwards (negative distances) would make evanescent waves
grow; there, the components with kx^2 + ky^2 > k^2 are removed.

The time convention is exp(-i omega t) by default (the TS's, and BEMPP's, FOCUS's ...), for
which a wave travelling towards +z is exp(+i kz z); ``time_sign=+1`` uses exp(+i omega t).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy import fft as _fft

from acoustic_outputs._types import ComplexArray


def angular_spectrum_planes(
    plane: Any,
    dx: float,
    dy: float,
    distances: Any,
    wavenumber: complex,
    *,
    time_sign: int = -1,
    pad: float = 2.0,
    workers: int = 1,
) -> Iterator[ComplexArray]:
    """Yield the plane projected over each of ``distances`` [m], one (nx, ny) array at a time.

    Args:
        plane: complex pressure (nx, ny) on the source plane, sampled at dx, dy [m].
        distances: distances along +z from the source plane [m].
        wavenumber: k̂ = k + i alpha [rad/m] (alpha >= 0: the wave decays as it travels).
        time_sign: -1 for exp(-i omega t), +1 for exp(+i omega t).
        pad: the padded grid is at least ``pad`` times the plane in each direction.
        workers: threads for the FFTs.
    """
    if time_sign not in (1, -1):
        raise ValueError("time_sign must be +1 or -1")
    p = np.asarray(plane, dtype=complex)
    if p.ndim != 2:
        raise ValueError(f"the plane must be 2D, not {p.shape}")
    nx, ny = p.shape
    mx, my = (_fft.next_fast_len(max(n, int(np.ceil(pad * n)))) for n in (nx, ny))
    kx = 2 * np.pi * _fft.fftfreq(mx, dx)
    ky = 2 * np.pi * _fft.fftfreq(my, dy)
    kk = complex(wavenumber)
    kperp2 = kx[:, None] ** 2 + ky[None, :] ** 2
    kz = np.sqrt(kk**2 - kperp2 + 0j)
    kz = np.where(kz.imag < 0, -kz, kz)  # decaying (and evanescent) branch for +z
    propagating = kperp2 <= kk.real**2
    # for exp(+i omega t) the field is the conjugate: conjugate in, project, conjugate out
    spectrum = _fft.fft2(p if time_sign == -1 else p.conj(), s=(mx, my), workers=workers)
    for d in np.atleast_1d(np.asarray(distances, dtype=float)):
        factor = np.exp(1j * kz * d)
        if d < 0:
            factor = np.where(propagating, factor, 0.0)
        out = _fft.ifft2(spectrum * factor, workers=workers)[:nx, :ny]
        yield out if time_sign == -1 else out.conj()


def angular_spectrum(
    plane: Any,
    dx: float,
    dy: float,
    distances: Any,
    wavenumber: complex,
    *,
    time_sign: int = -1,
    pad: float = 2.0,
    workers: int = 1,
) -> ComplexArray:
    """The plane projected over ``distances``: (n_distances, nx, ny) (see :func:`angular_spectrum_planes`)."""
    return np.stack(list(angular_spectrum_planes(plane, dx, dy, distances, wavenumber, time_sign=time_sign,
                                                 pad=pad, workers=workers)))  # fmt: skip
