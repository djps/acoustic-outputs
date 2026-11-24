"""Output of mSOUND (the mixed-domain method of Gu and Jing, a MATLAB toolbox).

mSOUND marches a field plane by plane from the source plane, like a spatial-marching code, in
one of two forms:

* **frequency-specific (FSMDM)**: ``P_fundamental = Forward3D_fund(mgrid, medium, source_p,
  omega_c, ...)`` and ``P_second = Forward3D_sec(mgrid, medium, P_fundamental, omega_c, ...)``
  return the complex amplitude [Pa] of the fundamental and of the second harmonic on every
  plane: (num_x, num_y, num_z + 1) in 3D, (num_x, num_y + 1) in 2D, where the second axis
  (mgrid.y) is the propagation axis. Plane i is at (i - 1) dz from the source plane. The
  propagator is exp(i K dz) with K = -sqrt(omega^2 / c0^2 - kx^2 - ky^2), so the field is
  p = Re P exp(+i omega t), in absolute phase, and |P| is the peak amplitude (the examples' p0
  is "pressure magnitude"). :func:`from_msound` reads these.
* **time-domain (TMDM)**: ``p_time = Forward3D(mgrid, medium, source_p, sensor_mask, ...)``
  returns (num_t, n_sensor) time series at the points of ``sensor_mask``
  (num_x, num_y, num_z + 1) (2D: (num_x, num_y + 1)), in column-major (MATLAB) order of the
  mask, at the times mgrid.t = 0, dt, ..., t_length. The time is absolute and periodic over the
  window (the method is spectral in t). :func:`from_msound_time_series` fits the harmonics of a
  steady window, as for k-Wave; the mask must be a rectangular block.

Coordinates: mgrid.x (and mgrid.y in 3D) are centred on 0 (``set_grid``'s ``makeDim``);
mgrid.z is centred as well and is *not* the distance from the source, so the planes are placed
at z0 + (i - 1) dz, with z0 = 0 at the source plane unless given. These conventions were checked
on mSOUND v0.1 run in Octave: ``Forward2D_fund`` and ``Forward3D_fund`` match an angular-spectrum
propagation exp(-i kz z) of their source plane (with their evanescent filter) to round-off, and
``Forward2D``'s fitted record matches ``Forward2D_fund`` (``tests/test_real_outputs.py``).

A ``set_grid`` object cannot be read outside MATLAB, and Octave cannot save ``struct(mgrid)``;
save the fields that are needed (MATLAB can also save ``struct(mgrid)``)::

    mgrid_s = struct('dx', mgrid.dx, 'dy', mgrid.dy, 'dz', mgrid.dz, 'dt', mgrid.dt, 'x', mgrid.x, 'y', mgrid.y);
    fc = omega_c / (2 * pi);
    save('-v7', 'msound.mat', 'P_fundamental', 'P_second', 'mgrid_s', 'medium', 'fc')
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from acoustic_outputs._types import RealArray
from acoustic_outputs.preprocess.frequency_domain import from_harmonics
from acoustic_outputs.preprocess.time_domain import from_time_series, sensor_block
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results


def centred_coordinates(n: int, d: float) -> RealArray:
    """mSOUND's ``makeDim`` coordinates: n points of spacing d centred on 0 (one more below 0 if n is even)."""
    start = (-n / 2 + 1 - 0.5) if n % 2 else -n / 2
    return d * (start + np.arange(n))


def _transverse(
    shape: tuple[int, ...], x: Any, y: Any, dx: float | None, dy: float | None
) -> tuple[RealArray, RealArray | None]:
    nx = shape[0]
    xv = np.asarray(x, dtype=float).ravel() if x is not None else None
    if xv is None:
        if dx is None:
            raise ValueError("give the x coordinates (mgrid.x) or the spacing dx")
        xv = centred_coordinates(nx, dx)
    if xv.size != nx:
        raise ValueError(f"x has {xv.size} points, the field {nx}")
    if len(shape) == 2:
        return xv, None
    ny = shape[1]
    yv = np.asarray(y, dtype=float).ravel() if y is not None else None
    if yv is None:
        if dy is None:
            raise ValueError("give the y coordinates (mgrid.y) or the spacing dy")
        yv = centred_coordinates(ny, dy)
    if yv.size != ny:
        raise ValueError(f"y has {yv.size} points, the field {ny}")
    return xv, yv


def from_msound(
    p_fundamental: Any,
    p_second: Any = None,
    *,
    dz: float,
    frequency: float,
    sound_speed: float,
    density: float,
    x: Any = None,
    y: Any = None,
    dx: float | None = None,
    dy: float | None = None,
    z0: float = 0.0,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of mSOUND's frequency-specific fields (``Forward2D_fund``, ``Forward3D_fund``, ``..._sec``).

    Args:
        p_fundamental: the fundamental [Pa], (nx, ny, n_planes) in 3D or (nx, n_planes) in 2D.
        p_second: the second harmonic, the same shape (optional).
        dz: plane spacing [m]: mgrid.dz in 3D, mgrid.dy in 2D (whose second axis is the beam's).
        frequency: f0 = omega_c / (2 pi) [Hz].
        sound_speed, density: c0 (medium.c0) and rho0 for the intensities.
        x, y: transverse coordinates (mgrid.x, mgrid.y); or their spacings ``dx``, ``dy``, for
            mSOUND's centred grid.
        z0: position of the source plane [m].
        settings, axis_xy, alpha, focusing_depth: as in
            :func:`acoustic_outputs.preprocess.frequency_domain.from_harmonics`.
    """
    p1 = np.asarray(p_fundamental)
    if p1.ndim not in (2, 3):
        raise ValueError(f"the fundamental must be (nx, ny, n_planes) or (nx, n_planes), not {p1.shape}")
    h = [p1]
    if p_second is not None:
        p2 = np.asarray(p_second)
        if p2.shape != p1.shape:
            raise ValueError(f"the second harmonic {p2.shape} does not fit the fundamental {p1.shape}")
        h.append(p2)
    xv, yv = _transverse(p1.shape, x, y, dx, dy)
    z = z0 + dz * np.arange(p1.shape[-1])
    return from_harmonics(
        np.stack(h),
        layout="kxz" if p1.ndim == 2 else "kxyz",
        x=xv,
        y=yv,
        z=z,
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        phase="absolute",
        time_sign=1,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )


def from_msound_time_series(
    p_time: Any,
    sensor_mask: Any,
    *,
    dt: float,
    dz: float,
    frequency: float,
    sound_speed: float,
    density: float,
    x: Any = None,
    y: Any = None,
    dx: float | None = None,
    dy: float | None = None,
    z0: float = 0.0,
    start_time: float = 0.0,
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of mSOUND's time-domain output (``Forward2D``, ``Forward3D``).

    Args:
        p_time: (num_t, n_sensor) [Pa], the sensors in column-major order of ``sensor_mask``.
        sensor_mask: (num_x, num_y, num_z + 1) (2D: (num_x, num_y + 1)); a rectangular block.
        dt: mgrid.dt [s]; the first sample is at ``start_time`` (mgrid.t starts at 0).
        dz, x, y, dx, dy, z0: the grid, as in :func:`from_msound`.
        n_harmonics, periods, window_start: the fit, as in
            :func:`acoustic_outputs.preprocess.time_domain.from_time_series`.
    """
    mask = np.asarray(sensor_mask)
    if mask.ndim not in (2, 3):
        raise ValueError(f"the sensor mask must be 2D or 3D, not {mask.shape}")
    block = sensor_block(mask, mask.shape)
    sub = tuple(s.stop - s.start for s in block)
    p = np.asarray(p_time, dtype=float)
    if p.ndim != 2 or p.shape[1] != math.prod(sub):
        raise ValueError(f"p_time {p.shape} does not fit the {math.prod(sub)} sensors of the mask")
    # column-major order of the block: x fastest, then y, then the planes
    blk = np.transpose(p.T.reshape((*sub[::-1], p.shape[0])), (*range(len(sub) - 1, -1, -1), len(sub)))
    xv, yv = _transverse(mask.shape, x, y, dx, dy)
    coords: dict[str, RealArray] = {"x": xv[block[0]], "z": z0 + dz * np.arange(mask.shape[-1])[block[-1]]}
    if yv is not None:
        coords["y"] = yv[block[1]]
    return from_time_series(
        blk,
        dt,
        coords=coords,
        layout="xzt" if mask.ndim == 2 else "xyzt",
        propagation="z",
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        n_harmonics=n_harmonics,
        periods=periods,
        window_start=window_start,
        start_time=start_time,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )
