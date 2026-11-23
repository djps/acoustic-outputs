"""Output of FOCUS (the Fast Object-oriented C++ Ultrasound Simulator of Michigan State University).

FOCUS computes pressures on a coordinate grid made by ``set_coordinate_grid([dx dy dz], xmin,
xmax, ymin, ymax, zmin, zmax)`` (or from coordinate vectors), indexed (x, y, z); a 2D grid has
min = max in one transverse axis, and so a singleton axis. Its pressure functions give:

* **CW** (``cw_pressure``, ``fnm_cw``, ``cw_angular_spectrum``): the complex pressure, (nx, ny, nz),
  in absolute phase;
* **KZK** (``kzk_cw``): the harmonics, (nx, ny, nz, n) with ``pressure(x, y, z, n)`` the n-th;
* **transient** (``transient_pressure``, ``fnm_tsd``): the pressure [Pa], (nx, ny, nz, nt), at the
  times ``set_time_samples(deltat, tmin, tmax)`` (tmin + i deltat). It is computed in a lossless
  medium. A tone burst long enough to reach a steady state is fitted for its harmonics as a
  time-domain record (:func:`acoustic_outputs.preprocess.time_domain.from_time_series`).

FOCUS's documentation does not state its time convention (exp(+j omega t) or exp(-j omega t)),
whether the KZK harmonics are in retarded time, or the normalisation of the CW and KZK
pressures. So by default :func:`from_focus` reads the convention from the field, by how fast its
phase turns along z (:func:`acoustic_outputs.preprocess._common.phase_convention`): about -k or
+k per metre in absolute phase, far less in retarded time. That needs planes closer than 0.45
wavelengths; otherwise KZK harmonics are taken as retarded, exp(+j omega t) (give ``phase`` and
``time_sign`` if not). A single frequency (CW) does not depend on the convention: every metric
is the same either way. ``scale`` multiplies a pressure that is
not in Pa (e.g. one computed for a unit surface velocity). The beam is taken to travel along +z,
FOCUS's transducer axis.

Save the grid's vectors (or ``delta`` and the limits) and the medium with the pressure::

    save('focus.mat', 'pressure', 'x', 'y', 'z', 'f0')     % x = xmin:dx:xmax, ...
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from acoustic_outputs._types import RealArray
from acoustic_outputs.preprocess._common import phase_convention
from acoustic_outputs.preprocess.spectral import from_harmonics
from acoustic_outputs.preprocess.time_domain import from_time_series
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results

log = logging.getLogger(__name__)

PhaseChoice = Literal["auto", "retarded", "absolute"]


def grid_vectors(
    delta: Any, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> tuple[RealArray, RealArray, RealArray]:
    """The x, y, z vectors of ``set_coordinate_grid(delta, xmin, xmax, ...)``."""
    d = np.asarray(delta, dtype=float).ravel()
    if d.size != 3:
        raise ValueError("delta must be [dx dy dz]")

    def axis(lo: float, hi: float, step: float) -> RealArray:
        if hi == lo:
            return np.array([float(lo)])
        return lo + step * np.arange(round((hi - lo) / step) + 1)

    return axis(xmin, xmax, d[0]), axis(ymin, ymax, d[1]), axis(zmin, zmax, d[2])


def from_focus(
    pressure: Any,
    *,
    x: Any,
    y: Any,
    z: Any,
    frequency: float,
    sound_speed: float,
    density: float,
    phase: PhaseChoice = "auto",
    time_sign: int | None = None,
    scale: float = 1.0,
    dt: float | None = None,
    start_time: float = 0.0,
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of a FOCUS pressure on a coordinate grid.

    Args:
        pressure: complex (nx, ny, nz) (CW), complex (nx, ny, nz, n) (KZK harmonics) or real
            (nx, ny, nz, nt) (transient) [Pa, after ``scale``].
        x, y, z: the grid's coordinate vectors [m] (see :func:`grid_vectors`); one of x and y
            may have a single point (a 2D grid).
        frequency: f0 [Hz].
        sound_speed, density: c0 and rho0 of the medium.
        phase, time_sign: the convention of complex pressures (default: read from the field;
            a retarded field's time sign cannot be read, and defaults to +1).
        scale: factor to Pa.
        dt, start_time: for a transient: deltat and tmin of ``set_time_samples`` [s].
        n_harmonics, periods, window_start: the fit of a transient, as in
            :func:`acoustic_outputs.preprocess.time_domain.from_time_series`.
        settings, axis_xy, alpha, focusing_depth: as in
            :func:`acoustic_outputs.preprocess.spectral.from_harmonics`.
    """
    p = np.asarray(pressure)
    xv, yv, zv = (np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in (x, y, z))
    if p.ndim == 2:  # MATLAB drops a trailing singleton: (nx, ny) on one plane is not a beam
        raise ValueError(f"the pressure must be (nx, ny, nz[, n or nt]), not {p.shape}")
    if p.shape[:3] != (xv.size, yv.size, zv.size):
        raise ValueError(f"pressure {p.shape} does not fit the grid ({xv.size}, {yv.size}, {zv.size})")
    if xv.size == 1 and yv.size > 1:  # a yz grid: y is the transverse axis
        xv, yv, p = yv, xv, np.swapaxes(p, 0, 1)
    y_out = None if yv.size == 1 else yv
    p = p * scale

    if not np.iscomplexobj(p):
        if p.ndim != 4:
            raise ValueError("a real pressure is a transient: (nx, ny, nz, nt)")
        if dt is None:
            raise ValueError("a transient needs the time step (deltat of set_time_samples)")
        coords = {"x": xv, "z": zv} if y_out is None else {"x": xv, "y": yv, "z": zv}
        return from_time_series(
            p[:, 0] if y_out is None else p,
            dt,
            coords=coords,
            layout="xzt" if y_out is None else "xyzt",
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

    h = p if p.ndim == 4 else p[..., None]  # (x, y, z, n)
    if phase == "auto" or time_sign is None:
        guess = phase_convention(np.transpose(h[..., 0], (2, 0, 1)), zv, frequency, sound_speed)
        if guess is None:
            # one harmonic: no metric depends on the convention; several (KZK): retarded, as KZK codes march
            guess = ("retarded", 1) if h.shape[-1] > 1 else ("absolute", 1)
            (log.warning if h.shape[-1] > 1 else log.debug)(
                "FOCUS field: the planes are too far apart to read the phase convention; taken as %s, "
                "exp(%si omega t) (give phase and time_sign)", guess[0] if phase == "auto" else phase,
                "+" if (time_sign or 1) > 0 else "-")  # fmt: skip
        else:
            log.info("FOCUS field: phase read as %s%s", guess[0],
                     "" if guess[1] == 0 else f", exp({'+' if guess[1] > 0 else '-'}i omega t)")  # fmt: skip
        if phase == "auto":
            phase = guess[0]  # type: ignore[assignment]
        if time_sign is None:
            time_sign = guess[1] or 1
    return from_harmonics(
        h,
        layout="xyzk",
        x=xv,
        y=y_out,
        z=zv,
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        phase=phase,  # type: ignore[arg-type]
        time_sign=time_sign,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )
