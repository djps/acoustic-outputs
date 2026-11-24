"""Time-domain fields on unstructured meshes: pressure time series at the nodes of 3D (or 2D) meshes.

A time-domain finite-element solver (a FEniCSx / DOLFINx wave or Westervelt model stepped in
time, such as FEniCSx-FUS; tetrahedra or hexahedra in 3D, triangles or quadrilaterals in a
planar 2D model) gives the pressure in time at the nodes of an unstructured mesh. Once the
transient has passed, the field is periodic: the harmonics of f0 are fitted at each node, by
least squares, over a steady window (the last ``periods`` periods, or from ``window_start``), as
for k-Wave (:class:`~acoustic_outputs.preprocess._common.HarmonicFit`; a residual above 1% of
the signal is warned of). The harmonics, in absolute phase, are then resampled on planes as in
:mod:`~acoustic_outputs.preprocess.unstructured_frequency_domain`.

Routes in:

* :func:`from_unstructured_time_series`: nodes, real series (n, nt), the time step and
  (optionally) cells;
* :func:`from_unstructured_xdmf_time_series`: a real function that a DOLFINx ``XDMFFile`` holds
  at many times (``write_function(u, t)`` at each saved step). Only the steps of the fitted
  window are read, so a long record is not loaded whole;
* inside a DOLFINx program, a :class:`~acoustic_outputs.preprocess._mesh.DolfinxProbe` records the
  pressure at the points of the planes at each step, exactly and in parallel; the series go to
  :func:`acoustic_outputs.preprocess.time_domain.from_time_series`::

      pts, (nz, nx, ny) = plane_points(x, y, z)
      probe = DolfinxProbe(mesh, pts)
      for step in range(n_steps):
          ...                       # advance u
          probe.record(u)
      p = probe.series().reshape(nz, nx, ny, -1)
      res = from_time_series(p, dt, coords={"z": z, "x": x, "y": y}, layout="zxyt", frequency=f0,
                             sound_speed=c0, density=rho0, start_time=t_first_recorded)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_outputs._types import RealArray
from acoustic_outputs.preprocess._mesh import MeshData, fit_series, read_xdmf, xdmf_time_series
from acoustic_outputs.preprocess.unstructured_frequency_domain import from_unstructured_harmonics
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results


def from_unstructured_time_series(
    points: Any,
    series: Any,
    dt: float,
    *,
    frequency: float,
    sound_speed: float,
    density: float,
    start_time: float = 0.0,
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    detrend: bool = False,
    cells: Any = None,
    cell_type: str | None = None,
    max_edge: float | None = None,
    propagation: str | None = None,
    x: Any = None,
    y: Any = None,
    z: Any = None,
    spacing: float | None = None,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of pressure time series at the nodes of an unstructured 3D (or planar 2D) mesh.

    Args:
        points: node coordinates (n, 3), or (n, 2) for a planar 2D mesh.
        series: real pressure (n, nt) [Pa], sampled every ``dt`` from ``start_time``.
        n_harmonics, periods, window_start, detrend: the fit, done at the nodes, as in
            :func:`acoustic_outputs.preprocess.time_domain.from_time_series`.
        cells, cell_type, max_edge, propagation, x, y, z, spacing, settings, axis_xy, alpha,
        focusing_depth: as in
            :func:`~acoustic_outputs.preprocess.unstructured_frequency_domain.from_unstructured_harmonics`.
    """
    v = np.asarray(series)
    if np.iscomplexobj(v) or v.ndim != 2:
        raise ValueError("time series at the nodes must be real, (n, nt)")
    h = fit_series(v.astype(float), dt, frequency, n_harmonics, periods, window_start, start_time, detrend)
    return from_unstructured_harmonics(
        points, np.moveaxis(h, 0, -1), cells=cells, cell_type=cell_type, max_edge=max_edge,
        propagation=propagation, x=x, y=y, z=z, spacing=spacing, phase="absolute", time_sign=1,
        frequency=frequency, sound_speed=sound_speed, density=density, settings=settings, axis_xy=axis_xy,
        alpha=alpha, focusing_depth=focusing_depth,
    )  # fmt: skip


def from_unstructured_xdmf_time_series(
    path: str | Path | MeshData,
    field: str | Sequence[str] | None = None,
    *,
    frequency: float,
    sound_speed: float,
    density: float,
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    detrend: bool = False,
    **kwargs: Any,
) -> Results:
    """:class:`Results` of a time-domain record in a DOLFINx XDMF file, on its 3D (or planar 2D) mesh.

    Args:
        path: the ``.xdmf`` file (its ``.h5`` beside it), or what ``read_xdmf`` read.
        field: the real function written at every saved step (default the only one).
        n_harmonics, periods, window_start, detrend: the fit; only the steps of the window are read.
        frequency, sound_speed, density: f0 [Hz], c0 [m/s], rho0 [kg/m^3].
        kwargs: the options of :func:`from_unstructured_time_series` (``propagation``,
            ``spacing``, ``x``, ``y``, ``z``, ``settings``, ...).
    """
    mesh = path if isinstance(path, MeshData) else read_xdmf(path)
    rec = xdmf_time_series(mesh, field, frequency, periods, window_start)
    return from_unstructured_time_series(
        rec.points, rec.series, rec.dt, start_time=rec.start_time, cells=rec.cells, cell_type=mesh.cell_type,
        n_harmonics=n_harmonics, periods=periods, window_start=window_start, detrend=detrend,
        frequency=frequency, sound_speed=sound_speed, density=density, **kwargs,
    )  # fmt: skip
