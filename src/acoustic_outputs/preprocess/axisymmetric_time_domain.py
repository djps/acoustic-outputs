"""Time-domain axisymmetric fields: pressure p(r, z, t) on (r, z) grids, meshes and XDMF files.

A time-domain axisymmetric solver (a 2D finite-element wave or Westervelt model in cylindrical
coordinates stepped in time, as a FEniCSx / DOLFINx model of a focused bowl would be; or an
axisymmetric k-space or FDTD code) gives the pressure in time on a half plane (r >= 0, z). Once
the transient has passed, the field is periodic: the harmonics of f0 are fitted, by least
squares, over a steady window (the last ``periods`` periods, or from ``window_start``), as for
k-Wave (:class:`~acoustic_outputs.preprocess._common.HarmonicFit`; a residual above 1% of the
signal is warned of). The harmonics are then those of the frequency-domain case, and the planes
are revolved onto (x, y) as in
:mod:`~acoustic_outputs.preprocess.axisymmetric_frequency_domain`.

Routes in:

* :func:`from_axisymmetric_time_series`: time series on a structured (r, z) grid, axes named by
  ``layout`` from ``r``, ``z`` and ``t``;
* :func:`from_axisymmetric_mesh_time_series`: time series (n, nt) at the nodes of a mesh, fitted
  at the nodes and then resampled on an (r, z) grid;
* :func:`from_axisymmetric_xdmf_time_series`: a real function that a DOLFINx ``XDMFFile`` holds
  at many times (``write_function(u, t)`` at each saved step). Only the steps of the fitted
  window are read, so a long record is not loaded whole;
* inside a DOLFINx program, a :class:`~acoustic_outputs.preprocess._mesh.DolfinxProbe` on
  :func:`~acoustic_outputs.preprocess.axisymmetric_frequency_domain.rz_points` records the
  pressure on an (r, z) grid at each step, exactly and in parallel, for
  :func:`from_axisymmetric_time_series`.

The fitted harmonics are in absolute phase, p = Re sum P_n exp(+i n omega t), with t the time of
the record (the first sample is at ``start_time``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from acoustic_outputs._types import RealArray
from acoustic_outputs.preprocess._common import parse_layout
from acoustic_outputs.preprocess._mesh import MeshData, fit_series, read_xdmf, xdmf_time_series
from acoustic_outputs.preprocess.axisymmetric_frequency_domain import (
    Coordinates,
    from_axisymmetric_harmonics,
    from_axisymmetric_mesh_harmonics,
)
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results


def from_axisymmetric_time_series(
    p: Any,
    dt: float,
    *,
    r: Any,
    z: Any,
    layout: str = "rzt",
    frequency: float,
    sound_speed: float,
    density: float,
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    start_time: float = 0.0,
    detrend: bool = False,
    dx: float | None = None,
    r_max: float | None = None,
    settings: RecordSettings | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of pressure time series on an (r, z) grid.

    Args:
        p: pressure [Pa], axes named by ``layout`` from ``r``, ``z`` and ``t``.
        dt: time step [s]; the first sample is at ``start_time``.
        n_harmonics, periods, window_start, detrend: the fit, as in
            :func:`acoustic_outputs.preprocess.time_domain.from_time_series`.
        r, z, dx, r_max, frequency, sound_speed, density, settings, alpha, focusing_depth: as in
            :func:`~acoustic_outputs.preprocess.axisymmetric_frequency_domain.from_axisymmetric_harmonics`.
    """
    arr = np.asarray(p, dtype=float)
    parse_layout(layout, arr.ndim, "rzt")
    if set(layout) != set("rzt"):
        raise ValueError("the layout must name the axes 'r', 'z' and 't'")
    arr = np.transpose(arr, [layout.index(c) for c in "zrt"])
    h = fit_series(arr, dt, frequency, n_harmonics, periods, window_start, start_time, detrend)  # (K, nz, nr)
    return from_axisymmetric_harmonics(
        h, r=r, z=z, layout="kzr", frequency=frequency, sound_speed=sound_speed, density=density,
        phase="absolute", time_sign=1, dx=dx, r_max=r_max, settings=settings, alpha=alpha,
        focusing_depth=focusing_depth,
    )  # fmt: skip


def from_axisymmetric_mesh_time_series(
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
    cell_type: str = "triangle",
    coordinates: Coordinates = "rz",
    spacing: float | None = None,
    r: Any = None,
    z: Any = None,
    max_edge: float | None = None,
    dx: float | None = None,
    r_max: float | None = None,
    settings: RecordSettings | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of pressure time series at the nodes of an (r, z) mesh.

    Args:
        points: node coordinates (n, 2) (or (n, 3), the third zero).
        series: real pressure (n, nt) [Pa], sampled every ``dt`` from ``start_time``.
        n_harmonics, periods, window_start, detrend: the fit, done at the nodes.
        cells, cell_type, coordinates, spacing, r, z, max_edge, dx, r_max, settings, alpha,
        focusing_depth: as in
            :func:`~acoustic_outputs.preprocess.axisymmetric_frequency_domain.from_axisymmetric_mesh_harmonics`.
    """
    v = np.asarray(series)
    if np.iscomplexobj(v) or v.ndim != 2:
        raise ValueError("time series at the nodes must be real, (n, nt)")
    h = fit_series(v.astype(float), dt, frequency, n_harmonics, periods, window_start, start_time, detrend)
    return from_axisymmetric_mesh_harmonics(
        points, np.moveaxis(h, 0, -1), cells=cells, cell_type=cell_type, coordinates=coordinates,
        spacing=spacing, r=r, z=z, max_edge=max_edge, phase="absolute", time_sign=1, dx=dx, r_max=r_max,
        frequency=frequency, sound_speed=sound_speed, density=density, settings=settings, alpha=alpha,
        focusing_depth=focusing_depth,
    )  # fmt: skip


def from_axisymmetric_xdmf_time_series(
    path: str | Path | MeshData,
    field: str | None = None,
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
    """:class:`Results` of a time-domain record in a DOLFINx XDMF file on an (r, z) mesh.

    Args:
        path: the ``.xdmf`` file (its ``.h5`` beside it), or what ``read_xdmf`` read.
        field: the real function written at every saved step (default the only one).
        n_harmonics, periods, window_start, detrend: the fit; only the steps of the window are read.
        frequency, sound_speed, density: f0 [Hz], c0 [m/s], rho0 [kg/m^3].
        kwargs: the options of :func:`from_axisymmetric_mesh_time_series` (``coordinates``,
            ``spacing``, ``r``, ``z``, ``settings``, ...).
    """
    mesh = path if isinstance(path, MeshData) else read_xdmf(path)
    if not mesh.planar:
        raise ValueError("the mesh is 3D: read it with unstructured_time_domain.from_unstructured_xdmf_time_series")
    rec = xdmf_time_series(mesh, field, frequency, periods, window_start)
    return from_axisymmetric_mesh_time_series(
        rec.points, rec.series, rec.dt, start_time=rec.start_time, cells=rec.cells, cell_type=mesh.cell_type,
        n_harmonics=n_harmonics, periods=periods, window_start=window_start, detrend=detrend,
        frequency=frequency, sound_speed=sound_speed, density=density, **kwargs,
    )  # fmt: skip
