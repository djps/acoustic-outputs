"""Frequency-domain axisymmetric fields: harmonics P_n(r, z) on (r, z) grids, meshes and XDMF files.

A frequency-domain axisymmetric solver (a 2D finite-element Helmholtz model in cylindrical
coordinates, such as a FEniCSx / DOLFINx model of a focused bowl, or the harmonics of a
Westervelt model recast as a sequence of Helmholtz problems; or an axisymmetric KZK or
angular-spectrum code) gives the complex amplitudes of one or more harmonics on a half plane
(r >= 0, z). The field parameters of :class:`~acoustic_outputs.results.Results` are computed from
transverse planes, so each plane z = const is **revolved** onto a square (x, y) grid,
P(x, y) = P(r = sqrt(x^2 + y^2)), and recorded as for any 3D field. Power, areas and widths are
those of the 3D beam (not per metre, as for a 2D Cartesian field), and every IEC parameter is
available.

The radial profile is interpolated by a cubic spline of its even extension, P(-r) = P(r), so
its slope is zero on the axis as symmetry requires; points beyond ``r_max`` (by default the last
radius) are zero. The (x, y) spacing ``dx`` defaults to the smallest radial spacing.

Routes in:

* :func:`from_axisymmetric_harmonics`: harmonics on a structured (r, z) grid, axes named by
  ``layout`` from ``z`` (planes), ``k`` (harmonics, absent for one frequency) and ``r``;
* :func:`from_axisymmetric_mesh_harmonics`: harmonics at the nodes of a mesh, such as the degrees
  of freedom of a finite-element solution (``V.tabulate_dof_coordinates()`` and ``u.x.array``),
  resampled on an (r, z) grid by linear interpolation over the cells (or a Delaunay
  triangulation of the nodes);
* :func:`from_axisymmetric_xdmf_harmonics`: complex functions in a DOLFINx ``XDMFFile``;
* inside a DOLFINx program, :func:`sample_dolfinx_rz` evaluates functions of any degree, exactly
  and in parallel, on an (r, z) grid, for :func:`from_axisymmetric_harmonics`.

Time-domain axisymmetric records are in :mod:`~acoustic_outputs.preprocess.axisymmetric_time_domain`.

Which mesh coordinate is r is set by ``coordinates``: ``"rz"`` (x[0] = r, x[1] = z, as in the
DOLFINx axisymmetric demo) or ``"zr"`` (x[0] = z along the beam, x[1] = r).

The phase convention (``phase="auto"``) is read from the field as for FOCUS
(:func:`acoustic_outputs.preprocess._common.phase_convention`); it changes only the time frame of
the waveforms written for output points, not the metrics. A mesh field in absolute phase has
its carrier exp(-i n k z) taken out at the nodes before it is interpolated, so that only the
smooth envelope is interpolated linearly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from acoustic_outputs._types import ComplexArray, RealArray
from acoustic_outputs.preprocess._common import parse_layout, phase_convention, plane_grid, record, to_retarded
from acoustic_outputs.preprocess._mesh import (
    Mesh,
    MeshData,
    base_type,
    default_spacing,
    evaluate_dolfinx,
    read_xdmf,
    xdmf_harmonics,
)
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results

log = logging.getLogger(__name__)

PhaseChoice = Literal["auto", "retarded", "absolute"]
Coordinates = Literal["rz", "zr"]


# -- revolving a radial profile onto (x, y) planes --------------------------------------------


def radii(r: Any) -> RealArray:
    """The radii as a vector: at least 2, non-negative and increasing."""
    rv = np.asarray(r, dtype=float).ravel()
    if rv.size < 2 or rv[0] < 0 or np.any(np.diff(rv) <= 0):
        raise ValueError("the radii must be at least 2, non-negative and increasing")
    return rv


class Revolver:
    """Revolve radial profiles P(r) onto the square (x, y) grid ``x = y = dx * (-n .. n)``.

    Args:
        r: radii of the profiles [m], increasing, from 0 or more (need not be uniform).
        dx: spacing of the (x, y) grid [m]; default the smallest radial spacing.
        r_max: radius of the revolved disc [m]; default (and at most) ``r[-1]``.
    """

    def __init__(self, r: Any, dx: float | None = None, r_max: float | None = None) -> None:
        self.r = radii(r)
        self.dx = float(np.min(np.diff(self.r))) if dx is None else float(dx)
        self.r_max = float(self.r[-1]) if r_max is None else float(r_max)
        if self.dx <= 0 or self.r_max <= 0:
            raise ValueError("dx and r_max must be positive")
        if self.r_max > self.r[-1] * (1 + 1e-9):
            raise ValueError(f"r_max {self.r_max} is beyond the last radius {self.r[-1]}")
        n = int(np.floor(self.r_max / self.dx + 1e-9))
        if n < 1:
            raise ValueError("dx is larger than r_max: no plane to revolve onto")
        self.x = self.dx * np.arange(-n, n + 1)
        rho = np.hypot(self.x[:, None], self.x[None, :])
        self.inside = rho <= self.r_max * (1 + 1e-12)
        self.rho, self.inverse = np.unique(rho[self.inside], return_inverse=True)
        # the even extension P(-r) = P(r): zero slope on the axis
        start = 1 if self.r[0] == 0 else 0
        self._r_even = np.concatenate([-self.r[start:][::-1], self.r])
        self._even = np.concatenate([np.arange(self.r.size)[start:][::-1], np.arange(self.r.size)])

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.size, self.x.size

    def __call__(self, profiles: ComplexArray) -> ComplexArray:
        """Revolve ``profiles`` (K, nr) to planes (K, nx, nx)."""
        from scipy.interpolate import CubicSpline

        if profiles.shape[-1] != self.r.size:
            raise ValueError(f"profiles of {profiles.shape[-1]} radii, expected {self.r.size}")
        spline = CubicSpline(self._r_even, profiles[:, self._even], axis=1)
        out = np.zeros((profiles.shape[0], *self.shape), dtype=complex)
        out[:, self.inside] = spline(self.rho)[:, self.inverse]
        return out


# -- structured (r, z) grids --------------------------------------------------------------------


def standard_layout(harmonics: Any, layout: str) -> ComplexArray:
    """``harmonics`` as (nz, K, nr), from axes named by ``layout`` (``z``, ``r``, optional ``k``)."""
    arr = np.asarray(harmonics)
    parse_layout(layout, arr.ndim, "zkr")
    if "z" not in layout or "r" not in layout:
        raise ValueError("the layout must name the plane axis 'z' and the radial axis 'r'")
    full = layout if "k" in layout else layout + "k"
    arr = arr.reshape(arr.shape + (1,) * (len(full) - arr.ndim))
    return np.transpose(arr, [full.index(c) for c in "zkr"]).astype(complex, copy=False)


def _resolve_phase(
    h: ComplexArray, z: RealArray, frequency: float, sound_speed: float, phase: PhaseChoice, time_sign: int | None
) -> tuple[str, int]:
    """(phase, time_sign) of harmonics (nz, K, nr): given, or read from the fundamental."""
    if phase not in ("auto", "retarded", "absolute"):
        raise ValueError("phase must be 'auto', 'retarded' or 'absolute'")
    if time_sign not in (None, 1, -1):
        raise ValueError("time_sign must be +1, -1 or None")
    if phase != "auto" and time_sign is not None:
        return phase, time_sign
    guess = phase_convention(h[:, 0, :, None], z, frequency, sound_speed)
    if guess is None:
        guess = ("retarded", 1) if h.shape[1] > 1 else ("absolute", 1)
        (log.warning if h.shape[1] > 1 else log.debug)(
            "axisymmetric field: the planes are too far apart to read the phase convention; taken as %s "
            "(give phase and time_sign)", guess[0] if phase == "auto" else phase)  # fmt: skip
    else:
        log.info("axisymmetric field: phase read as %s%s", guess[0],
                 "" if guess[1] == 0 else f", exp({'+' if guess[1] > 0 else '-'}i omega t)")  # fmt: skip
    return (guess[0] if phase == "auto" else phase), (time_sign if time_sign is not None else guess[1] or 1)


def _check_z(z: Any, n: int) -> RealArray:
    zv = np.atleast_1d(np.asarray(z, dtype=float)).ravel()
    if zv.size != n:
        raise ValueError(f"{zv.size} z coordinates for {n} planes")
    if np.any(np.diff(zv) <= 0):
        raise ValueError("the z coordinates must increase")
    return zv


def from_axisymmetric_harmonics(
    harmonics: Any,
    *,
    r: Any,
    z: Any,
    layout: str = "zkr",
    frequency: float,
    sound_speed: float,
    density: float,
    phase: PhaseChoice = "auto",
    time_sign: int | None = None,
    dx: float | None = None,
    r_max: float | None = None,
    settings: RecordSettings | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of an axisymmetric field given by its harmonics on an (r, z) grid.

    Args:
        harmonics: complex amplitudes [Pa], axes named by ``layout``.
        r: radii [m], increasing from 0 (or more); need not be uniform.
        z: plane positions [m], increasing; need not be uniform.
        layout: order of the axes of ``harmonics``: ``z``, ``r`` and, for several harmonics, ``k``.
        frequency: f0 [Hz]; harmonic n is at n f0.
        sound_speed, density: c0 [m/s] and rho0 [kg/m^3].
        phase, time_sign: the convention, as in
            :func:`acoustic_outputs.preprocess.frequency_domain.from_harmonics`; default read from
            the field.
        dx: spacing of the revolved (x, y) planes [m] (default the smallest radial spacing).
        r_max: radius of the revolved planes [m] (default the last radius).
        settings, alpha, focusing_depth: as in
            :func:`~acoustic_outputs.preprocess.frequency_domain.from_harmonics`.
    """
    h = standard_layout(harmonics, layout)
    nz, k, nr = h.shape
    zv = _check_z(z, nz)
    rev = Revolver(r, dx, r_max)
    if rev.r.size != nr:
        raise ValueError(f"{rev.r.size} radii for a radial axis of {nr}")
    ph, sign = _resolve_phase(h, zv, frequency, sound_speed, phase, time_sign)
    if sign == -1:
        h = h.conj()

    def planes() -> Iterable[tuple[float, ComplexArray]]:
        for iz in range(nz):
            p = h[iz]
            if ph == "absolute":
                p = to_retarded(p, float(zv[iz]), float(zv[0]), frequency, sound_speed)
            yield float(zv[iz]), rev(p)

    log.info("revolving %d planes onto %d x %d points of %.4g mm", nz, rev.x.size, rev.x.size, rev.dx * 1e3)
    return record(
        planes(),
        plane_grid(rev.x, rev.x, zv, k),
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        settings=settings,
        axis_xy=(0.0, 0.0),
        alpha=alpha,
        focusing_depth=focusing_depth,
    )


# -- meshes ---------------------------------------------------------------------------------------


def rz_nodes(points: Any, coordinates: Coordinates) -> RealArray:
    """Node coordinates as (r, z), from (n, 2) or (n, 3) (third column zero) points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points must be (n, 2) or (n, 3), not {pts.shape}")
    if coordinates not in ("rz", "zr"):
        raise ValueError("coordinates must be 'rz' or 'zr'")
    if pts.shape[1] == 3 and np.any(pts[:, 2] != 0):
        raise ValueError("the points have a non-zero third coordinate: not a 2D (r, z) mesh")
    rz = pts[:, :2] if coordinates == "rz" else pts[:, 1::-1]
    if np.any(rz[:, 0] < -1e-9 * np.ptp(rz[:, 0])):
        raise ValueError("some points have r < 0: check 'coordinates' (rz or zr)")
    return rz


def rz_grid(
    points: Any,
    *,
    coordinates: Coordinates = "rz",
    spacing: float | None = None,
    r: Any = None,
    z: Any = None,
) -> tuple[RealArray, RealArray]:
    """The (r, z) vectors on which nodes are resampled: given, or over the nodes' extent.

    The default spacing (``spacing``, in both r and z) is the median distance from a node to
    its nearest neighbour; r runs from 0.
    """
    rz = rz_nodes(points, coordinates)
    h = default_spacing(rz) if spacing is None else float(spacing)
    if r is None:
        r = h * np.arange(int(np.floor(rz[:, 0].max() / h + 1e-9)) + 1)
    if z is None:
        z0, z1 = rz[:, 1].min(), rz[:, 1].max()
        z = z0 + h * np.arange(int(np.floor((z1 - z0) / h + 1e-9)) + 1)
    return radii(r), np.asarray(z, dtype=float).ravel()


def grid_locator(rz: RealArray, r: Any, z: Any, cells: Any, cell_type: str, max_edge: float | None) -> Any:
    """A function from node values (n, ...) to the (r, z) grid (nr, nz, ...)."""
    rv, zv = np.asarray(r, dtype=float).ravel(), np.asarray(z, dtype=float).ravel()
    q = np.stack(np.meshgrid(rv, zv, indexing="ij"), axis=-1).reshape(-1, 2)
    if cells is not None and base_type(cell_type) not in ("triangle", "quadrilateral"):
        raise ValueError(f"cells of type {cell_type!r}: an (r, z) mesh has triangles or quadrilaterals")
    loc = Mesh(rz, cells, cell_type, max_edge).locate(q)
    missing = (~loc.found).mean()
    if missing > 0:
        log.info("%.1f%% of the (r, z) grid is outside the mesh: set to zero", 100 * missing)

    def to_grid(values: Any) -> Any:
        v = np.asarray(values)
        return loc(v).reshape(rv.size, zv.size, *v.shape[1:])

    return to_grid


def resample(
    points: Any,
    values: Any,
    r: Any,
    z: Any,
    *,
    cells: Any = None,
    cell_type: str = "triangle",
    coordinates: Coordinates = "rz",
    max_edge: float | None = None,
) -> Any:
    """Values at the nodes of a mesh, linearly interpolated onto the (r, z) grid: (nr, nz, ...).

    Args:
        points: node coordinates (n, 2) (or (n, 3) with a zero third column).
        values: (n, ...) at the nodes.
        r, z: the grid vectors [m].
        cells: the mesh cells (m, nodes per cell), indices into ``points``: the interpolation
            then follows the mesh, and points outside it are zero. Without cells, a Delaunay
            triangulation of the nodes is used, less its triangles with an edge longer than
            ``max_edge`` (default 5 times the median edge), which would bridge a concave part
            of the domain (such as the cavity behind a bowl).
        cell_type: ``triangle`` or ``quadrilateral`` (any degree: the corner nodes are used).
        coordinates: ``"rz"`` or ``"zr"``: which column of ``points`` is r.
    """
    rz = rz_nodes(points, coordinates)
    v = np.asarray(values)
    if v.shape[0] != rz.shape[0]:
        raise ValueError(f"{v.shape[0]} values for {rz.shape[0]} nodes")
    return grid_locator(rz, r, z, cells, cell_type, max_edge)(v)


def from_axisymmetric_mesh_harmonics(
    points: Any,
    values: Any,
    *,
    frequency: float,
    sound_speed: float,
    density: float,
    cells: Any = None,
    cell_type: str = "triangle",
    coordinates: Coordinates = "rz",
    spacing: float | None = None,
    r: Any = None,
    z: Any = None,
    max_edge: float | None = None,
    phase: PhaseChoice = "auto",
    time_sign: int | None = None,
    dx: float | None = None,
    r_max: float | None = None,
    settings: RecordSettings | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of axisymmetric harmonics at the nodes of an (r, z) mesh.

    Args:
        points: node coordinates (n, 2) (or (n, 3), the third zero), e.g. DOLFINx's
            ``V.tabulate_dof_coordinates()`` or the mesh geometry ``mesh.geometry.x``.
        values: complex (n,) (one frequency) or (n, K) (harmonics) [Pa].
        cells, cell_type, max_edge: the mesh, for the interpolation (see :func:`resample`).
        coordinates: ``"rz"`` (x[0] = r) or ``"zr"`` (x[0] = z).
        spacing, r, z: the (r, z) grid of the resampling (see :func:`rz_grid`).
        phase, time_sign, dx, r_max, frequency, sound_speed, density, settings, alpha,
        focusing_depth: as in :func:`from_axisymmetric_harmonics`.
    """
    v = np.asarray(values)
    if not np.iscomplexobj(v):
        raise ValueError(
            "harmonics must be complex amplitudes: a real record in time is read by "
            "axisymmetric_time_domain.from_axisymmetric_mesh_time_series"
        )
    rz = rz_nodes(points, coordinates)
    if v.shape[0] != rz.shape[0]:
        raise ValueError(f"{v.shape[0]} values for {rz.shape[0]} nodes")
    hv = v.reshape(v.shape[0], -1)  # (n, K)
    rv, zv = rz_grid(points, coordinates=coordinates, spacing=spacing, r=r, z=z)
    to_grid = grid_locator(rz, rv, zv, cells, cell_type, max_edge)
    if phase == "auto" or time_sign is None:
        phase, time_sign = _resolve_phase(  # type: ignore[assignment]
            np.transpose(to_grid(hv), (1, 2, 0)), zv, frequency, sound_speed, phase, time_sign
        )
    if time_sign == -1:
        hv = hv.conj()
    if phase == "absolute":
        # take out the carrier exp(-i n k z) at the nodes, so that only the slowly varying
        # envelope is interpolated (a linear interpolation of the carrier loses amplitude)
        n = np.arange(1, hv.shape[1] + 1)
        hv = hv * np.exp(1j * n[None, :] * (2 * np.pi * frequency / sound_speed) * (rz[:, 1:2] - zv[0]))
    return from_axisymmetric_harmonics(
        to_grid(hv), r=rv, z=zv, layout="rzk", frequency=frequency, sound_speed=sound_speed, density=density,
        phase="retarded", time_sign=1, dx=dx, r_max=r_max, settings=settings, alpha=alpha,
        focusing_depth=focusing_depth,
    )  # fmt: skip


def from_axisymmetric_xdmf_harmonics(
    path: str | Path | MeshData,
    fields: str | Sequence[str] | None = None,
    *,
    frequency: float,
    sound_speed: float,
    density: float,
    time: float | None = None,
    **kwargs: Any,
) -> Results:
    """:class:`Results` of axisymmetric harmonics in a DOLFINx XDMF file (see ``read_xdmf``).

    Args:
        path: the ``.xdmf`` file (its ``.h5`` beside it), or what ``read_xdmf`` read.
        fields: the complex function(s): one is a single frequency, several the harmonics 1,
            2, ... in order. Default: the only function in the file.
        time: for functions written at several times, the time to use (default the last).
        frequency, sound_speed, density: f0 [Hz], c0 [m/s], rho0 [kg/m^3].
        kwargs: the options of :func:`from_axisymmetric_mesh_harmonics` (``coordinates``,
            ``spacing``, ``r``, ``z``, ``settings``, ...).
    """
    mesh = path if isinstance(path, MeshData) else read_xdmf(path)
    if not mesh.planar:
        raise ValueError("the mesh is 3D: read it with unstructured_frequency_domain.from_unstructured_xdmf_harmonics")
    points, cells, values = xdmf_harmonics(mesh, fields, time)
    return from_axisymmetric_mesh_harmonics(points, values, cells=cells, cell_type=mesh.cell_type,
                                            frequency=frequency, sound_speed=sound_speed, density=density,
                                            **kwargs)  # fmt: skip


# -- inside a DOLFINx program ------------------------------------------------------------------------


def rz_points(r: Any, z: Any, coordinates: Coordinates = "rz") -> RealArray:
    """The points (nr * nz, 3) of the (r, z) grid in mesh coordinates, r varying slowest."""
    rv, zv = np.asarray(r, dtype=float).ravel(), np.asarray(z, dtype=float).ravel()
    R, Z = np.meshgrid(rv, zv, indexing="ij")
    pts = np.zeros((R.size, 3))
    ir, iz = (0, 1) if coordinates == "rz" else (1, 0)
    pts[:, ir], pts[:, iz] = R.ravel(), Z.ravel()
    return pts


def sample_dolfinx_rz(functions: Any, r: Any, z: Any, *, coordinates: Coordinates = "rz") -> Any:  # pragma: no cover
    """Evaluate DOLFINx functions on the (r, z) grid: (nr, nz) for one, (nr, nz, K) for a list.

    Exact for any degree (``Function.eval``), and parallel: every rank gets the whole grid,
    each point evaluated on a rank that owns it. Points outside the mesh are zero. The result
    goes to :func:`from_axisymmetric_harmonics` with ``layout="rz"`` or ``"rzk"``::

        H = sample_dolfinx_rz([p1, p2, p3], r, z)
        res = from_axisymmetric_harmonics(H, r=r, z=z, layout="rzk", frequency=f0, sound_speed=c0,
                                          density=rho0)
    """
    rv, zv = np.asarray(r).ravel(), np.asarray(z).ravel()
    out = evaluate_dolfinx(functions, rz_points(rv, zv, coordinates)).reshape(rv.size, zv.size, -1)
    return out if isinstance(functions, (list, tuple)) else out[..., 0]
