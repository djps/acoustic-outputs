"""Frequency-domain fields on unstructured meshes: harmonics at the nodes of 3D (or planar 2D) meshes.

A frequency-domain finite-element solver (a FEniCSx / DOLFINx Helmholtz model, or the harmonics
of a Westervelt model recast as a sequence of Helmholtz problems) gives the complex amplitudes
of one or more harmonics at the nodes of an unstructured mesh: tetrahedra or hexahedra in 3D,
triangles or quadrilaterals in a planar 2D model. The field parameters of
:class:`~acoustic_outputs.results.Results` are computed from transverse planes on a regular grid,
so the nodal values are resampled, one plane at a time, on a structured grid: planes across the
beam (``propagation``, default z in 3D and x in 2D), each a regular grid of the other
coordinates. The interpolation is linear over the cells (split into tetrahedra or triangles),
or over a Delaunay triangulation of the nodes; grid points outside the mesh are zero. Only one
plane of the grid is held at a time.

The grid defaults to the box of the nodes, at the median spacing of the nodes (``spacing``),
centred across the beam; give the coordinate vectors ``x``, ``y``, ``z`` (the mesh's own axes) to
choose it. The beam axis defaults to the centre of the planes (``axis_xy`` sets it, in the
planes' coordinates: the two transverse axes in their order).

The phase convention (``phase="auto"``) is read at the nodes, from the gradient of the
fundamental's phase along the beam
(:func:`~acoustic_outputs.preprocess._mesh.node_phase_convention`). An absolute-phase field has
its carrier exp(-i n k z) taken out at the nodes before it is interpolated, so that only the
smooth envelope is interpolated linearly.

Routes in:

* :func:`from_unstructured_harmonics`: nodes, complex values and (optionally) cells, e.g.
  DOLFINx's ``V.tabulate_dof_coordinates()`` and ``u.x.array``, or ``mesh.geometry.x`` with
  ``mesh.geometry.dofmap`` for a function of the mesh's degree;
* :func:`from_unstructured_xdmf_harmonics`: complex functions in a DOLFINx ``XDMFFile``;
* inside a DOLFINx program, :func:`sample_dolfinx_planes` evaluates functions of any degree,
  exactly and in parallel, on the planes, for
  :func:`~acoustic_outputs.preprocess.frequency_domain.from_harmonics`.

Time-domain records on such meshes are in :mod:`~acoustic_outputs.preprocess.unstructured_time_domain`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from acoustic_outputs._types import ComplexArray, RealArray
from acoustic_outputs.preprocess._common import plane_grid, record
from acoustic_outputs.preprocess._mesh import (
    Mesh,
    MeshData,
    axis_vector,
    default_spacing,
    evaluate_dolfinx,
    node_phase_convention,
    read_xdmf,
    xdmf_harmonics,
)
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results

log = logging.getLogger(__name__)

PhaseChoice = Literal["auto", "retarded", "absolute"]
AXES = "xyz"


def mesh_points(points: Any) -> RealArray:
    """Node coordinates (n, d): 3D, or 2D (a 3D array whose third column is zero is 2D)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"points must be (n, 2) or (n, 3), not {pts.shape}")
    if pts.shape[1] == 3 and np.all(pts[:, 2] == 0):
        pts = pts[:, :2]
    return pts


def _axes(d: int, propagation: str | None) -> tuple[int, list[int]]:
    prop = propagation or ("z" if d == 3 else "x")
    if prop not in AXES[:d]:
        raise ValueError(f"propagation {prop!r} is not one of the axes {AXES[:d]!r} of a {d}D mesh")
    p = AXES.index(prop)
    return p, [i for i in range(d) if i != p]


def plane_axes(
    pts: RealArray, propagation: str | None, spacing: float | None, coords: dict[str, Any]
) -> tuple[int, list[int], list[RealArray]]:
    """The beam axis, the transverse axes and the grid vector of each axis (in mesh order)."""
    d = pts.shape[1]
    p, transverse = _axes(d, propagation)
    h = default_spacing(pts) if spacing is None else float(spacing)
    vectors = []
    for i in range(d):
        given = coords.get(AXES[i])
        if given is not None:
            v = np.atleast_1d(np.asarray(given, dtype=float)).ravel()
        else:
            lo, hi = float(pts[:, i].min()), float(pts[:, i].max())
            if i == p:
                v = axis_vector(lo, hi, h)
            else:  # symmetric about the middle of the box, so that its centre is a grid point
                mid, half = (lo + hi) / 2, (hi - lo) / 2
                n = int(np.floor(half / h + 1e-9))
                v = mid + h * np.arange(-n, n + 1)
        vectors.append(v)
    return p, transverse, vectors


def _resolve_phase(
    pts: RealArray, h: ComplexArray, axis: int, frequency: float, sound_speed: float,
    phase: PhaseChoice, time_sign: int | None,
) -> tuple[str, int]:  # fmt: skip
    if phase not in ("auto", "retarded", "absolute"):
        raise ValueError("phase must be 'auto', 'retarded' or 'absolute'")
    if time_sign not in (None, 1, -1):
        raise ValueError("time_sign must be +1, -1 or None")
    if phase != "auto" and time_sign is not None:
        return phase, time_sign
    guess = node_phase_convention(pts, h[:, 0], axis, frequency, sound_speed)
    if guess is None:
        guess = ("retarded", 1) if h.shape[1] > 1 else ("absolute", 1)
        (log.warning if h.shape[1] > 1 else log.debug)(
            "mesh field: the nodes are too far apart to read the phase convention; taken as %s "
            "(give phase and time_sign)", guess[0] if phase == "auto" else phase)  # fmt: skip
    else:
        log.info("mesh field: phase read as %s%s", guess[0],
                 "" if guess[1] == 0 else f", exp({'+' if guess[1] > 0 else '-'}i omega t)")  # fmt: skip
    return (guess[0] if phase == "auto" else phase), (time_sign if time_sign is not None else guess[1] or 1)


def from_unstructured_harmonics(
    points: Any,
    values: Any,
    *,
    frequency: float,
    sound_speed: float,
    density: float,
    cells: Any = None,
    cell_type: str | None = None,
    max_edge: float | None = None,
    propagation: str | None = None,
    x: Any = None,
    y: Any = None,
    z: Any = None,
    spacing: float | None = None,
    phase: PhaseChoice = "auto",
    time_sign: int | None = None,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of harmonics at the nodes of an unstructured 3D (or planar 2D) mesh.

    Args:
        points: node coordinates (n, 3), or (n, 2) for a planar 2D mesh (also (n, 3) with a
            zero third column).
        values: complex (n,) (one frequency) or (n, K) (harmonics) [Pa].
        cells, cell_type: the mesh cells, indices into ``points`` (tetrahedra or hexahedra in
            3D, triangles or quadrilaterals in 2D; any degree, the corners are used); default
            type the simplex. Without cells, the nodes are triangulated (see
            :class:`~acoustic_outputs.preprocess._mesh.Mesh`, and ``max_edge``).
        propagation: the mesh axis along the beam (``"x"``, ``"y"``, ``"z"``); default z in 3D,
            x in 2D. It becomes the planes' z; the other axes, in order, their x and y.
        x, y, z: grid vectors of the mesh's axes [m] (default: the box of the nodes at
            ``spacing``, the median node spacing); the transverse ones uniformly spaced.
        phase, time_sign: the convention of complex values (default read from the field).
        frequency, sound_speed, density, settings, axis_xy, alpha, focusing_depth: as in
            :func:`acoustic_outputs.preprocess.frequency_domain.from_harmonics`.
    """
    pts = mesh_points(points)
    d = pts.shape[1]
    v = np.asarray(values)
    if v.shape[0] != pts.shape[0]:
        raise ValueError(f"{v.shape[0]} values for {pts.shape[0]} nodes")
    if not np.iscomplexobj(v):
        raise ValueError(
            "harmonics must be complex amplitudes: a real record in time is read by "
            "unstructured_time_domain.from_unstructured_time_series"
        )
    hv = v.reshape(v.shape[0], -1)  # (n, K)
    k = hv.shape[1]
    p, transverse, vectors = plane_axes(pts, propagation, spacing, {"x": x, "y": y, "z": z})
    zv = vectors[p]
    if np.any(np.diff(zv) <= 0):
        raise ValueError("the plane positions must increase")
    ph, sign = _resolve_phase(pts, hv, p, frequency, sound_speed, phase, time_sign)
    if sign == -1:
        hv = hv.conj()
    if ph == "absolute":
        n = np.arange(1, k + 1)
        hv = hv * np.exp(1j * n[None, :] * (2 * np.pi * frequency / sound_speed) * (pts[:, p : p + 1] - zv[0]))
    mesh = Mesh(pts, cells, cell_type or ("tetrahedron" if d == 3 else "triangle"), max_edge)
    tx = vectors[transverse[0]]
    ty = vectors[transverse[1]] if d == 3 else None
    grid = plane_grid(tx, ty, zv, k)
    shape = (tx.size, 1 if ty is None else ty.size)
    q = np.zeros((shape[0] * shape[1], d))
    mesh_t = np.meshgrid(tx, ty if ty is not None else np.zeros(1), indexing="ij")
    q[:, transverse[0]] = mesh_t[0].ravel()
    if ty is not None:
        q[:, transverse[1]] = mesh_t[1].ravel()
    log.info("resampling %d nodes on %d planes of %d x %d points", pts.shape[0], zv.size, *shape)

    def planes() -> Iterable[tuple[float, ComplexArray]]:
        outside = 0
        for zi in zv:
            q[:, p] = zi
            loc = mesh.locate(q)
            outside += int((~loc.found).sum())
            yield float(zi), np.moveaxis(loc(hv).reshape(*shape, k), -1, 0)
        if outside:
            log.info("%.1f%% of the grid is outside the mesh: set to zero", 100 * outside / (q.shape[0] * zv.size))

    return record(
        planes(),
        grid,
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )


def from_unstructured_xdmf_harmonics(
    path: str | Path | MeshData,
    fields: str | Sequence[str] | None = None,
    *,
    frequency: float,
    sound_speed: float,
    density: float,
    time: float | None = None,
    **kwargs: Any,
) -> Results:
    """:class:`Results` of harmonics in a DOLFINx XDMF file, on its 3D (or planar 2D) mesh.

    Args:
        path: the ``.xdmf`` file (its ``.h5`` beside it), or what ``read_xdmf`` read.
        fields: the complex function(s): one is a single frequency, several the harmonics 1,
            2, ... in order. Default: the only function in the file.
        time: for functions written at several times, the one to use (default the last).
        frequency, sound_speed, density: f0 [Hz], c0 [m/s], rho0 [kg/m^3].
        kwargs: the options of :func:`from_unstructured_harmonics` (``propagation``, ``spacing``,
            ``x``, ``y``, ``z``, ``settings``, ...).
    """
    mesh = path if isinstance(path, MeshData) else read_xdmf(path)
    points, cells, values = xdmf_harmonics(mesh, fields, time)
    return from_unstructured_harmonics(points, values, cells=cells, cell_type=mesh.cell_type, frequency=frequency,
                                       sound_speed=sound_speed, density=density, **kwargs)  # fmt: skip


def plane_points(x: Any, y: Any, z: Any, *, propagation: str = "z") -> tuple[RealArray, tuple[int, ...]]:
    """The points (m, 3) of the planes of a grid, and the shape (nz, nx, ny) they fill (C order).

    ``x``, ``y``, ``z`` are the grid vectors of the mesh's axes; ``propagation`` names the one
    along the beam, whose points are the planes (the other two, in order, the planes' x and y).
    For a planar 2D mesh give ``y=None`` (``propagation`` x or y of the mesh's (x, y); the planes
    are then lines, ny = 1).
    """
    given = [None if v is None else np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in (x, y, z)]
    if given[1] is None:  # 2D: the mesh's axes are x and the second vector
        vecs = [given[0], given[2]]
    else:
        vecs = [v for v in given if v is not None]
    d = len(vecs)
    p, transverse = _axes(d, propagation)
    order = [p, *transverse]
    grids = np.meshgrid(*[vecs[i] for i in order], indexing="ij")
    pts = np.zeros((grids[0].size, 3))
    for g, i in zip(grids, order, strict=True):
        pts[:, i] = g.ravel()
    shape = tuple(vecs[i].size for i in order) + ((1,) if d == 2 else ())
    return pts, shape


def sample_dolfinx_planes(
    functions: Any, x: Any, y: Any, z: Any, *, propagation: str = "z"
) -> ComplexArray:  # pragma: no cover - needs DOLFINx
    """DOLFINx functions (the harmonics, or one) on the planes of a grid: (nz, K, nx, ny).

    The planes are those of :func:`plane_points`. The result goes to
    :func:`~acoustic_outputs.preprocess.frequency_domain.from_harmonics` with the transverse
    vectors (for ``propagation="z"``, the planes' x and y are the mesh's)::

        H = sample_dolfinx_planes([p1, p2, p3], x, y, z)
        res = from_harmonics(H, x=x, y=y, z=z, frequency=f0, sound_speed=c0, density=rho0,
                             phase="absolute")

    For a time-domain run, record the same points at each step with a
    :class:`~acoustic_outputs.preprocess._mesh.DolfinxProbe` instead (see
    :mod:`~acoustic_outputs.preprocess.unstructured_time_domain`).
    """
    pts, shape = plane_points(x, y, z, propagation=propagation)
    vals = evaluate_dolfinx(functions, pts)
    return np.moveaxis(vals.reshape(*shape, -1), -1, 1)
