"""What the finite-element readers share: cells, point location, linear interpolation and XDMF.

A finite-element solution is known at the nodes of a mesh (its geometry nodes, or the degrees of
freedom of a function space). The readers resample it on structured grids by linear
interpolation over simplices: the cells' corner nodes split into triangles (2D) or tetrahedra
(3D), or, without cells, a Delaunay triangulation of the nodes.

Quadrilaterals and hexahedra are read in either node order: VTK's (XDMF files, counterclockwise
faces) or DOLFINx's (tensor-product, as in ``mesh.geometry.dofmap``), told apart by the shape of
the cells.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_outputs._types import ComplexArray, IntArray, RealArray
from acoustic_outputs.preprocess._common import STEADY_TOLERANCE, HarmonicFit

log = logging.getLogger("acoustic_outputs.preprocess")

#: corner nodes, and the split of the cell into simplices (VTK node order)
_SPLIT: dict[str, tuple[int, list[tuple[int, ...]]]] = {
    "triangle": (3, [(0, 1, 2)]),
    "quadrilateral": (4, [(0, 1, 2), (0, 2, 3)]),
    "tetrahedron": (4, [(0, 1, 2, 3)]),
    # the six tetrahedra around the diagonal 0-6
    "hexahedron": (8, [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]),
}
#: DOLFINx (tensor-product) to VTK order of the corners
_DOLFINX_TO_VTK = {"quadrilateral": [0, 1, 3, 2], "hexahedron": [0, 1, 3, 2, 4, 5, 7, 6]}


def base_type(cell_type: str) -> str:
    """The linear cell type of an XDMF / VTK / DOLFINx name (``Triangle_6`` -> ``triangle``)."""
    name = cell_type.lower().split("_")[0]
    name = {"tet": "tetrahedron", "hex": "hexahedron", "quad": "quadrilateral", "tri": "triangle"}.get(name, name)
    if name not in _SPLIT:
        raise ValueError(f"cells of type {cell_type!r}: only triangles, quadrilaterals, tetrahedra and hexahedra")
    return name


def corners(cells: Any, cell_type: str) -> IntArray:
    """The corner nodes of ``cells`` (any degree: the corners come first)."""
    n = _SPLIT[base_type(cell_type)][0]
    c = np.asarray(cells, dtype=np.int64)
    if c.ndim != 2 or c.shape[1] < n:
        raise ValueError(f"{cell_type} cells need at least {n} nodes each, not {c.shape}")
    return c[:, :n]


def _tensor_order(points: RealArray, c: IntArray) -> bool:
    """Whether quadrilateral faces (corners 0-3) are in tensor-product order, not VTK's.

    In VTK order 0-1-2-3 runs round the face and its diagonals 0-2 and 1-3 cross at their
    midpoints (for a parallelogram); in tensor-product order 0-2 and 1-3 are opposite edges.
    """
    sample = c[:: max(1, len(c) // 200), :4]
    p = points[sample]
    gap = np.linalg.norm((p[:, 0] + p[:, 2]) / 2 - (p[:, 1] + p[:, 3]) / 2, axis=1)
    size = np.linalg.norm(p[:, 0] - p[:, 1], axis=1) + np.linalg.norm(p[:, 0] - p[:, 3], axis=1)
    return bool(np.median(gap / size) > 0.1)


def simplices(cells: Any, cell_type: str, points: Any) -> IntArray:
    """The cells split into triangles (m, 3) or tetrahedra (m, 4) of their corner nodes."""
    name = base_type(cell_type)
    c = corners(cells, cell_type)
    if name in _DOLFINX_TO_VTK and _tensor_order(np.asarray(points, dtype=float), c):
        log.debug("%s cells in tensor-product (DOLFINx) order", name)
        c = c[:, _DOLFINX_TO_VTK[name]]
    return np.concatenate([c[:, list(t)] for t in _SPLIT[name][1]])


def barycentric(vertices: RealArray, q: RealArray) -> RealArray:
    """Barycentric coordinates (n, d + 1) of points q (n, d) in simplices ``vertices`` (n, d + 1, d)."""
    t = np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2)  # (n, d, d)
    rhs = (q - vertices[:, 0])[..., None]
    lam = np.full(q.shape, np.nan)
    ok = np.abs(np.linalg.det(t)) > 0
    if ok.any():
        lam[ok] = np.linalg.solve(t[ok], rhs[ok])[..., 0]
    return np.concatenate([1 - lam.sum(axis=1, keepdims=True), lam], axis=1)


@dataclass
class Locator:
    """Linear interpolation from the nodes at query points: the containing simplex and weights."""

    nodes: IntArray  # (nq, d + 1)
    weights: RealArray  # (nq, d + 1)
    found: Any  # (nq,) bool

    def __call__(self, values: Any) -> Any:
        """``values`` (n, ...) at the nodes, interpolated at the query points (nq, ...); 0 outside."""
        v = np.asarray(values)
        out = np.einsum("qj,qj...->q...", self.weights, v[self.nodes])
        out[~self.found] = 0
        return out


class Mesh:
    """Nodes and simplices in which query points are located.

    Args:
        points: node coordinates (n, d), d = 2 or 3.
        cells, cell_type: the mesh cells (indices into ``points``); without them, a Delaunay
            triangulation of the nodes less its simplices with an edge longer than ``max_edge``
            (default 5 times the median edge), which would bridge a concave part of the domain.
    """

    def __init__(self, points: RealArray, cells: Any = None, cell_type: str = "triangle",
                 max_edge: float | None = None) -> None:  # fmt: skip
        from scipy.spatial import Delaunay, cKDTree

        self.points = np.asarray(points, dtype=float)
        d = self.points.shape[1]
        self._delaunay = None
        if cells is None:
            self._delaunay = Delaunay(self.points)
            self.simplices = self._delaunay.simplices
            v = self.points[self.simplices]
            edges = [np.linalg.norm(v[:, i] - v[:, j], axis=1) for i in range(d + 1) for j in range(i)]
            longest = np.max(edges, axis=0)
            limit = 5.0 * float(np.median(np.concatenate(edges))) if max_edge is None else max_edge
            self._bridging = longest > limit
        else:
            self.simplices = simplices(cells, cell_type, self.points)
            if self.simplices.shape[1] != d + 1:
                raise ValueError(f"{cell_type} cells in {d} dimensions")
            v = self.points[self.simplices]
            self._centroids = v.mean(axis=1)
            self._tree = cKDTree(self._centroids)
            self._reach = float(np.max(np.linalg.norm(v - self._centroids[:, None], axis=2)))

    def locate(self, q: Any) -> Locator:
        """The simplices containing the points ``q`` (nq, d), and their barycentric weights."""
        q = np.asarray(q, dtype=float)
        nq = q.shape[0]
        eps = 1e-9
        if self._delaunay is not None:
            s = self._delaunay.find_simplex(q)
            found = s >= 0
            found &= ~self._bridging[np.where(found, s, 0)]
            nodes = self.simplices[np.where(found, s, 0)]
        else:
            nodes = np.zeros((nq, self.simplices.shape[1]), dtype=np.int64)
            found = np.zeros(nq, dtype=bool)
            kk = min(32, len(self.simplices))
            dist, cand = self._tree.query(q, k=kk)
            dist, cand = dist.reshape(nq, kk), cand.reshape(nq, kk)
            for j in range(kk):
                todo = ~found & (dist[:, j] <= self._reach)
                if not todo.any():
                    break
                t = self.simplices[cand[todo, j]]
                inside = np.all(barycentric(self.points[t], q[todo]) >= -eps, axis=1)
                idx = np.flatnonzero(todo)[inside]
                nodes[idx], found[idx] = t[inside], True
            # points not in their nearest simplices: all within reach (slivers, graded meshes)
            rest = np.flatnonzero(~found & (dist[:, -1] <= self._reach))
            for i in rest:
                for c in self._tree.query_ball_point(q[i], self._reach):
                    t = self.simplices[c]
                    if np.all(barycentric(self.points[t][None], q[i][None]) >= -eps):
                        nodes[i], found[i] = t, True
                        break
        weights = barycentric(self.points[nodes], q)
        weights[~found] = 0
        return Locator(nodes=nodes, weights=np.nan_to_num(weights), found=found)


def default_spacing(points: RealArray) -> float:
    """The median distance from a node to its nearest neighbour."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def axis_vector(lo: float, hi: float, h: float) -> RealArray:
    """Points ``lo + i h`` up to ``hi``."""
    return lo + h * np.arange(int(np.floor((hi - lo) / h + 1e-9)) + 1)


def fit_series(
    series: RealArray,
    dt: float,
    frequency: float,
    n_harmonics: int,
    periods: float | None,
    window_start: float | None,
    start_time: float,
    detrend: bool,
) -> ComplexArray:
    """Harmonics (K, ...) of series (..., nt), absolute phase, with a check of the steady state."""
    fit = HarmonicFit(series.shape[-1], dt, frequency, n_harmonics, periods=periods, start_time=start_time,
                      window_start=window_start, detrend=detrend)  # fmt: skip
    log.info("harmonic fit over %.2f periods, %d harmonics", fit.window_periods, n_harmonics)
    h, rel = fit(series, time_axis=-1)
    if rel > STEADY_TOLERANCE:
        log.warning("the harmonic fit leaves %.2g of the signal (rms): the record is not a steady periodic "
                    "field over the window (transient, too few harmonics, or noise)", rel)  # fmt: skip
    return h


def node_phase_convention(
    points: RealArray, fundamental: ComplexArray, axis: int, frequency: float, sound_speed: float
) -> tuple[str, int] | None:
    """Guess (phase, time_sign) of a fundamental at nodes, travelling along coordinate ``axis``.

    The phase gradient is fitted, by weighted least squares, to the phase differences between
    each node and its nearest neighbours; its component along the beam is about -k (absolute,
    exp(+i omega t)), +k (absolute, exp(-i omega t)) or far less than k (retarded); see
    :func:`acoustic_outputs.preprocess._common.phase_convention`. None if the nodes are too far
    apart (more than 0.45 wavelengths) to tell.
    """
    from scipy.spatial import cKDTree

    k = 2 * np.pi * frequency / sound_speed
    _, nb = cKDTree(points).query(points, k=min(7, len(points)))
    i = np.repeat(np.arange(len(points)), nb.shape[1] - 1)
    j = nb[:, 1:].ravel()
    d = points[j] - points[i]
    ok = np.linalg.norm(d, axis=1) * k < 0.9 * np.pi
    if ok.sum() < 3 * points.shape[1]:
        return None
    c = fundamental[i[ok]].conj() * fundamental[j[ok]]
    w = np.abs(c)
    if not w.sum() > 0:
        return None
    dw = d[ok] * np.sqrt(w)[:, None]
    g, *_ = np.linalg.lstsq(dw, np.angle(c) * np.sqrt(w), rcond=None)
    rate = float(g[axis])
    if abs(rate) < k / 2:
        return "retarded", 0
    return "absolute", -1 if rate > 0 else 1


# -- inside a DOLFINx program --------------------------------------------------------------------------


class DolfinxProbe:  # pragma: no cover - needs DOLFINx
    """Values of DOLFINx functions at fixed points, on every rank (exact for any degree).

    The points are located in the mesh once; each call evaluates the functions there
    (``Function.eval``) on the ranks whose cells hold them, and shares the values. Points outside
    the mesh are zero. In a time-domain run, :meth:`record` keeps the values of each step and
    :meth:`series` returns them as time series::

        probe = DolfinxProbe(mesh, points)       # e.g. axisymmetric_time_domain.rz_points(r, z)
        for step in range(n_steps):
            ...                                  # advance u to t
            probe.record(u)
        p = probe.series()                       # (n_points, n_steps), on every rank

    Args:
        mesh: the DOLFINx mesh.
        points: (n, 3) (or (n, 2), padded with zeros) coordinates [m].
    """

    def __init__(self, mesh: Any, points: Any) -> None:
        import dolfinx.geometry as geometry
        from mpi4py import MPI

        pts = np.asarray(points, dtype=float)
        if pts.shape[1] < 3:
            pts = np.column_stack([pts, np.zeros((pts.shape[0], 3 - pts.shape[1]))])
        self.mesh = mesh
        self.points = np.ascontiguousarray(pts, dtype=mesh.geometry.x.dtype)
        tdim = mesh.topology.dim
        tree = geometry.bb_tree(mesh, tdim) if hasattr(geometry, "bb_tree") else geometry.BoundingBoxTree(mesh, tdim)
        cand = geometry.compute_collisions_points(tree, self.points)
        coll = geometry.compute_colliding_cells(mesh, cand, self.points)
        offsets = np.asarray(coll.offsets)
        self.found = np.diff(offsets) > 0
        self.cells = np.full(self.points.shape[0], -1, dtype=np.int32)
        self.cells[self.found] = np.asarray(coll.array)[offsets[:-1][self.found]]
        self._sum = MPI.SUM
        self.count = mesh.comm.allreduce(self.found.astype(float), op=self._sum)
        self._steps: list[Any] = []

    def local(self, u: Any) -> Any:
        """The values of ``u`` at the points this rank holds (0 elsewhere): (n,)."""
        out = np.zeros(self.points.shape[0], dtype=u.x.array.dtype)
        if self.found.any():
            out[self.found] = np.asarray(u.eval(self.points[self.found], self.cells[self.found])).reshape(-1)
        return out

    def _share(self, local: Any) -> Any:
        out = self.mesh.comm.allreduce(local, op=self._sum)
        held = self.count > 0
        out[held] = out[held] / self.count[held].reshape((-1,) + (1,) * (out.ndim - 1))
        return out

    def __call__(self, functions: Any) -> ComplexArray:
        """Values of one function (n,), or of a list of them (n, K), on every rank."""
        us = list(functions) if isinstance(functions, (list, tuple)) else [functions]
        out = self._share(np.stack([self.local(u).astype(complex) for u in us], axis=1))
        return out if isinstance(functions, (list, tuple)) else out[:, 0]

    def record(self, u: Any) -> None:
        """Keep the values of ``u`` at this step (shared between the ranks by :meth:`series`)."""
        self._steps.append(self.local(u))

    def series(self) -> Any:
        """The recorded steps as time series (n, n_steps), on every rank."""
        if not self._steps:
            raise ValueError("no step was recorded")
        return self._share(np.stack(self._steps, axis=1)).real


def evaluate_dolfinx(functions: Any, points: Any) -> ComplexArray:  # pragma: no cover - needs DOLFINx
    """Evaluate DOLFINx functions at ``points`` (n, 3): (n, K), on every rank; 0 outside the mesh."""
    us = list(functions) if isinstance(functions, (list, tuple)) else [functions]
    return DolfinxProbe(us[0].function_space.mesh, points)(us)


# -- XDMF files (DOLFINx's XDMFFile) -----------------------------------------------------------------


@dataclass
class Function:
    """A scalar function in an XDMF file: its times and, for each, where its values are.

    The values are read only when :meth:`load` asks for them, so that a time-domain record of
    many steps is not held whole.
    """

    times: RealArray
    items: list[tuple[ET.Element, ET.Element | None]]  # (real or whole, imaginary)
    base: Path
    on_cells: bool = False

    @property
    def is_complex(self) -> bool:
        return self.items[0][1] is not None or _complex_type(self.items[0][0])

    def load(self, steps: Any = None) -> Any:
        """The values (n_steps, n) of the steps (indices; default all)."""
        idx = range(len(self.items)) if steps is None else np.atleast_1d(steps)
        files: dict[str, Any] = {}
        try:
            rows = []
            for i in idx:
                re, im = self.items[int(i)]
                v = _scalar(_data_item(re, self.base, files))
                if im is not None:
                    v = v + 1j * _scalar(_data_item(im, self.base, files))
                rows.append(v)
        finally:
            for f in files.values():
                f.close()
        return np.stack(rows)


def _complex_type(item: ET.Element) -> bool:
    return item.get("NumberType", item.get("DataType", "")).lower() == "complex"


def _scalar(v: Any) -> Any:
    return v[..., 0] if v.ndim > 1 and v.shape[-1] == 1 else v


@dataclass
class MeshData:
    """A mesh and the functions on it, as read from an XDMF file.

    Attributes:
        points: node coordinates (n, gdim).
        cells: cell nodes (m, nodes per cell), or None.
        cell_type: the XDMF topology type (``Triangle``, ``Tetrahedron``, ``Hexahedron_27``, ...).
        functions: each scalar function (see :class:`Function`), by name.
    """

    points: RealArray
    cells: IntArray | None
    cell_type: str
    functions: dict[str, Function] = field(default_factory=dict)

    def centroids(self) -> RealArray:
        """Centroids of the cells (of their corner nodes)."""
        if self.cells is None:
            raise ValueError("the file has no cells")
        return self.points[corners(self.cells, self.cell_type)].mean(axis=1)

    @property
    def planar(self) -> bool:
        """Whether the mesh lies in the plane x[2] = 0 (a 2D mesh)."""
        return self.points.shape[1] == 2 or bool(np.all(self.points[:, 2] == 0))

    def where(self, name: str) -> tuple[RealArray, IntArray | None]:
        """The points of the values of a function and the cells to interpolate them over."""
        if self.functions[name].on_cells:
            return self.centroids(), None
        return self.points, self.cells


def _data_item(item: ET.Element, base: Path, files: dict[str, Any] | None = None) -> Any:
    fmt = item.get("Format", "XML").upper()
    dims = [int(d) for d in item.get("Dimensions", "").split()]
    text = (item.text or "").strip()
    if fmt == "HDF":
        try:
            import h5py
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise ImportError("reading XDMF needs h5py: pip install 'acoustic-outputs[hdf5]'") from err
        fname, _, dset = text.partition(":")
        if files is None:
            with h5py.File(base / fname, "r") as f:
                arr = np.asarray(f[dset][()])
        else:
            if fname not in files:
                files[fname] = h5py.File(base / fname, "r")
            arr = np.asarray(files[fname][dset][()])
    elif fmt == "XML":
        kind = item.get("NumberType", item.get("DataType", "Float")).lower()
        arr = np.array(text.split(), dtype=int if kind in ("int", "uint") else float)
    else:
        raise ValueError(f"XDMF data of format {fmt!r} is not read")
    return arr.reshape(dims) if dims and int(np.prod(dims)) == arr.size else arr


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_xdmf(path: str | Path) -> MeshData:
    """Read the mesh of a (DOLFINx) XDMF file, and where the values of its scalar functions are.

    DOLFINx's ``XDMFFile.write_mesh`` and ``write_function(u, t)`` write the geometry, the
    topology and, for each time, the values at the nodes (the function must have the degree of
    the mesh geometry, e.g. P1 on a linear mesh). A complex function is written as two real
    attributes ``real_<name>`` and ``imag_<name>``, joined again here. A frequency-domain run
    writes its harmonics as complex functions (one or more); a time-domain run writes a real
    function at every saved step. The values are read when they are used.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    base = path.parent
    elements = list(root.iter())
    geom = next((e for e in elements if _local(e.tag) == "Geometry"), None)
    topo = next((e for e in elements if _local(e.tag) == "Topology"), None)
    if geom is None:
        raise ValueError(f"{path} has no Geometry")
    gitem = next(e for e in geom.iter() if _local(e.tag) == "DataItem")
    points = np.asarray(_data_item(gitem, base), dtype=float)
    gtype = geom.get("GeometryType", "XYZ").upper()
    points = points.reshape(-1, len(gtype) if gtype in ("XY", "XYZ") else points.shape[-1])
    cells, cell_type = None, "Triangle"
    if topo is not None:
        cell_type = topo.get("TopologyType", topo.get("Type", "Triangle"))
        titem = next((e for e in topo.iter() if _local(e.tag) == "DataItem"), None)
        if titem is not None:
            cells = np.asarray(_data_item(titem, base), dtype=np.int64)
            per = topo.get("NodesPerElement")
            if per is not None:
                cells = cells.reshape(-1, int(per))
    found: dict[str, list[tuple[float, ET.Element]]] = {}
    on_cells: set[str] = set()
    for grid in (e for e in elements if _local(e.tag) == "Grid"):
        time_el = next((c for c in grid if _local(c.tag) == "Time"), None)
        t = float(time_el.get("Value", "nan")) if time_el is not None else np.nan
        for att in (c for c in grid if _local(c.tag) == "Attribute"):
            if att.get("AttributeType", "Scalar").lower() != "scalar":
                log.debug("XDMF attribute %s is not scalar: skipped", att.get("Name"))
                continue
            item = next(e for e in att.iter() if _local(e.tag) == "DataItem")
            name = att.get("Name", "u")
            found.setdefault(name, []).append((t, item))
            if att.get("Center", "Node").lower() == "cell":
                on_cells.add(name)
    for items in found.values():
        items.sort(key=lambda ti: (np.isnan(ti[0]), ti[0]))
    functions: dict[str, Function] = {}
    for name, items in found.items():
        if name.startswith("imag_") and "real_" + name.removeprefix("imag_") in found:
            continue
        times = np.array([t for t, _ in items])
        if name.startswith("real_") and "imag_" + name.removeprefix("real_") in found:
            imag = found["imag_" + name.removeprefix("real_")]
            functions[name.removeprefix("real_")] = Function(
                times, [(re, im) for (_, re), (_, im) in zip(items, imag, strict=True)], base, name in on_cells
            )
        else:
            functions[name] = Function(times, [(it, None) for _, it in items], base, name in on_cells)
    return MeshData(points=points, cells=cells, cell_type=cell_type, functions=functions)


def _names(mesh: MeshData, fields: Any) -> list[str]:
    if fields is None:
        if len(mesh.functions) != 1:
            raise ValueError(f"the file holds the functions {sorted(mesh.functions)}: name the field(s)")
        return list(mesh.functions)
    names = [fields] if isinstance(fields, str) else [str(f) for f in fields]
    missing = [n for n in names if n not in mesh.functions]
    if missing:
        raise KeyError(f"no function {missing} in the file (it has {sorted(mesh.functions)})")
    return names


def is_time_series(mesh: MeshData, fields: Any) -> bool:
    """Whether ``fields`` is one real function written at several times (a time-domain record)."""
    names = _names(mesh, fields)
    f = mesh.functions[names[0]]
    return len(names) == 1 and not f.is_complex and f.times.size > 1


def xdmf_harmonics(mesh: MeshData, fields: Any, time: float | None) -> tuple[RealArray, IntArray | None, Any]:
    """Points, cells and complex values (n[, K]) of frequency-domain functions in a file.

    One complex function is one frequency; several are the harmonics 1, 2, ... in order. Of a
    function written at several times, the one nearest ``time`` is used (default the last).
    """
    names = _names(mesh, fields)
    if len({mesh.functions[n].on_cells for n in names}) > 1:
        raise ValueError("the fields mix node and cell values")
    cols = []
    for n in names:
        f = mesh.functions[n]
        if not f.is_complex:
            log.warning("function %s is real: taken as a complex amplitude of zero phase", n)
        i = -1 if time is None else int(np.argmin(np.abs(f.times - time)))
        cols.append(np.asarray(f.load(i % f.times.size)[0], dtype=complex))
    points, cells = mesh.where(names[0])
    return points, cells, cols[0] if len(cols) == 1 else np.stack(cols, axis=1)


@dataclass
class TimeRecord:
    """A time-domain record read from a file: the window of steps to be fitted."""

    points: RealArray
    cells: IntArray | None
    series: RealArray  # (n, n_steps)
    dt: float
    start_time: float


def xdmf_time_series(
    mesh: MeshData, field: str | None, frequency: float, periods: float | None, window_start: float | None
) -> TimeRecord:
    """The steps of a real function that the harmonic fit uses: the last ``periods`` periods
    (or those from ``window_start``), read alone so that the rest of the record is not loaded."""
    names = _names(mesh, field)
    if len(names) != 1:
        raise ValueError(f"a time series is one function, not {names}")
    name = names[0]
    f = mesh.functions[name]
    t = f.times
    if t.size < 3 or np.any(np.isnan(t)):
        raise ValueError(f"the function {name!r} is not written at many times: not a time series")
    if f.is_complex:
        raise ValueError(f"the function {name!r} is complex: read it as harmonics")
    dts = np.diff(t)
    if not np.allclose(dts, dts[0], rtol=1e-6):
        raise ValueError("the times of the series are not uniformly spaced")
    dt = float(dts[0])
    if window_start is not None:
        first = int(np.searchsorted(t, window_start - 1e-9 * dt))
    else:
        assert periods is not None
        first = max(0, t.size - round(periods / (frequency * dt)))
    log.info("reading steps %d .. %d of %d of %s", first, t.size - 1, t.size, name)
    points, cells = mesh.where(name)
    return TimeRecord(points, cells, f.load(range(first, t.size)).T.astype(float), dt, float(t[first]))
