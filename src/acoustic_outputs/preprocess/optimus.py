"""Output of OptimUS (a boundary element library for 3D acoustics, on BEMPP).

OptimUS solves the Helmholtz equation, at one frequency, in an unbounded medium with
homogeneous scatterers (skull, ribs, tissue layers). Its sources (``create_piston``,
``create_bowl``, ``create_array``) radiate by the Rayleigh integral into the exterior medium, and
the boundary element solution adds the scattered field. The fields are evaluated at chosen
points by a postprocess object:

* ``VisualisePlane``: ``create_computational_grid(resolution, plane_axes, plane_offset,
  bounding_box)`` makes a regular grid on one plane (``points`` (3, N), ``numpy.mgrid`` order);
* ``VisualiseCloudPoints``: ``create_computational_grid(points)`` takes any points (3, N);

and ``compute_fields()`` fills ``field.total_field``, ``field.incident_field`` and
``field.scattered_field`` (N,). ``optimus.postprocess.export_to_file(model, post_process, ...,
file_format="mat")`` saves those and the ``points`` to a ``.mat`` file (without the frequency or
the medium).

The fields are frequency domain: p(x, t) = Re(u(x) exp(-i omega t)) (BEMPP's convention), in
absolute phase, in Pa for the sources' surface velocities. Only the fundamental is computed, so
p+ = |p-| = |P| and there are no nonlinear parameters. A pulse is simulated in OptimUS by
running one frequency at a time; give each frequency to :func:`from_optimus` separately.

:func:`from_optimus` takes the points and one of the fields (``which``, default the total
field), from postprocess objects, from a ``.mat`` export or as arrays. It finds the regular grid
the points lie on, in the frame of the beam: the beam axis is a coordinate axis (``propagation``
or the source's ``source_axis``, either sign; OptimUS's default is +x), through ``axis_point``
(the source's ``location``, default the origin). Then:

* a 3D grid (several planes across the beam, e.g. ``VisualisePlane`` objects at several
  ``plane_offset``, or a cloud of points on a grid) is recorded as a 3D field;
* a single plane through the beam axis (``VisualisePlane`` with the axis in ``plane_axes``) is,
  by default (``geometry="auto"``), taken as axisymmetric: its half from the axis outwards is
  revolved onto (x, y) (:mod:`~acoustic_outputs.preprocess.axisymmetric_frequency_domain`). The
  two halves are compared, and a warning is given if the field is not symmetric (an array
  steered off axis, or an asymmetric scatterer). ``geometry="planar"`` reads the plane as a 2D
  field instead (per metre in the third direction);
* a plane across the beam alone is not a beam: an error.

The planes are taken from ``z_min`` along the beam (default the source's location, when it is
known: points behind the source are left out). With a ``model``, the frequency, the medium
(``material_exterior``), the source axis and location and, for a bowl, the focusing depth
(``radius_of_curvature``) are read from it.

Example, in an OptimUS session::

    planes = []
    for x in np.linspace(5e-3, 80e-3, 151):            # planes across a beam along +x
        pp = optimus.postprocess.VisualisePlane(model)
        pp.create_computational_grid(resolution=(81, 81), plane_axes=(1, 2), plane_offset=x,
                                     bounding_box=[-10e-3, 10e-3, -10e-3, 10e-3])
        pp.compute_fields()
        planes.append(pp)
    res = from_optimus(planes, model=model)

    pp = optimus.postprocess.VisualisePlane(model)       # or one plane through the axis
    pp.create_computational_grid(resolution=(301, 81), plane_axes=(0, 1),
                                 bounding_box=[5e-3, 80e-3, -10e-3, 10e-3])
    pp.compute_fields()
    res = from_optimus(pp, model=model)                  # revolved, if the source is axisymmetric
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from acoustic_outputs._types import ComplexArray, IntArray, RealArray
from acoustic_outputs.preprocess.axisymmetric_frequency_domain import from_axisymmetric_harmonics
from acoustic_outputs.preprocess.frequency_domain import from_harmonics
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results

log = logging.getLogger(__name__)

Which = Literal["total", "incident", "scattered"]
Geometry = Literal["auto", "3d", "axisymmetric", "planar"]

#: relative asymmetry of a plane through the axis above which it is not taken as axisymmetric
SYMMETRY_TOLERANCE = 1e-2


def _attr(obj: Any, *names: str) -> Any:
    """The first attribute (or key) ``names`` of ``obj`` that is set, following dots."""
    for name in names:
        cur = obj
        for part in name.split("."):
            cur = cur.get(part) if isinstance(cur, Mapping) else getattr(cur, part, None)
            if cur is None:
                break
        if cur is not None:
            return cur
    return None


def postprocess_fields(postprocess: Any, which: Which = "total") -> tuple[RealArray, ComplexArray]:
    """The points (3, N) and the field (N,) of OptimUS postprocess objects (one, or a list)."""
    objs = list(postprocess) if isinstance(postprocess, (list, tuple)) else [postprocess]
    pts, vals = [], []
    for o in objs:
        p = _attr(o, "points")
        v = _attr(o, f"field.{which}_field", f"{which}_field")
        if p is None or v is None:
            raise ValueError(f"the postprocess object has no points or {which} field: run compute_fields() first")
        p = np.asarray(p, dtype=float)
        pts.append(p if p.shape[0] == 3 else p.T)
        vals.append(np.asarray(v, dtype=complex).ravel())
    return np.concatenate(pts, axis=1), np.concatenate(vals)


def _levels(v: RealArray) -> tuple[RealArray, IntArray]:
    """The distinct values of ``v`` (within round-off) and the index of each entry among them."""
    order = np.argsort(v)
    s = v[order]
    tol = 1e-9 * (float(np.ptp(v)) + 1e-3)
    new = np.concatenate([[True], np.diff(s) > tol])
    group = np.cumsum(new) - 1
    levels = np.array([s[group == g].mean() for g in range(group[-1] + 1)])
    idx = np.empty(v.size, dtype=np.int64)
    idx[order] = group
    return levels, idx


def beam_frame(axis: Any) -> tuple[int, int]:
    """The coordinate (0, 1, 2) along the beam and its sign, from a name (``"-x"``) or a vector."""
    if isinstance(axis, str):
        name = axis.strip().lower()
        sign = -1 if name.startswith("-") else 1
        letter = name.lstrip("+-")
        if letter not in ("x", "y", "z"):
            raise ValueError(f"propagation {axis!r}: use x, y or z, with a sign")
        return "xyz".index(letter), sign
    a = np.asarray(axis, dtype=float).ravel()
    if a.size != 3 or not np.linalg.norm(a) > 0:
        raise ValueError(f"the source axis {axis!r} is not a 3-vector")
    a = a / np.linalg.norm(a)
    i = int(np.argmax(np.abs(a)))
    if abs(abs(a[i]) - 1) > 1e-6:
        raise ValueError(
            f"the source axis {tuple(a)} is not along x, y or z: evaluate the field on a grid aligned with the beam"
        )
    return i, int(np.sign(a[i]))


def gridded(points: Any, values: Any, axis: int, sign: int, origin: RealArray) -> tuple[list[RealArray], ComplexArray]:
    """The field on its regular grid in the beam frame: vectors (t1, t2, z) and values (nt1, nt2, nz).

    z = sign (x_axis - origin_axis) runs along the beam; t1, t2 are the other two coordinates,
    in order, from the origin.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or 3 not in pts.shape:
        raise ValueError(f"points must be (3, N) or (N, 3), not {pts.shape}")
    if pts.shape[0] != 3:
        pts = pts.T
    v = np.asarray(values).ravel()
    if v.size != pts.shape[1]:
        raise ValueError(f"{v.size} field values for {pts.shape[1]} points")
    rel = pts - origin[:, None]
    transverse = [i for i in range(3) if i != axis]
    coords = [rel[transverse[0]], rel[transverse[1]], sign * rel[axis]]
    vecs, idx = zip(*(_levels(c) for c in coords), strict=True)
    shape = tuple(len(x) for x in vecs)
    grid = np.full(shape, np.nan + 0j)
    flat = np.ravel_multi_index(idx, shape)
    if np.unique(flat).size != flat.size:
        raise ValueError("some points appear twice")
    grid.ravel()[flat] = v
    if np.isnan(grid).any():
        raise ValueError(
            f"the points do not fill a regular grid ({shape[0]} x {shape[1]} x {shape[2]} values in the beam frame, "
            f"{v.size} points): evaluate the field on a grid aligned with the beam"
        )
    return list(vecs), grid


def from_optimus(
    postprocess: Any = None,
    *,
    points: Any = None,
    field: Any = None,
    which: Which = "total",
    model: Any = None,
    frequency: float | None = None,
    sound_speed: float | None = None,
    density: float | None = None,
    propagation: Any = None,
    axis_point: Any = None,
    z_min: float | None = None,
    geometry: Geometry = "auto",
    phase: Literal["retarded", "absolute"] = "absolute",
    time_sign: int = -1,
    dx: float | None = None,
    r_max: float | None = None,
    settings: RecordSettings | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float | None = None,
) -> Results:
    """:class:`Results` of an OptimUS field on a regular grid (planes across the beam, or a plane through it).

    Args:
        postprocess: OptimUS postprocess object(s) after ``compute_fields()``
            (``VisualisePlane``, ``VisualiseCloudPoints``); or None, with ``points`` and ``field``.
        points, field: the evaluation points (3, N) and the complex field (N,) (as
            ``export_to_file(..., file_format="mat")`` saves them).
        which: the field of a postprocess object: ``"total"``, ``"incident"`` or ``"scattered"``.
        model: the OptimUS model; its frequency, exterior material, source axis and location
            (and a bowl's radius of curvature) are used when not given.
        frequency, sound_speed, density: f0 [Hz], c0 [m/s], rho0 [kg/m^3] of the exterior medium.
        propagation: the beam axis: ``"x"``, ``"-z"``, ... or a vector along a coordinate axis
            (default the source's ``source_axis``, or +x, OptimUS's default).
        axis_point: a point (3,) on the beam axis [m] (default the source's ``location``, or 0).
        z_min: the first plane, as a distance along the beam from ``axis_point`` [m] (default 0
            when the source is known: nothing behind it; else every plane).
        geometry: ``"auto"`` (a 3D grid as 3D; a plane through the axis as axisymmetric),
            ``"3d"``, ``"axisymmetric"`` or ``"planar"`` (a plane through the axis as a 2D field).
        phase, time_sign: the convention (OptimUS: absolute, exp(-i omega t)).
        dx, r_max: the revolved planes of an axisymmetric field (see
            :func:`~acoustic_outputs.preprocess.axisymmetric_frequency_domain.from_axisymmetric_harmonics`).
        settings, alpha: as in :func:`~acoustic_outputs.preprocess.frequency_domain.from_harmonics`.
        focusing_depth: the geometric focus from ``axis_point`` [m] (default a bowl's radius of
            curvature, else 0).
    """
    if postprocess is not None:
        points, field = postprocess_fields(postprocess, which)
    elif points is None or field is None:
        raise ValueError("give OptimUS postprocess objects, or the points and the field")
    source = _attr(model, "source") if model is not None else None
    frequency = frequency if frequency is not None else _attr(model, "frequency", "source.frequency")
    sound_speed = sound_speed if sound_speed is not None else _attr(model, "material_exterior.speed_of_sound")
    density = density if density is not None else _attr(model, "material_exterior.density")
    missing = [
        n for n, v in (("frequency", frequency), ("sound_speed", sound_speed), ("density", density)) if v is None
    ]
    if missing:
        raise ValueError(f"no {', '.join(missing)}: give them, or the OptimUS model")
    if propagation is None:
        propagation = _attr(source, "source_axis") if source is not None else None
    axis, sign = beam_frame(propagation if propagation is not None else "x")
    if axis_point is None and source is not None:
        axis_point = _attr(source, "location")
    origin = np.zeros(3) if axis_point is None else np.asarray(axis_point, dtype=float).ravel()
    if z_min is None and (source is not None or axis_point is not None):
        z_min = 0.0
    if focusing_depth is None:
        roc = _attr(source, "radius_of_curvature") if source is not None else None
        focusing_depth = float(roc) if roc is not None else 0.0
    if time_sign not in (1, -1):
        raise ValueError("time_sign must be +1 or -1")

    (t1, t2, z), grid = gridded(points, field, axis, sign, origin)
    if z_min is not None:
        keep = z >= z_min - 1e-9 * (np.ptp(z) + 1e-3)
        if not keep.any():
            raise ValueError(f"no plane at or beyond z_min = {z_min} m along the beam")
        if not keep.all():
            log.info("OptimUS field: %d planes behind z_min = %.4g m left out", int((~keep).sum()), z_min)
        z, grid = z[keep], grid[..., keep]
    names = "xyz"
    t_names = [names[i] for i in range(3) if i != axis]
    log.info("OptimUS field: beam along %s%s, %d x %d x %d points (%s, %s, beam)",
             "-" if sign < 0 else "+", names[axis], *grid.shape, *t_names)  # fmt: skip
    if z.size < 2:
        raise ValueError("the points hold one plane across the beam: evaluate several planes along it")
    medium = {"frequency": float(frequency), "sound_speed": float(sound_speed), "density": float(density)}
    # Results measure the focal shift from the first plane; the focusing depth is from the source
    depth = float(focusing_depth) - float(z[0]) if focusing_depth else 0.0
    if focusing_depth and depth <= 0:
        log.warning("OptimUS field: the first plane is beyond the geometric focus: no focal shift")
        depth = 0.0
    common = {"settings": settings, "alpha": alpha, "focusing_depth": depth}
    flat = [i for i, n in enumerate(grid.shape[:2]) if n == 1]
    if not flat:
        if geometry in ("axisymmetric", "planar"):
            raise ValueError(f"the field is on a 3D grid, not on a plane: geometry {geometry!r} does not apply")
        return from_harmonics(np.transpose(grid, (2, 0, 1)), layout="zxy", x=t1, y=t2, z=z, phase=phase,
                              time_sign=time_sign, axis_xy=(0.0, 0.0), **medium, **common)  # fmt: skip
    if len(flat) == 2:
        raise ValueError("the points lie on a line along the beam: evaluate a plane through it, or planes across it")
    if geometry == "3d":
        raise ValueError("the field is on one plane through the beam, not a 3D grid")
    across, off = (t2, t1) if flat[0] == 0 else (t1, t2)
    plane = grid[0] if flat[0] == 0 else grid[:, 0]  # (n_across, nz)
    if geometry == "planar":
        return from_harmonics(plane.T, layout="zx", x=across, z=z, phase=phase, time_sign=time_sign,
                              axis_xy=(0.0, 0.0), **medium, **common)  # fmt: skip
    if abs(off[0]) > 1e-9 * (np.ptp(across) + 1e-3):
        raise ValueError(f"the plane is {off[0]:.4g} m from the beam axis: an axisymmetric field needs the axis")
    return _revolve(plane, across, z, phase, time_sign, dx, r_max, medium, common)


def _revolve(
    plane: ComplexArray, across: RealArray, z: RealArray, phase: str, time_sign: int,
    dx: float | None, r_max: float | None, medium: dict[str, float], common: dict[str, Any],
) -> Results:  # fmt: skip
    """The half of a plane through the axis, revolved; the other half checks the symmetry."""
    tol = 1e-9 * (np.ptp(across) + 1e-3)
    on_axis = np.flatnonzero(np.abs(across) <= tol)
    if on_axis.size == 0:
        raise ValueError("the beam axis is not a point of the plane's grid (use an odd number of points across)")
    i0 = int(on_axis[0])
    pos, neg = across[i0:], -across[:i0][::-1]
    half = plane[i0:]
    if neg.size:
        n = min(neg.size, pos.size - 1)
        if n > 0 and np.allclose(neg[:n], pos[1 : n + 1], rtol=1e-6, atol=tol):
            other = plane[:i0][::-1][:n]
            scale = float(np.abs(half).max()) or 1.0
            asym = float(np.abs(other - half[1 : n + 1]).max()) / scale
            if asym > SYMMETRY_TOLERANCE:
                log.warning("OptimUS field: the two halves of the plane differ by %.2g of the peak: the field is "
                            "not axisymmetric, and the revolved metrics are those of its positive half "
                            "(evaluate planes across the beam for a 3D field)", asym)  # fmt: skip
            else:
                log.info("OptimUS field: the two halves agree to %.2g of the peak: revolved", asym)
    if pos.size < 2:
        raise ValueError("the plane does not extend from the axis outwards")
    return from_axisymmetric_harmonics(
        half, r=pos - pos[0], z=z, layout="rz", phase=phase, time_sign=time_sign,  # type: ignore[arg-type]
        dx=dx, r_max=r_max, **medium, **common,
    )  # fmt: skip


def is_optimus(data: Mapping[str, Any]) -> bool:
    """Whether the keys look like OptimUS's ``export_to_file(..., file_format="mat")``."""
    return "points" in data and any(k in data for k in ("total_field", "L2_norm_ptot_MPa", "gmres_iter_count"))
