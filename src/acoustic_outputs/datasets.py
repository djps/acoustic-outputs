"""Read a solver's output from a file and compute its :class:`~acoustic_outputs.results.Results`.

A dataset is a set of named arrays and scalars, in a ``.npz`` file (``numpy.savez``), a ``.mat``
file (MATLAB or ``scipy.io.savemat``; structs such as ``kgrid`` and ``sensor`` are read field by
field) or an HDF5 file (``.h5``/``.hdf5``, needs ``h5py``; datasets and root attributes). Its
``kind`` (stored in the file as ``kind``, or given) names one of the solver families:

=================  ==================================================================================
kind               keys (optional ones in brackets)
=================  ==================================================================================
``harmonics``      ``harmonics`` (complex), ``x``, ``z``, [``y``, ``layout`` = "zkxy",
                   ``phase`` = "retarded" | "absolute", ``time_sign`` = +1 | -1]
``time-series``    ``p``, ``dt``, the coordinates ``x``, ``y``, ``z`` named in ``layout``
                   [= "xyzt"], [``propagation`` = "z", ``start_time``, ``detrend`` = false]
``kwave``          ``p`` (n_sensor, Nt) (or ``sensor_data`` with a field ``p``), ``dt``,
                   ``x_vec`` [, ``y_vec``, ``z_vec``] (or a ``kgrid`` struct holding them),
                   [``sensor_mask`` (or ``mask``), ``record_start_index`` = 1, ``propagation`` =
                   "x", ``order`` = "C" (use "F" for MATLAB k-Wave)]
``fdtd``           as ``time-series``, with ``layout`` = "xyt", ``propagation`` = "x" and
                   ``detrend`` = true
``fullwave``       ``sensor_data`` (n_sensors, nt), ``outcoords`` (n_sensors, ndim), ``dt``,
                   ``dx`` [, ``dy``, ``dz``], [``sampling_modulus_time`` = 1, ``origin``,
                   ``propagation`` = "x", ``detrend`` = true, ``f0``, ``c0``]
``babelbrain``     BabelBrain's ``DataForSim.h5`` as it is: ``p_complex``, ``x_vec``, ``y_vec``,
                   ``z_vec``, ``Material``, ``MaterialMap``, ``TargetLocation`` [``field`` =
                   "p_complex", ``direction`` = "auto"]; the frequency comes from the file name,
                   rho0 and c0 from the material at the target (see
                   :mod:`acoustic_outputs.preprocess.babelbrain`). Recognised without ``kind``.
``msound``         mSOUND's frequency-specific ``P_fundamental`` [, ``P_second``] (nx, [ny,] planes),
                   or its time-domain ``p_time`` (num_t, n_sensor) with ``sensor_mask``; the grid
                   as ``struct(mgrid)`` (or ``dx``, [``dy``,] ``dz``, ``x``, [``y``], ``dt``;
                   in 2D the plane spacing is ``dy``), [``z0`` = 0, ``fc`` or ``omega_c``]; see
                   :mod:`acoustic_outputs.preprocess.msound`. Recognised without ``kind``.
``focus``          FOCUS's ``pressure`` (complex (nx, ny, nz) CW or (nx, ny, nz, n) KZK; real
                   (nx, ny, nz, nt) transient) with ``x``, ``y``, ``z`` (or ``delta`` and
                   ``xmin`` .. ``zmax``), [``phase`` = "auto", ``time_sign``, ``scale`` = 1;
                   transient: ``deltat`` (or ``dt``), ``tmin``]; see
                   :mod:`acoustic_outputs.preprocess.focus`.
``axisymmetric``   a field p(r, z), revolved onto (x, y) planes (see
                   :mod:`acoustic_outputs.preprocess.axisymmetric_frequency_domain` and
                   ``axisymmetric_time_domain``): ``harmonics`` (complex) with
                   ``r``, ``z`` [``layout`` = "zkr"]; or ``p`` (real) with ``r``, ``z``, ``dt``
                   [``layout`` = "rzt", ``start_time``]; or ``points`` (n, 2) and ``values``
                   (complex (n[, K]), or real (n, nt) with ``dt``) [``cells``, ``cell_type``,
                   ``spacing``]; or an ``.xdmf`` file written by DOLFINx [``field``, a name or a
                   list of them]. All take [``phase`` = "auto", ``time_sign``, ``coordinates`` =
                   "rz", ``dx``, ``r_max``]. A 2D ``.xdmf`` file is taken as axisymmetric.
``optimus``        OptimUS's ``export_to_file(..., file_format="mat")``: ``points`` (3, N) and
                   ``total_field`` (or ``incident_field``, ``scattered_field``: ``field`` =
                   "total"), the frequency and the medium (not in the file); [``propagation`` =
                   "x" (or ``source_axis``), ``axis_point`` (or ``location``), ``z_min``,
                   ``geometry`` = "auto", ``dx``, ``r_max``]; see
                   :mod:`acoustic_outputs.preprocess.optimus`. Recognised without ``kind``.
``unstructured``   a field on a 3D (or planar 2D) mesh, resampled on planes (see
                   :mod:`acoustic_outputs.preprocess.unstructured_frequency_domain` and
                   ``unstructured_time_domain``): ``points`` (n, 3) and
                   ``values`` (complex (n[, K]), or real (n, nt) with ``dt``) [``cells``,
                   ``cell_type``, ``max_edge``]; or an ``.xdmf`` file [``field``, ``time``];
                   [``propagation`` (z in 3D, x in 2D), ``x``, ``y``, ``z``, ``spacing``,
                   ``phase`` = "auto", ``time_sign``]. A 3D ``.xdmf`` file is recognised
                   without ``kind``.
=================  ==================================================================================

Every kind takes the medium and analysis keys ``frequency`` (or ``f0``), ``sound_speed`` (or
``c0``), ``density`` (or ``rho0``), and optionally ``n_harmonics`` (time series, default 4),
``periods`` (2), ``window_start``, ``alpha`` (Np/m per harmonic), ``focusing_depth``,
``axis_xy``, and the :class:`~acoustic_outputs.recorder.RecordSettings` fields
``focal_quantity``, ``levels_db``, ``plane_peaks``, ``metrics_harmonics``,
``hydrophone_diameter``, ``derating`` and ``face_distance``. Keyword arguments of
:func:`from_dataset` override the file.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from acoustic_outputs.preprocess import (
    axisymmetric_frequency_domain,
    axisymmetric_time_domain,
    babelbrain,
    fdtd,
    focus,
    frequency_domain,
    msound,
    optimus,
    time_domain,
    unstructured_frequency_domain,
    unstructured_time_domain,
)
from acoustic_outputs.preprocess._mesh import is_time_series, read_xdmf
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results

log = logging.getLogger(__name__)

KINDS = (
    "harmonics",
    "time-series",
    "kwave",
    "fdtd",
    "fullwave",
    "babelbrain",
    "msound",
    "focus",
    "axisymmetric",
    "unstructured",
    "optimus",
)

#: names under which a key may also be found (the first is the one used)
ALIASES = {
    "frequency": ("frequency", "f0", "fc"),
    "sound_speed": ("sound_speed", "c0", "soundspeed"),
    "density": ("density", "rho0", "rho"),
    "sensor_mask": ("sensor_mask", "mask"),
}

#: structs whose fields are read as top-level keys (k-Wave and fullwave25 objects saved whole)
STRUCTS = (
    "kgrid",
    "grid",
    "sensor",
    "medium",
    "source_info",
    "settings",
    "mgrid",
    "mgrid_s",
    "coordinate_grid",
    "time_struct",
    "time_samples",
)

_SETTINGS = {f.name for f in fields(RecordSettings)} - {"points", "workers"}


# -- reading ---------------------------------------------------------------------------------


def _fields(obj: Any) -> dict[str, Any]:
    """The fields of a struct read from a file (dict, MATLAB struct or HDF5 group)."""
    if isinstance(obj, Mapping):
        return dict(obj)
    names = getattr(obj, "_fieldnames", None)  # scipy.io mat_struct
    if names is not None:
        return {n: getattr(obj, n) for n in names}
    if isinstance(obj, np.ndarray) and obj.dtype.names:  # a struct read as a record array
        rec = obj.reshape(-1)[0]
        return {n: rec[n] for n in obj.dtype.names}
    return {}


def _flatten(data: Mapping[str, Any]) -> dict[str, Any]:
    """Lift the fields of known structs to the top level, without overriding top-level keys."""
    out = dict(data)
    for name in STRUCTS:
        if name in data:
            sub = _fields(data[name])
            if sub:
                out.pop(name)
                for k, v in sub.items():
                    out.setdefault(k, v)
    if "sensor_data" in out:
        sd = _fields(out["sensor_data"])
        if "p" in sd:
            out.setdefault("p", sd["p"])
    return out


def _read_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("reading HDF5 needs h5py: pip install 'acoustic-outputs[hdf5]'") from err
    try:  # registers the blosc and other filters (BabelBrain's files are blosc-compressed)
        import hdf5plugin  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the environment
        pass

    def group(g: Any) -> dict[str, Any]:
        out: dict[str, Any] = {k: v for k, v in g.attrs.items()}
        for k, v in g.items():
            out[k] = group(v) if isinstance(v, h5py.Group) else v[()]
        return out

    with h5py.File(path, "r") as f:
        return group(f)


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Read the arrays and scalars of a dataset file (``.npz``, ``.mat``, ``.h5``/``.hdf5``).

    A DOLFINx ``.xdmf`` file is read as the mesh and functions under the key ``xdmf``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        with np.load(p, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
    elif suffix == ".mat":
        from scipy.io import loadmat

        raw = loadmat(p, squeeze_me=True, struct_as_record=False)
        data = {k: v for k, v in raw.items() if not k.startswith("__")}
    elif suffix in (".h5", ".hdf5"):
        data = _read_hdf5(p)
    elif suffix == ".xdmf":
        mesh = read_xdmf(p)
        return {"kind": "axisymmetric" if mesh.planar else "unstructured", "xdmf": mesh}
    else:
        raise ValueError(f"unknown dataset format {p.suffix!r}: use .npz, .mat, .h5, .hdf5 or .xdmf")
    return _flatten(data)


# -- values ------------------------------------------------------------------------------------


def _value(v: Any) -> Any:
    """A scalar or string read from a file as a Python value; arrays are returned unchanged."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "SU" and v.size == 1:
            return str(v.reshape(-1)[0].decode() if v.dtype.kind == "S" else v.reshape(-1)[0])
        if v.size == 1 and v.dtype.kind in "biuf":
            return v.reshape(-1)[0].item()
        if v.size == 1 and v.dtype.kind == "c":
            return complex(v.reshape(-1)[0])
        return v
    if isinstance(v, np.generic):
        return v.item()
    return v


class _Keys:
    """Lookup of the keys of a dataset, with aliases, recording which were used."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = data
        self.used: set[str] = {"kind"}

    def has(self, key: str) -> bool:
        return any(k in self.data for k in ALIASES.get(key, (key,)))

    def raw(self, key: str, default: Any = None, *, required: bool = False) -> Any:
        for k in ALIASES.get(key, (key,)):
            if k in self.data and self.data[k] is not None:
                self.used.add(k)
                return self.data[k]
        if required:
            raise KeyError(f"the dataset has no {key!r}" + (f" (or {ALIASES[key][1:]})" if key in ALIASES else ""))
        return default

    def get(self, key: str, default: Any = None, *, required: bool = False) -> Any:
        return _value(self.raw(key, default, required=required))

    def num(self, key: str, default: float | None = None, *, required: bool = False) -> float | None:
        v = self.get(key, default, required=required)
        if isinstance(v, np.ndarray):
            raise ValueError(f"{key!r} must be a number, not an array of shape {v.shape} (give it, e.g. --{key} ...)")
        return None if v is None else float(v)

    def unused(self) -> list[str]:
        return sorted(set(self.data) - self.used)


def _settings(keys: _Keys, points: list[tuple[str, tuple[float, float, float]]] | None) -> RecordSettings:
    kw: dict[str, Any] = {}
    for name in _SETTINGS:
        if keys.has(name):
            v = keys.get(name)
            if name == "levels_db":
                v = [float(a) for a in np.atleast_1d(np.asarray(v, dtype=float))]
            elif name in ("plane_peaks",):
                v = bool(v)
            elif name == "metrics_harmonics":
                v = int(v)
            elif name != "focal_quantity":
                v = float(v)
            kw[name] = v
    if points:
        kw["points"] = list(points)
    return RecordSettings(**kw)


def _common(keys: _Keys, points: list[tuple[str, tuple[float, float, float]]] | None) -> dict[str, Any]:
    axis = keys.get("axis_xy")
    alpha = keys.get("alpha")
    return {
        "settings": _settings(keys, points),
        "axis_xy": None if axis is None else tuple(float(a) for a in np.asarray(axis, dtype=float).ravel()[:2]),
        "alpha": None if alpha is None else np.atleast_1d(np.asarray(alpha, dtype=float)),
        "focusing_depth": float(keys.get("focusing_depth", 0.0)),
    }


def _medium(keys: _Keys) -> dict[str, float]:
    return {
        "frequency": float(keys.num("frequency", required=True)),  # type: ignore[arg-type]
        "sound_speed": float(keys.num("sound_speed", required=True)),  # type: ignore[arg-type]
        "density": float(keys.num("density", required=True)),  # type: ignore[arg-type]
    }


def _fit(keys: _Keys) -> dict[str, Any]:
    return {
        "n_harmonics": int(keys.get("n_harmonics", 4)),
        "periods": keys.num("periods", 2.0),
        "window_start": keys.num("window_start"),
    }


# -- processing --------------------------------------------------------------------------------


def _harmonics(keys: _Keys, common: dict[str, Any]) -> Results:
    y = keys.raw("y")
    return frequency_domain.from_harmonics(
        np.asarray(keys.raw("harmonics", required=True)),
        x=np.asarray(keys.raw("x", required=True), dtype=float),
        y=None if y is None else np.asarray(y, dtype=float),
        z=np.atleast_1d(np.asarray(keys.raw("z", required=True), dtype=float)),
        layout=str(keys.get("layout", "zkxy")),
        phase=str(keys.get("phase", "retarded")),  # type: ignore[arg-type]
        time_sign=int(keys.get("time_sign", 1)),
        **_medium(keys),
        **common,
    )


def _time_series(keys: _Keys, common: dict[str, Any], *, fdtd_defaults: bool) -> Results:
    layout = str(keys.get("layout", "xyt" if fdtd_defaults else "xyzt"))
    coords = {c: np.atleast_1d(np.asarray(keys.raw(c, required=True), dtype=float)) for c in layout if c != "t"}
    module = fdtd if fdtd_defaults else time_domain
    return module.from_time_series(
        np.asarray(keys.raw("p", required=True), dtype=float),
        float(keys.num("dt", required=True)),  # type: ignore[arg-type]
        coords=coords,
        layout=layout,
        propagation=str(keys.get("propagation", "x" if fdtd_defaults else "z")),
        start_time=float(keys.get("start_time", 0.0)),
        detrend=bool(keys.get("detrend", fdtd_defaults)),
        **_fit(keys),
        **_medium(keys),
        **common,
    )


def _kwave(keys: _Keys, common: dict[str, Any]) -> Results:
    vecs = {}
    for c in "xyz":
        v = keys.raw(f"{c}_vec")
        if v is None:
            v = keys.raw(c)
        if v is not None:
            vecs[c] = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    if "x" not in vecs:
        raise KeyError("the dataset has no 'x_vec' (or 'x', or a 'kgrid' holding it)")
    dt = float(keys.num("dt", required=True))  # type: ignore[arg-type]
    kgrid = SimpleNamespace(dim=len(vecs), dt=dt, **{f"{c}_vec": v for c, v in vecs.items()})
    mask = keys.raw("sensor_mask")
    return time_domain.from_kwave(
        np.asarray(keys.raw("p", required=True), dtype=float),
        kgrid,
        sensor_mask=None if mask is None else np.asarray(mask),
        record_start_index=int(keys.get("record_start_index", 1)),
        propagation=str(keys.get("propagation", "x")),
        order=str(keys.get("order", "C")),
        **_fit(keys),
        **_medium(keys),
        **common,
    )


def _fullwave(keys: _Keys, common: dict[str, Any]) -> Results:
    outcoords = np.asarray(keys.raw("outcoords", required=True))
    ndim = outcoords.shape[1]
    spacing = {f"d{c}": float(keys.num(f"d{c}", required=True)) for c in "xyz"[:ndim]}  # type: ignore[arg-type]
    grid = SimpleNamespace(
        dt=float(keys.num("dt", required=True)),  # type: ignore[arg-type]
        f0=keys.num("frequency"),
        c0=keys.num("sound_speed"),
        **spacing,
    )
    sensor = SimpleNamespace(
        outcoords=outcoords,
        sampling_modulus_time=int(keys.get("sampling_modulus_time", 1)),
        is_sparse_grid=bool(keys.get("is_sparse_grid", False)),
    )
    origin = keys.raw("origin")
    return fdtd.from_fullwave(
        np.asarray(keys.raw("sensor_data", required=True), dtype=float),
        sensor,
        grid,
        **_medium(keys),
        propagation=str(keys.get("propagation", "x")),
        origin=None if origin is None else tuple(float(o) for o in np.asarray(origin, dtype=float).ravel()),
        detrend=bool(keys.get("detrend", True)),
        **_fit(keys),
        **common,
    )


def _babelbrain(keys: _Keys, common: dict[str, Any]) -> Results:
    field = str(keys.get("field", "p_complex"))
    used = [field, field.replace("p_complex", "p_amp"), "x_vec", "y_vec", "z_vec", "Material", "MaterialMap",
            "TargetLocation"]  # fmt: skip
    data = {k: keys.raw(k) for k in used if keys.has(k)}
    return babelbrain.from_babelbrain(
        data,
        frequency=keys.num("frequency", required=True),
        field=field,
        density=keys.num("density"),
        sound_speed=keys.num("sound_speed"),
        direction=str(keys.get("direction", "auto")),  # type: ignore[arg-type]
        **common,
    )


MSOUND_FIELDS = {
    "P_fundamental": ("P_fundamental", "p_fundamental", "p_fund"),
    "P_second": ("P_second", "p_second", "p_sec"),
    "p_time": ("p_time", "p_total"),
}


def _first(keys: _Keys, names: tuple[str, ...]) -> Any:
    for n in names:
        if n in keys.data and keys.data[n] is not None:
            return keys.raw(n)
    return None


def is_msound(data: Mapping[str, Any]) -> bool:
    """Whether the keys look like mSOUND's output (a fundamental, or time series with a mask)."""
    return any(k in data for k in MSOUND_FIELDS["P_fundamental"]) or (
        any(k in data for k in MSOUND_FIELDS["p_time"]) and "sensor_mask" in data
    )


def _msound(keys: _Keys, common: dict[str, Any]) -> Results:
    if not keys.has("frequency"):
        omega = _first(keys, ("omega_c", "omegac"))
        if omega is not None:
            keys.data = {**keys.data, "frequency": float(_value(omega)) / (2 * np.pi)}  # type: ignore[assignment]
    p1 = _first(keys, MSOUND_FIELDS["P_fundamental"])
    ptime = _first(keys, MSOUND_FIELDS["p_time"])
    if p1 is None and ptime is None:
        raise KeyError("the dataset has no 'P_fundamental' (or 'p_time' with a 'sensor_mask')")
    field = np.asarray(p1 if p1 is not None else keys.raw("sensor_mask", required=True))
    two_d = field.ndim == 2
    # in 2D, mgrid.y is the beam axis and dy the plane spacing; in 3D, dz
    dz = keys.num("dz_planes") or keys.num("dy" if two_d else "dz", required=True)
    grid: dict[str, Any] = {
        "dz": float(dz),  # type: ignore[arg-type]
        "x": keys.raw("x"),
        "dx": keys.num("dx"),
        "y": None if two_d else keys.raw("y"),
        "dy": None if two_d else keys.num("dy"),
        "z0": float(keys.get("z0", 0.0)),
    }
    if p1 is not None:
        p2 = _first(keys, MSOUND_FIELDS["P_second"])
        return msound.from_msound(
            np.asarray(p1), None if p2 is None else np.asarray(p2), **grid, **_medium(keys), **common
        )
    return msound.from_msound_time_series(
        np.asarray(ptime, dtype=float),
        np.asarray(keys.raw("sensor_mask", required=True)),
        dt=float(keys.num("dt", required=True)),  # type: ignore[arg-type]
        start_time=float(keys.get("start_time", 0.0)),
        **grid,
        **_fit(keys),
        **_medium(keys),
        **common,
    )


def _focus(keys: _Keys, common: dict[str, Any]) -> Results:
    if keys.has("x") and keys.has("z"):
        x, y, z = (np.atleast_1d(np.asarray(keys.raw(c, 0.0), dtype=float)) for c in "xyz")
    else:
        lim = {k: float(keys.num(k, required=True)) for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}  # type: ignore[arg-type]
        x, y, z = focus.grid_vectors(np.asarray(keys.raw("delta", required=True)), **lim)
    p = keys.raw("pressure")
    if p is None:
        p = keys.raw("p", required=True)
    p = np.asarray(p)
    if y.size == 1 and (p.ndim < 2 or p.shape[1] != 1):  # the singleton y squeezed out on reading
        p = np.expand_dims(p, 1)
    dt = keys.num("deltat")
    ts = keys.get("time_sign")
    return focus.from_focus(
        p,
        x=x,
        y=y,
        z=z,
        phase=str(keys.get("phase", "auto")),  # type: ignore[arg-type]
        time_sign=None if ts is None else int(ts),
        scale=float(keys.get("scale", 1.0)),
        dt=dt if dt is not None else keys.num("dt"),
        start_time=float(keys.get("tmin", keys.get("start_time", 0.0))),
        **_fit(keys),
        **_medium(keys),
        **common,
    )


def _field_names(keys: _Keys) -> Any:
    names = keys.get("field")
    return [str(n) for n in names.ravel()] if isinstance(names, np.ndarray) else names


def _time_options(keys: _Keys) -> dict[str, Any]:
    return {**_fit(keys), "detrend": bool(keys.get("detrend", False))}


def _axisymmetric(keys: _Keys, common: dict[str, Any]) -> Results:
    """Harmonics or time series, on an (r, z) grid, at mesh nodes or in an XDMF file."""
    common = {k: v for k, v in common.items() if k != "axis_xy"}  # the axis is r = 0
    ts = keys.get("time_sign")
    revolve = {"dx": keys.num("dx"), "r_max": keys.num("r_max"), **common, **_medium(keys)}
    angle = {"phase": str(keys.get("phase", "auto")), "time_sign": None if ts is None else int(ts)}
    mesh = {
        "coordinates": str(keys.get("coordinates", "rz")),
        "spacing": keys.num("spacing"),
        "max_edge": keys.num("max_edge"),
        "r": keys.raw("r"),
        "z": keys.raw("z"),
    }
    xdmf = keys.raw("xdmf")
    if xdmf is not None:
        names = _field_names(keys)
        if is_time_series(xdmf, names):
            return axisymmetric_time_domain.from_axisymmetric_xdmf_time_series(
                xdmf, names, **_time_options(keys), **mesh, **revolve
            )
        return axisymmetric_frequency_domain.from_axisymmetric_xdmf_harmonics(
            xdmf, names, time=keys.num("time"), **mesh, **angle, **revolve
        )
    if keys.has("points"):
        points = np.asarray(keys.raw("points", required=True), dtype=float)
        values = np.asarray(keys.raw("values", required=True))
        ct = keys.raw("cells")
        cells = {"cells": None if ct is None else np.asarray(ct), "cell_type": str(keys.get("cell_type", "triangle"))}
        if not np.iscomplexobj(values):
            return axisymmetric_time_domain.from_axisymmetric_mesh_time_series(
                points, values, float(keys.num("dt", required=True)),  # type: ignore[arg-type]
                start_time=float(keys.get("start_time", 0.0)), **_time_options(keys), **cells, **mesh, **revolve,
            )  # fmt: skip
        return axisymmetric_frequency_domain.from_axisymmetric_mesh_harmonics(
            points, values, **cells, **mesh, **angle, **revolve
        )
    r = np.asarray(keys.raw("r", required=True), dtype=float)
    z = np.atleast_1d(np.asarray(keys.raw("z", required=True), dtype=float))
    if keys.has("harmonics"):
        return axisymmetric_frequency_domain.from_axisymmetric_harmonics(
            np.asarray(keys.raw("harmonics")), r=r, z=z, layout=str(keys.get("layout", "zkr")), **angle, **revolve
        )
    return axisymmetric_time_domain.from_axisymmetric_time_series(
        np.asarray(keys.raw("p", required=True), dtype=float),
        float(keys.num("dt", required=True)),  # type: ignore[arg-type]
        r=r,
        z=z,
        layout=str(keys.get("layout", "rzt")),
        start_time=float(keys.get("start_time", 0.0)),
        **_time_options(keys),
        **revolve,
    )


def _unstructured(keys: _Keys, common: dict[str, Any]) -> Results:
    """Harmonics or time series at the nodes of a 3D (or planar 2D) mesh, or in an XDMF file."""
    prop = keys.get("propagation")
    grid: dict[str, Any] = {
        "propagation": None if prop is None else str(prop),
        "spacing": keys.num("spacing"),
        "max_edge": keys.num("max_edge"),
        **{c: keys.raw(c) for c in "xyz"},
        **_medium(keys),
        **common,
    }
    ts = keys.get("time_sign")
    angle = {"phase": str(keys.get("phase", "auto")), "time_sign": None if ts is None else int(ts)}
    xdmf = keys.raw("xdmf")
    if xdmf is not None:
        names = _field_names(keys)
        if is_time_series(xdmf, names):
            return unstructured_time_domain.from_unstructured_xdmf_time_series(
                xdmf, names, **_time_options(keys), **grid
            )
        return unstructured_frequency_domain.from_unstructured_xdmf_harmonics(
            xdmf, names, time=keys.num("time"), **angle, **grid
        )
    points = np.asarray(keys.raw("points", required=True), dtype=float)
    values = np.asarray(keys.raw("values", required=True))
    ct, cell_type = keys.raw("cells"), keys.get("cell_type")
    cells = {
        "cells": None if ct is None else np.asarray(ct),
        "cell_type": None if cell_type is None else str(cell_type),
    }
    if not np.iscomplexobj(values):
        return unstructured_time_domain.from_unstructured_time_series(
            points, values, float(keys.num("dt", required=True)),  # type: ignore[arg-type]
            start_time=float(keys.get("start_time", 0.0)), **_time_options(keys), **cells, **grid,
        )  # fmt: skip
    return unstructured_frequency_domain.from_unstructured_harmonics(points, values, **cells, **angle, **grid)


def _optimus(keys: _Keys, common: dict[str, Any]) -> Results:
    which = str(keys.get("field", "total")).removesuffix("_field")
    common = {k: v for k, v in common.items() if k != "axis_xy"}  # the beam axis is the source's
    if not keys.has("focusing_depth"):
        common["focusing_depth"] = None
    prop = keys.get("propagation")
    if prop is None:
        prop = keys.raw("source_axis")
    point = keys.raw("axis_point")
    if point is None:
        point = keys.raw("location")
    z_min = keys.num("z_min")
    return optimus.from_optimus(
        points=np.asarray(keys.raw("points", required=True), dtype=float),
        field=np.asarray(keys.raw(f"{which}_field", required=True)),
        propagation=prop if isinstance(prop, str) or prop is None else np.asarray(prop, dtype=float),
        axis_point=None if point is None else np.asarray(point, dtype=float),
        z_min=z_min,
        geometry=str(keys.get("geometry", "auto")),  # type: ignore[arg-type]
        phase=str(keys.get("phase", "absolute")),  # type: ignore[arg-type]
        time_sign=int(keys.get("time_sign", -1)),
        dx=keys.num("dx"),
        r_max=keys.num("r_max"),
        **_medium(keys),
        **common,
    )


def from_dataset(
    data: str | Path | Mapping[str, Any],
    kind: str | None = None,
    *,
    points: list[tuple[str, tuple[float, float, float]]] | None = None,
    **overrides: Any,
) -> Results:
    """:class:`Results` of a dataset (a file, or a mapping of its keys) of one ``kind``.

    Args:
        data: a dataset file (see :func:`load_dataset`) or its keys.
        kind: one of :data:`KINDS`; default the dataset's ``kind`` key.
        points: output points ``(name, (x, y, z))`` whose waveforms ``save()`` writes.
        overrides: keys that replace (or complete) those of the dataset, e.g. ``density=998``.
    """
    d = load_dataset(data) if isinstance(data, (str, Path)) else _flatten(data)
    if isinstance(d.get("xdmf"), (str, Path)):  # a file name given as a key
        d["xdmf"] = read_xdmf(d["xdmf"])
    if isinstance(data, (str, Path)) and "frequency" not in d and "f0" not in d:
        f = babelbrain.frequency_from_name(data)
        if f is not None:
            d["frequency"] = f
    d.update({k: v for k, v in overrides.items() if v is not None})
    keys = _Keys(d)
    kind = kind or keys.get("kind")
    if kind is None:
        kind = (
            "babelbrain" if babelbrain.is_babelbrain(d)
            else "msound" if is_msound(d)
            else "optimus" if optimus.is_optimus(d)
            else ("axisymmetric" if d["xdmf"].planar else "unstructured") if "xdmf" in d
            else None
        )  # fmt: skip
    if kind is None:
        raise ValueError(f"the dataset has no 'kind': give one of {KINDS}")
    kind = str(kind).replace("_", "-").lower()
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}: use one of {KINDS}")
    common = _common(keys, points)
    if kind == "harmonics":
        res = _harmonics(keys, common)
    elif kind in ("time-series", "fdtd"):
        res = _time_series(keys, common, fdtd_defaults=kind == "fdtd")
    elif kind == "kwave":
        res = _kwave(keys, common)
    elif kind == "babelbrain":
        res = _babelbrain(keys, common)
    elif kind == "msound":
        res = _msound(keys, common)
    elif kind == "focus":
        res = _focus(keys, common)
    elif kind == "axisymmetric":
        res = _axisymmetric(keys, common)
    elif kind == "unstructured":
        res = _unstructured(keys, common)
    elif kind == "optimus":
        res = _optimus(keys, common)
    else:
        res = _fullwave(keys, common)
    extra = keys.unused()
    if extra:
        log.info("keys of the dataset not used: %s", ", ".join(extra))
    return res


def save_dataset(path: str | Path, kind: str, **arrays: Any) -> Path:
    """Write a dataset of ``kind`` to a ``.npz`` file (None values are left out)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}: use one of {KINDS}")
    p = Path(path)
    np.savez_compressed(p, kind=np.array(kind), **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
    return p if p.suffix == ".npz" else p.with_suffix(p.suffix + ".npz")
