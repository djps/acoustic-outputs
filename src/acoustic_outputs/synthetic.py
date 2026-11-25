"""A focused Gaussian beam with harmonics, and datasets of it as each kind of solver writes them.

The field has known metrics, so it serves to test the package and to try the command line:

    P_n(r, z) = a_n g(r, z) exp(-i n theta(r, z)),   g = (w0 / w) exp(-r^2 / w^2)

(in 2D, g = sqrt(w0 / w) exp(-x^2 / w^2)), with the beam width w(z) = w0 sqrt(1 + ((z - zf) / zR)^2),
zR = k w0^2 / 2 and theta the curvature and Gouy phase of the fundamental. Every harmonic has the
envelope g, so the waveform at a point is the same up to a time shift and a factor g: p+ and p-
are proportional to g, the focus is at zf, the focal widths at a level L (pressure ratio) are
2 w0 sqrt(-ln L) and the focal length 2 zR sqrt(1/L^2 - 1). The plane-wave power is the same on
every plane: sum |a_n|^2 / (2 rho0 c0) times pi w0^2 / 2 (3D) or w0 sqrt(pi / 2) (2D, per metre).

The datasets (:func:`harmonics_dataset`, :func:`kwave_dataset`, :func:`fullwave_dataset`, ...)
are the keys :func:`acoustic_outputs.datasets.from_dataset` reads; time series start with noise
before the wave arrives, use a time step that does not divide the period, and for FDTD carry a
slow drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from acoustic_outputs._types import ComplexArray, RealArray


@dataclass(frozen=True)
class GaussianBeam:
    """A focused Gaussian beam with ``len(amplitudes)`` harmonics in phase at the focus."""

    frequency: float = 1.0e6
    sound_speed: float = 1500.0
    density: float = 1000.0
    waist: float = 2.0e-3
    focus: float = 30.0e-3
    amplitudes: tuple[float, ...] = (1.0e6, 3.0e5, 1.0e5, 3.0e4)
    x: RealArray = field(default_factory=lambda: np.linspace(-10e-3, 10e-3, 81))
    y: RealArray | None = field(default_factory=lambda: np.linspace(-10e-3, 10e-3, 81))
    z: RealArray = field(default_factory=lambda: np.linspace(15e-3, 45e-3, 31))

    def coarse(self) -> GaussianBeam:
        """The same beam on a coarser grid (for time series, which hold every sample)."""
        y = None if self.y is None else np.linspace(self.y[0], self.y[-1], 33)
        return replace(self, x=np.linspace(self.x[0], self.x[-1], 33), y=y, z=np.linspace(self.z[0], self.z[-1], 11))

    @property
    def k(self) -> float:
        """Wavenumber of the fundamental [rad/m]."""
        return 2 * math.pi * self.frequency / self.sound_speed

    @property
    def rayleigh(self) -> float:
        """Rayleigh range zR = k w0^2 / 2 [m]."""
        return self.k * self.waist**2 / 2

    @property
    def impedance(self) -> float:
        return self.density * self.sound_speed

    @property
    def n_harmonics(self) -> int:
        return len(self.amplitudes)

    def width(self, z: RealArray) -> RealArray:
        """Beam width w(z) [m]."""
        return self.waist * np.sqrt(1 + ((z - self.focus) / self.rayleigh) ** 2)

    def harmonics(self) -> ComplexArray:
        """Retarded harmonics on the planes, (nz, K, nx, ny) (ny = 1 in 2D)."""
        y = self.y if self.y is not None else np.zeros(1)
        h = self.harmonics_at(self.x[None, :, None], y[None, None, :], self.z[:, None, None])
        return np.moveaxis(h, 0, 1)

    def harmonics_at(self, x: Any, y: Any, z: Any) -> ComplexArray:
        """Retarded harmonics (K, ...) at the points (x, y, z) (broadcast together; y unused in 2D)."""
        x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
        r2 = x**2 + (y**2 if self.y is not None else 0.0)
        zr = self.rayleigh
        dz = z - self.focus
        w = self.width(z)
        if self.y is None:
            g = np.sqrt(self.waist / w) * np.exp(-r2 / w**2)
            gouy = 0.5 * np.arctan(dz / zr)
        else:
            g = (self.waist / w) * np.exp(-r2 / w**2)
            gouy = np.arctan(dz / zr)
        inv_r = dz / (dz**2 + zr**2)  # 1 / R(z), 0 at the focus
        theta = self.k * r2 * inv_r / 2 - gouy
        shape = (-1,) + (1,) * np.broadcast(x, y, z).ndim
        n = np.arange(1, self.n_harmonics + 1).reshape(shape)
        a = np.asarray(self.amplitudes, dtype=float).reshape(shape)
        return a * g * np.exp(-1j * n * theta)

    # -- the analytic metrics -------------------------------------------------------------

    def focal_width(self, level_db: float) -> float:
        """Width at ``level_db`` of p+ across the focus [m]."""
        return 2 * self.waist * math.sqrt(-math.log(10 ** (level_db / 20)))

    def focal_length(self, level_db: float) -> float:
        """Length at ``level_db`` of p+ along the axis [m]."""
        lv = 10 ** (level_db / 20)
        return 2 * self.rayleigh * math.sqrt((1 / lv**2 if self.y is not None else 1 / lv**4) - 1)

    @property
    def power(self) -> float:
        """Plane-wave power through every plane [W] (W/m in 2D)."""
        s = sum(a**2 for a in self.amplitudes) / (2 * self.impedance)
        return s * (math.pi * self.waist**2 / 2 if self.y is not None else self.waist * math.sqrt(math.pi / 2))

    # -- time series ------------------------------------------------------------------------

    def time_series(self, dt: float, nt: int, *, start_time: float = 0.0, seed: int = 3) -> RealArray:
        """p(x, y, z, t) at t = start_time + i dt, (nx, ny, nz, nt); noise until the wave arrives."""
        h = self.harmonics()
        t = start_time + dt * np.arange(nt)
        n = np.arange(1, self.n_harmonics + 1)
        w = 2 * math.pi * self.frequency
        delay = (self.z - self.z[0]) / self.sound_speed
        phase = np.exp(1j * w * n[None, :, None] * (t[None, None, :] - delay[:, None, None]))  # (nz, K, nt)
        p = np.einsum("zkxy,zkt->xyzt", h, phase).real
        late = t[None, :] >= delay[:, None] + 2 / self.frequency  # (nz, nt)
        noise = np.random.default_rng(seed).standard_normal(p.shape) * max(self.amplitudes)
        return np.where(late[None, None], p, noise)

    def duration(self, periods: float = 5.0) -> float:
        """Time for the wave to cross the planes, plus ``periods`` periods [s]."""
        return float((self.z[-1] - self.z[0]) / self.sound_speed + periods / self.frequency)

    def medium(self) -> dict[str, float]:
        return {"frequency": self.frequency, "sound_speed": self.sound_speed, "density": self.density}


# -- datasets as each kind of solver writes them -------------------------------------------


def harmonics_dataset(beam: GaussianBeam, *, absolute: bool = False) -> dict[str, Any]:
    """Harmonics on planes in the layout (x, y, z, k), as a frequency-domain solver writes them."""
    h = beam.harmonics()
    if absolute:
        n = np.arange(1, beam.n_harmonics + 1)[None, :, None, None]
        h = h * np.exp(-1j * n * beam.k * (beam.z - beam.z[0])[:, None, None, None])
    if beam.y is None:
        return {"harmonics": np.transpose(h[..., 0], (2, 0, 1)), "layout": "xzk", "x": beam.x, "z": beam.z,
                "phase": "absolute" if absolute else "retarded", **beam.medium()}  # fmt: skip
    return {
        "harmonics": np.transpose(h, (2, 3, 0, 1)),
        "layout": "xyzk",
        "x": beam.x,
        "y": beam.y,
        "z": beam.z,
        "phase": "absolute" if absolute else "retarded",
        **beam.medium(),
    }


def time_series_dataset(beam: GaussianBeam, *, steps_per_period: float = 23.7) -> dict[str, Any]:
    """Gridded time series (x, y, z, t) with z the propagation axis."""
    dt = 1 / (beam.frequency * steps_per_period)
    p = beam.time_series(dt, int(beam.duration() / dt))
    assert beam.y is not None
    return {"p": p, "dt": dt, "layout": "xyzt", "propagation": "z", "x": beam.x, "y": beam.y, "z": beam.z,
            "n_harmonics": beam.n_harmonics, **beam.medium()}  # fmt: skip


def kwave_dataset(
    beam: GaussianBeam,
    *,
    steps_per_period: float = 19.3,
    record_start_index: int = 40,
    pad: tuple[int, ...] = (2, 3, 1),
) -> dict[str, Any]:
    """k-wave-python sensor data: (n_sensor, Nt) in C order of a block mask inside a larger grid,
    with the beam along the k-Wave x axis and recording from ``record_start_index`` (1-based)."""
    dt = 1 / (beam.frequency * steps_per_period)
    nt = int(beam.duration() / dt)
    p = beam.time_series(dt, nt, start_time=(record_start_index - 1) * dt)  # (x, y, z, t)
    if beam.y is None:
        blk = np.transpose(p[:, 0], (1, 0, 2))  # k-Wave (x = propagation, y)
        axes = [beam.z, beam.x]
    else:
        blk = np.transpose(p, (2, 0, 1, 3))  # k-Wave (x = propagation, y, z)
        axes = [beam.z, beam.x, beam.y]
    pad = pad[: len(axes)]
    shape = tuple(n + 2 * q for n, q in zip(blk.shape[:-1], pad, strict=True))
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(q, q + n) for q, n in zip(pad, blk.shape[:-1], strict=True))] = True
    vecs = {f"{c}_vec": a[0] - q * (a[1] - a[0]) + (a[1] - a[0]) * np.arange(n)
            for c, a, q, n in zip("xyz", axes, pad, shape, strict=False)}  # fmt: skip
    return {"p": blk.reshape(-1, nt), "dt": dt, "sensor_mask": mask, "record_start_index": record_start_index,
            "propagation": "x", "n_harmonics": beam.n_harmonics, **vecs, **beam.medium()}  # fmt: skip


def fullwave_dataset(
    beam: GaussianBeam,
    *,
    steps_per_period: float = 31.1,
    modulus: int = 3,
    first: tuple[int, ...] = (7, 2, 5),
    drift: tuple[float, float] = (3e4, 2e10),
    seed: int = 0,
) -> dict[str, Any]:
    """fullwave25 sensor data: sensors by grid index in shuffled order, kept every ``modulus``
    steps, with the beam along x (axial) and a drift c0 + c1 t from the absorbing layers."""
    dt = 1 / (beam.frequency * steps_per_period)
    nt = int(beam.duration() / (dt * modulus))
    p = beam.time_series(dt * modulus, nt)
    if beam.y is None:
        blk = np.transpose(p[:, 0], (1, 0, 2))
        axes = [beam.z, beam.x]
    else:
        blk = np.transpose(p, (2, 0, 1, 3))
        axes = [beam.z, beam.x, beam.y]
    t = dt * modulus * np.arange(nt)
    blk = blk + drift[0] + drift[1] * t
    lo = np.array(first[: len(axes)])
    idx = np.argwhere(np.ones(blk.shape[:-1], dtype=bool)) + lo
    order = np.random.default_rng(seed).permutation(len(idx))
    d = [a[1] - a[0] for a in axes]
    return {
        "sensor_data": blk.reshape(-1, nt)[order],
        "outcoords": idx[order],
        "sampling_modulus_time": modulus,
        "dt": dt,
        **{f"d{c}": s for c, s in zip("xyz", d, strict=False)},
        "origin": np.array([a[0] - q * s for a, q, s in zip(axes, lo, d, strict=True)]),
        "propagation": "x",
        "n_harmonics": beam.n_harmonics,
        "f0": beam.frequency,
        "c0": beam.sound_speed,
        "density": beam.density,
    }


def fdtd_dataset(beam: GaussianBeam, *, steps_per_period: float = 27.9) -> dict[str, Any]:
    """A 2D FDTD record (x axial, y lateral, t) with drift, for a beam with ``y=None``."""
    if beam.y is not None:
        raise ValueError("fdtd_dataset is for a 2D beam (y=None)")
    dt = 1 / (beam.frequency * steps_per_period)
    nt = int(beam.duration() / dt)
    p = np.transpose(beam.time_series(dt, nt)[:, 0], (1, 0, 2))  # (axial, lateral, t)
    p = p + 1e4 + 1e10 * dt * np.arange(nt)
    return {"p": p, "dt": dt, "layout": "xyt", "propagation": "x", "x": beam.z, "y": beam.x,
            "n_harmonics": beam.n_harmonics, **beam.medium()}  # fmt: skip


def babelbrain_dataset(beam: GaussianBeam, *, attenuation: float = 5.0) -> dict[str, Any]:
    """BabelBrain's ``DataForSim.h5`` keys for the fundamental of ``beam``.

    The fundamental is given in absolute phase on the (x, y, z) grid and attenuated by
    ``attenuation`` [Np/m] along z, as through tissue. Brain tissue (material 1) fills the planes
    beyond the first quarter, which are water (material 0), and the target is at the focus.
    """
    if beam.y is None:
        raise ValueError("BabelBrain fields are 3D")
    h = beam.harmonics()[:, 0]  # (nz, nx, ny)
    h = h * np.exp(-1j * beam.k * (beam.z - beam.z[0]) - attenuation * (beam.z - beam.z[0]))[:, None, None]
    p = np.transpose(h, (1, 2, 0))  # (x, y, z)
    material = np.zeros(p.shape, dtype=np.uint32)
    material[:, :, beam.z.size // 4 :] = 1
    target = np.array([beam.x.size // 2, beam.y.size // 2, int(np.argmin(np.abs(beam.z - beam.focus)))])
    return {
        "p_complex": p.astype(np.complex64),
        "p_amp": np.abs(p).astype(np.float32),
        "x_vec": beam.x,
        "y_vec": beam.y,
        "z_vec": beam.z,
        "SpatialStep": float(beam.x[1] - beam.x[0]),
        "Material": np.array([[1000.0, 1500.0, 0.0, 0.0, 0.0], [1041.0, 1562.0, 0.0, attenuation, 0.0]]),
        "MaterialMap": material,
        "TargetLocation": target,
        "frequency": beam.frequency,
    }


def _absolute(beam: GaussianBeam, k_max: int | None = None) -> ComplexArray:
    """The harmonics (nz, K, nx, ny) in absolute phase, exp(-i n k (z - z0)), for p = Re P exp(+i omega t)."""
    h = beam.harmonics()[:, :k_max]
    n = np.arange(1, h.shape[1] + 1)[None, :]
    return h * np.exp(-1j * n * beam.k * (beam.z - beam.z[0])[:, None])[:, :, None, None]


def _centred(n: int, d: float) -> RealArray:
    start = (-n / 2 + 1 - 0.5) if n % 2 else -n / 2
    return d * (start + np.arange(n))


def msound_dataset(beam: GaussianBeam) -> dict[str, Any]:
    """mSOUND's ``Forward3D_fund`` / ``Forward3D_sec`` output (``Forward2D_...`` for a 2D beam).

    The fundamental and second harmonic of ``beam`` in absolute phase, (nx, ny, n_planes) (2D:
    (nx, n_planes)), with the grid as ``struct(mgrid)`` holds it: centred x (and y) from the
    spacing, the plane spacing dz (dy in 2D), and the source plane at ``z0`` = the first plane.
    """
    h = _absolute(beam, 2)  # (nz, 2, nx, ny)
    dx, dz = float(beam.x[1] - beam.x[0]), float(beam.z[1] - beam.z[0])
    if not np.allclose(beam.x, _centred(beam.x.size, dx)) or not np.allclose(np.diff(beam.z), dz):
        raise ValueError("mSOUND grids are centred in x and y and uniform in z")
    out: dict[str, Any] = {"fc": beam.frequency, "c0": beam.sound_speed, "rho0": beam.density, "z0": float(beam.z[0]),
                           "dx": dx, "num_x": beam.x.size, "beta": 3.6, "ca": 0.0, "cb": 2.0}  # fmt: skip
    if beam.y is None:
        p = np.transpose(h[:, :, :, 0], (1, 2, 0))  # (2, nx, n_planes)
        out.update(dy=dz, dim=2, y=_centred(beam.z.size - 1, dz))  # 2D: y is mSOUND's beam axis
    else:
        p = np.transpose(h, (1, 2, 3, 0))  # (2, nx, ny, n_planes)
        out.update(dy=float(beam.y[1] - beam.y[0]), dz=dz, dim=3, y=beam.y, x=beam.x)
    return {"P_fundamental": p[0], "P_second": p[1], **out}


def msound_time_series_dataset(beam: GaussianBeam, *, steps_per_period: float = 23.7, pad: int = 2) -> dict[str, Any]:
    """mSOUND's ``Forward3D`` output: ``p_time`` (num_t, n_sensor) on a block ``sensor_mask``.

    The mask leaves ``pad`` points on each side in x and y, and the source plane, unrecorded; the
    sensors are in column-major order, and the time starts at 0 (mgrid.t).
    """
    if beam.y is None:
        raise ValueError("a 3D beam is needed")
    dt = 1 / (beam.frequency * steps_per_period)
    nt = int(beam.duration() / dt)
    blk = beam.time_series(dt, nt)  # (x, y, z, t)
    dx, dz = float(beam.x[1] - beam.x[0]), float(beam.z[1] - beam.z[0])
    shape = (beam.x.size + 2 * pad, beam.y.size + 2 * pad, beam.z.size + 1)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[pad:-pad, pad:-pad, 1:] = 1
    p_time = blk.reshape(-1, nt, order="F").T
    return {"p_time": p_time, "sensor_mask": mask, "dt": dt, "dx": dx, "dy": float(beam.y[1] - beam.y[0]), "dz": dz,
            "z0": float(beam.z[0] - dz), "fc": beam.frequency, "c0": beam.sound_speed, "rho0": beam.density,
            "n_harmonics": beam.n_harmonics}  # fmt: skip


def focus_dataset(beam: GaussianBeam, mode: str = "cw", *, time_sign: int = 1,
                  steps_per_period: float = 21.3) -> dict[str, Any]:  # fmt: skip
    """FOCUS output of ``beam`` on its coordinate grid (x, y, z), indexed (x, y, z, ...).

    ``mode``: ``"cw"`` (``fnm_cw``: the complex fundamental in absolute phase, conjugated for
    ``time_sign`` -1), ``"kzk"`` (``kzk_cw``: every harmonic, retarded, (x, y, z, n)) or
    ``"transient"`` (``fnm_tsd``: real (x, y, z, t) from ``tmin`` in steps of ``deltat``).
    A 2D beam has a single y = 0.
    """
    y = beam.y if beam.y is not None else np.zeros(1)
    out: dict[str, Any] = {"x": beam.x, "y": y, "z": beam.z, "f0": beam.frequency, "soundspeed": beam.sound_speed,
                           "density": beam.density}  # fmt: skip
    if mode == "cw":
        p = np.transpose(_absolute(beam, 1)[:, 0], (1, 2, 0))
        out["pressure"] = p if time_sign == 1 else p.conj()
    elif mode == "kzk":
        out["pressure"] = np.transpose(beam.harmonics(), (2, 3, 0, 1))
    elif mode == "transient":
        dt = 1 / (beam.frequency * steps_per_period)
        tmin = 2e-6
        out.update(pressure=beam.time_series(dt, int(beam.duration() / dt), start_time=tmin), deltat=dt, tmin=tmin,
                   n_harmonics=beam.n_harmonics)  # fmt: skip
    else:
        raise ValueError(f"unknown mode {mode!r}: use 'cw', 'kzk' or 'transient'")
    return out


# -- axisymmetric fields ----------------------------------------------------------------------


def radial_beam(beam: GaussianBeam, r: RealArray | None = None) -> GaussianBeam:
    """The same (3D, axisymmetric) beam on the half line r = x >= 0, y = 0."""
    if beam.y is None:
        raise ValueError("an axisymmetric field needs a 3D beam")
    rv = np.asarray(beam.x[beam.x >= -1e-15] if r is None else r, dtype=float)
    return replace(beam, x=np.abs(rv), y=np.zeros(1))


def axisymmetric_dataset(beam: GaussianBeam, *, absolute: bool = False) -> dict[str, Any]:
    """Harmonics of ``beam`` on an (r, z) grid in the layout (r, z, k), retarded or absolute."""
    rb = radial_beam(beam)
    h = _absolute(rb) if absolute else rb.harmonics()
    return {"harmonics": np.transpose(h[..., 0], (2, 0, 1)), "r": rb.x, "z": rb.z, "layout": "rzk",
            "phase": "absolute" if absolute else "retarded", **beam.medium()}  # fmt: skip


def axisymmetric_mesh(beam: GaussianBeam, *, spacing: float = 0.25e-3, seed: int = 5) -> dict[str, Any]:
    """Absolute-phase harmonics at the nodes of an unstructured triangular (r, z) mesh.

    The nodes are a jittered grid (fixed on the axis and the edges) and the cells their
    Delaunay triangles, as a finite-element solver would give them (``points`` (n, 2) as
    (r, z), ``cells`` (m, 3), ``values`` (n, K)).
    """
    from scipy.spatial import Delaunay

    r_max = float(beam.x[-1])
    r = spacing * np.arange(round(r_max / spacing) + 1)
    z = beam.z[0] + spacing * np.arange(round((beam.z[-1] - beam.z[0]) / spacing) + 1)
    R, Z = np.meshgrid(r, z, indexing="ij")
    rng = np.random.default_rng(seed)
    jr = np.where((R > 0) & (R < r[-1]), rng.uniform(-0.3, 0.3, R.shape) * spacing, 0.0)
    jz = np.where((Z > z[0]) & (Z < z[-1]), rng.uniform(-0.3, 0.3, Z.shape) * spacing, 0.0)
    pts = np.stack([(R + jr).ravel(), (Z + jz).ravel()], axis=1)
    h = beam.harmonics_at(pts[:, 0], 0.0, pts[:, 1])
    n = np.arange(1, beam.n_harmonics + 1)[:, None]
    h = h * np.exp(-1j * n * beam.k * (pts[:, 1] - beam.z[0]))
    return {"points": pts, "cells": Delaunay(pts).simplices, "values": h.T, **beam.medium()}


# -- unstructured meshes ------------------------------------------------------------------------


def jittered_nodes(axes: list[RealArray], *, jitter: float = 0.3, seed: int = 5) -> RealArray:
    """The nodes of the grid ``axes`` (d vectors), moved at random by up to ``jitter`` of a
    spacing, except on the faces of the box."""
    grids = np.meshgrid(*axes, indexing="ij")
    rng = np.random.default_rng(seed)
    cols = []
    for g, a in zip(grids, axes, strict=True):
        h = float(a[1] - a[0])
        inner = (g > a[0]) & (g < a[-1])
        cols.append((g + np.where(inner, rng.uniform(-jitter, jitter, g.shape) * h, 0.0)).ravel())
    return np.stack(cols, axis=1)


def unstructured_mesh(beam: GaussianBeam, *, spacing: float = 0.5e-3, absolute: bool = True,
                      time_sign: int = 1, seed: int = 5) -> dict[str, Any]:  # fmt: skip
    """Harmonics of ``beam`` at the nodes of an unstructured mesh over its box (x, y, z).

    Tetrahedra in 3D, triangles in 2D (a beam without y; the nodes are then (x, z)), the
    Delaunay cells of a jittered grid of nodes ``spacing`` apart. ``points`` (n, d), ``cells``,
    ``values`` (n, K) in absolute phase (exp(-i n k (z - z0)), conjugated for ``time_sign`` -1)
    or retarded.
    """
    from scipy.spatial import Delaunay

    def axis(v: RealArray) -> RealArray:
        return v[0] + spacing * np.arange(round((v[-1] - v[0]) / spacing) + 1)

    axes = [axis(beam.x)] + ([] if beam.y is None else [axis(beam.y)]) + [axis(beam.z)]
    pts = jittered_nodes(axes, seed=seed)
    zc = pts[:, -1]
    h = beam.harmonics_at(pts[:, 0], 0.0 if beam.y is None else pts[:, 1], zc)
    if absolute:
        n = np.arange(1, beam.n_harmonics + 1)[:, None]
        h = h * np.exp(-1j * n * beam.k * (zc - beam.z[0]))
    if time_sign == -1:
        h = h.conj()
    return {"points": pts, "cells": Delaunay(pts).simplices, "values": h.T, **beam.medium()}


def mesh_time_series(
    points: RealArray, values: ComplexArray, beam: GaussianBeam, dt: float, nt: int, *, seed: int = 3
) -> RealArray:
    """Real time series (n, nt) at mesh nodes from their absolute-phase harmonics ``values`` (n, K).

    As a time-domain solver records them: p = Re sum P_n exp(+i n omega t) once the wave has
    arrived (two periods after the travel time from the first plane, along the last coordinate),
    noise before.
    """
    t = dt * np.arange(nt)
    n = np.arange(1, values.shape[1] + 1)
    w = 2 * math.pi * beam.frequency
    p = np.einsum("pk,kt->pt", values, np.exp(1j * w * n[:, None] * t[None])).real
    arrival = (points[:, -1] - beam.z[0]) / beam.sound_speed + 2 / beam.frequency
    noise = np.random.default_rng(seed).standard_normal(p.shape) * max(beam.amplitudes)
    return np.where(t[None] >= arrival[:, None], p, noise)


# -- OptimUS ------------------------------------------------------------------------------------


def optimus_dataset(
    beam: GaussianBeam,
    mode: str = "volume",
    *,
    propagation: str = "x",
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scattered: float = 0.0,
) -> dict[str, Any]:
    """The fundamental of ``beam`` as OptimUS's ``export_to_file(..., file_format="mat")`` saves it.

    The beam travels along ``propagation`` (``"x"``, OptimUS's default, or ``"-z"``, ...) from
    ``location``, its z the distance along the beam; the field is in absolute phase with
    exp(-i omega t) (BEMPP's convention). ``mode``: ``"volume"`` (the planes z = beam.z across
    the beam, on the beam's (x, y) grid, as ``VisualisePlane`` objects at each offset) or
    ``"plane"`` (one plane through the axis, the beam's x across, as one ``VisualisePlane``).
    ``scattered`` adds that fraction of the field as a "scattered" part. The points are in
    ``numpy.mgrid`` order of each plane, as OptimUS makes them.
    """
    name = propagation.lstrip("+-")
    axis, sign = "xyz".index(name), (-1 if propagation.startswith("-") else 1)
    transverse = [i for i in range(3) if i != axis]
    t2 = beam.y if (mode == "volume" and beam.y is not None) else np.zeros(1)
    if mode not in ("volume", "plane"):
        raise ValueError(f"unknown mode {mode!r}: use 'volume' or 'plane'")
    zb, a1, a2 = np.meshgrid(beam.z, beam.x, t2, indexing="ij")
    zb, a1, a2 = zb.ravel(), a1.ravel(), a2.ravel()
    pts = np.zeros((3, zb.size))
    pts[axis] = location[axis] + sign * zb
    pts[transverse[0]] = location[transverse[0]] + a1
    pts[transverse[1]] = location[transverse[1]] + a2
    p = beam.harmonics_at(a1, a2, zb)[0] * np.exp(-1j * beam.k * zb)  # absolute, exp(+i omega t)
    total = p.conj()  # exp(-i omega t)
    return {
        "points": pts,
        "total_field": total,
        "incident_field": total * (1 - scattered),
        "scattered_field": total * scattered,
        "L2_norm_ptot_MPa": float(np.linalg.norm(total)),
        "gmres_iter_count": 12,
    }
