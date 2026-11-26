"""Build :class:`~acoustic_outputs.results.Results` from planes of harmonics, whatever solver made them.

Every field parameter of :class:`~acoustic_outputs.results.Results` (and of ``docs/iec_outputs.md``) is
computed from transverse planes z = const holding the harmonics P_n(x, y), n = 1 .. K, of a
periodic (continuous-wave) field, p = Re sum P_n exp(+i n omega tau), on a regular (x, y) grid.
:class:`PlaneRecorder` takes those planes one at a time, in increasing z, and keeps only what the
metrics need: the beam axis, the two lines through it, plane extremes, power and beam centre per
plane, and copies of the source, focal, spatial-peak and face planes. It never holds the 3D field.

A marching solver can feed it as it goes (pyWave3D does). The modules of
:mod:`acoustic_outputs.preprocess` turn the output of other solvers into such planes: frequency-domain
pseudo-spectral codes (:mod:`~acoustic_outputs.preprocess.spectral`), time-domain pseudo-spectral codes
such as k-wave-python (:mod:`~acoustic_outputs.preprocess.time_domain`) and FDTD codes such as fullwave25
(:mod:`~acoustic_outputs.preprocess.fdtd`).

The phase of each plane may be retarded (tau = t - z / c0) or absolute: magnitudes, peak pressures
and intensities do not depend on a phase common to a plane. Only the waveforms written for output
points are shown in the time frame of their plane.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from acoustic_outputs import iec
from acoustic_outputs._types import ComplexArray, RealArray
from acoustic_outputs.grid import Grid
from acoustic_outputs.results import FocalQuantity, Results
from acoustic_outputs.waveform import peak_pressures

log = logging.getLogger(__name__)

#: a box of the sampled space as its two opposite corners, ((a0, b0), (a1, b1)) [m]
Extent = tuple[tuple[float, float], tuple[float, float]]


@dataclass
class RecordSettings:
    """What is recorded and how the metrics are reported.

    Attributes:
        focal_quantity: the quantity whose maximum on the beam axis defines the focus.
        plane_peaks: also find the peak pressures over each whole plane (and the spatial peak).
        points: output points ``(name, (x, y, z))``: the harmonics at the grid point nearest
            (x, y) on the recorded plane nearest z.
        metrics_harmonics: harmonics reported in the metrics and files.
        levels_db: levels of the focal widths, lengths and areas; the first is the main one.
        hydrophone_diameter: diameter of a simulated hydrophone [m] (0: none).
        derating: derating coefficient [dB / (cm MHz)].
        face_distance: distance from the first plane of the plane kept for IEC 61689 [m].
        workers: threads for the FFTs.
        margin: points at each edge of the planes (in x, and in y unless ny = 1) that are not
            part of the field, such as a solver's absorbing layer: they are left out of
            everything recorded, and the results' grid is the interior.
        plane_precision: precision of the whole planes kept (source, face, focal, spatial-peak
            and final): ``"single"`` halves their memory and size on disk. Everything computed
            while recording uses the planes as given, at their own precision. The slices and
            the x-y window below are kept at the same precision.
        slices: keep all the harmonics on the longitudinal slices through the beam axis:
            ``"xz"`` (y = axis), ``"yz"`` (x = axis), ``"both"`` or ``"none"``. They give the
            waveforms off the axis at every recorded depth (e.g. a numerical schlieren image).
        slice_extent: the part of the slices kept, as a pair of points (start, end) =
            ((s0, z0), (s1, z1)) [m], s the transverse coordinate of the slice (x for ``xz``,
            y for ``yz``, in the grid's coordinates) and z that of the planes. The grid points
            and planes with s0 <= s <= s1 and z0 <= z <= z1 are kept. None: the whole slices.
        xy_plane: keep all the harmonics on a window of an x-y plane.
        xy_extent: the window, as its bottom-left and top-right corners ((x0, y0), (x1, y1))
            [m]: the grid points with x0 <= x <= x1 and y0 <= y <= y1. None: the whole plane.
        xy_z: the depth of that plane [m]: the recorded plane nearest it. None: the plane of the
            focus (the maximum of ``focal_quantity`` on the axis).
    """

    focal_quantity: FocalQuantity = "p_positive"
    plane_peaks: bool = True
    points: list[tuple[str, tuple[float, float, float]]] = field(default_factory=list)
    metrics_harmonics: int = 4
    levels_db: list[float] = field(default_factory=lambda: [-6.0, -3.0, -12.0, -20.0])
    hydrophone_diameter: float = 0.0
    derating: float = iec.DERATING_DB_PER_CM_MHZ
    face_distance: float = 0.003
    workers: int = 1
    margin: int = 0
    plane_precision: Literal["double", "single"] = "double"
    slices: Literal["none", "xz", "yz", "both"] = "none"
    slice_extent: Extent | None = None
    xy_plane: bool = False
    xy_extent: Extent | None = None
    xy_z: float | None = None


def _window(coord: RealArray, lo: float, hi: float, name: str) -> slice:
    """The contiguous grid indices with lo <= coord <= hi (to round-off)."""
    if hi < lo:
        raise ValueError(f"the extent in {name} runs backwards: {lo} to {hi}")
    eps = 1e-9 * max(abs(float(coord[0])), abs(float(coord[-1])), 1e-3)
    inside = np.flatnonzero((coord >= lo - eps) & (coord <= hi + eps))
    if inside.size == 0:
        raise ValueError(f"the extent {lo} to {hi} in {name} holds no grid point ({coord[0]} to {coord[-1]})")
    return slice(int(inside[0]), int(inside[-1]) + 1)


class PlaneRecorder:
    """Accumulates the recorded quantities over planes of harmonics, then makes the Results.

    Args:
        grid: the transverse grid (x, y, dx, dy; ``z0`` is the source plane). For a 2D field
            (one transverse dimension) use ``ny = 1``; plane quantities are then per unit
            length in y when ``dy = 1``.
        n_harmonics: K, the number of harmonics of every plane.
        frequency: base frequency f0 [Hz].
        sound_speed: c0 [m/s].
        impedance: rho0 c0 [kg / (m^2 s)].
        settings: what to record.
        axis_index: (i, j) of the beam axis on the plane; by default the plane centre.
    """

    def __init__(
        self,
        grid: Grid,
        n_harmonics: int,
        *,
        frequency: float,
        sound_speed: float,
        impedance: float,
        settings: RecordSettings | None = None,
        axis_index: tuple[int, int] | None = None,
    ) -> None:
        self.settings = settings if settings is not None else RecordSettings()
        m = self.settings.margin
        if m < 0 or 2 * m >= grid.nx - 1 or (grid.ny > 1 and 2 * m >= grid.ny):
            raise ValueError(f"margin of {m} points does not fit the {grid.nx} x {grid.ny} plane")
        self.plane_shape = (grid.nx, grid.ny)
        self._crop = (slice(m, grid.nx - m), slice(m, grid.ny - m) if grid.ny > 1 else slice(None))
        if m:
            ny = grid.ny - 2 * m if grid.ny > 1 else grid.ny
            y0 = grid.y0 + m * grid.dy if grid.ny > 1 else grid.y0
            grid = dataclasses.replace(grid, nx=grid.nx - 2 * m, ny=ny, x0=grid.x0 + m * grid.dx, y0=y0)
        self.grid = grid
        self._dtype = np.complex64 if self.settings.plane_precision == "single" else np.complex128
        self.k = n_harmonics
        self.frequency = frequency
        self.sound_speed = sound_speed
        self.impedance = impedance
        self.axis_index = axis_index if axis_index is not None else grid.centre_index
        ix, iy = self.axis_index
        if not (0 <= ix < grid.nx and 0 <= iy < grid.ny):
            raise ValueError(f"beam axis {self.axis_index} is outside the {grid.nx} x {grid.ny} plane")
        self._rows: dict[str, list[Any]] = {
            k: []
            for k in (
                "z",
                "axis",
                "ax_pp",
                "ax_pm",
                "xz_pp",
                "xz_pm",
                "xz_a",
                "yz_pp",
                "yz_pm",
                "yz_a",
                "pl_pp",
                "pl_pm",
                "power",
                "pl_i",
                "cen_x",
                "cen_y",
            )
        }
        self._max_pp = np.zeros((grid.nx, grid.ny)) if self.settings.plane_peaks else None
        self._focus: dict[str, Any] = {"value": -math.inf, "index": 0, "field": None}
        self._peak: dict[str, Any] = {"value": -math.inf, "index": 0, "ij": (ix, iy), "field": None}
        self._face: dict[str, Any] = {"distance": math.inf, "index": 0, "field": None}
        self._source: ComplexArray | None = None
        self._last: ComplexArray | None = None
        self._points: dict[str, dict[str, Any]] = {}
        g = grid
        for name, pos in self.settings.points:
            x, y, _ = pos
            inside_x = g.x0 - g.dx / 2 <= x <= g.x[-1] + g.dx / 2
            inside_y = g.ny == 1 or g.y0 - g.dy / 2 <= y <= g.y[-1] + g.dy / 2  # 2D: y is not a coordinate
            if not (inside_x and inside_y):
                log.warning("output point %s at %s is outside the plane: not written", name, pos)
                continue
            i, j = g.nearest(x, y)
            self._points[name] = {"pos": pos, "ij": (i, j), "distance": math.inf, "z": math.nan, "value": None}

        # longitudinal slices through the axis, and a window of an x-y plane, with all harmonics
        s = self.settings
        if s.slices not in ("none", "xz", "yz", "both"):
            raise ValueError(f"slices must be 'none', 'xz', 'yz' or 'both', not {s.slices!r}")
        if s.slices in ("yz", "both") and g.ny == 1:
            raise ValueError("a 2D field (ny = 1) has no y-z slice")
        self._z_range = (-math.inf, math.inf)
        self._slice_cols: dict[str, slice] = {}
        if s.slices != "none":
            (s0, z0), (s1, z1) = s.slice_extent if s.slice_extent is not None else ((-math.inf, -math.inf),
                                                                                    (math.inf, math.inf))  # fmt: skip
            if z1 < z0:
                raise ValueError(f"the slice extent in z runs backwards: {z0} to {z1}")
            self._z_range = (z0, z1)
            for name, coord in (("xz", g.x), ("yz", g.y)):
                if s.slices in (name, "both"):
                    self._slice_cols[name] = _window(coord, s0, s1, name[0])
        self._slices: dict[str, list[ComplexArray]] = {name: [] for name in self._slice_cols}
        self._slice_z: list[float] = []
        self._xy_cols: tuple[slice, slice] | None = None
        self._xy: dict[str, Any] = {"distance": math.inf, "z": math.nan, "field": None}
        if s.xy_plane:
            if s.xy_extent is None:
                self._xy_cols = (slice(None), slice(None))
            else:
                (x0, y0), (x1, y1) = s.xy_extent
                wy = _window(g.y, y0, y1, "y") if g.ny > 1 else slice(None)
                self._xy_cols = (_window(g.x, x0, x1, "x"), wy)

    @property
    def n_planes(self) -> int:
        """Planes recorded so far."""
        return len(self._rows["z"])

    @property
    def axis_p_plus_max(self) -> float:
        """Largest peak positive pressure on the axis so far [Pa]."""
        return max(self._rows["ax_pp"], default=math.nan)

    def add(self, z: float, harmonics: ComplexArray, n_active: int | None = None) -> None:
        """Record the plane at ``z`` holding ``harmonics`` (K, nx, ny), on the whole plane (the
        ``margin`` of the settings is removed here).

        Planes must come in increasing z. The recorder keeps a reference to the last plane (it
        becomes ``Results.final``), so a caller reusing one buffer must not change it after the
        last ``add``.

        Args:
            z: position of the plane [m].
            harmonics: P_1 .. P_K on the plane.
            n_active: only the first ``n_active`` harmonics are non-zero (default: all K).
        """
        g, s, rows = self.grid, self.settings, self._rows
        if harmonics.shape != (self.k, *self.plane_shape):
            raise ValueError(f"plane of shape {harmonics.shape}, expected {(self.k, *self.plane_shape)}")
        if rows["z"] and z <= rows["z"][-1]:
            raise ValueError(f"planes must come in increasing z ({z} after {rows['z'][-1]})")
        f = harmonics[:, self._crop[0], self._crop[1]]  # a view of the interior
        na = self.k if n_active is None else n_active
        r = self.n_planes
        ix, iy = self.axis_index
        xz_pp, xz_pm = peak_pressures(f[:na, :, iy], s.workers)
        yz_pp, yz_pm = peak_pressures(f[:na, ix, :], s.workers)
        rows["z"].append(float(z))
        rows["axis"].append(f[:, ix, iy].astype(complex))
        rows["xz_pp"].append(xz_pp)
        rows["xz_pm"].append(xz_pm)
        rows["xz_a"].append(np.abs(f[0, :, iy]))
        rows["yz_pp"].append(yz_pp)
        rows["yz_pm"].append(yz_pm)
        rows["yz_a"].append(np.abs(f[0, ix, :]))
        rows["ax_pp"].append(float(xz_pp[ix]))
        rows["ax_pm"].append(float(xz_pm[ix]))
        imap = (f[:na].real ** 2 + f[:na].imag ** 2).sum(axis=0)  # 2 rho c I
        rows["power"].append(g.dx * g.dy / (2 * self.impedance) * float(imap.sum()))
        imax = float(imap.max())
        rows["pl_i"].append(imax / (2 * self.impedance))
        cx = cy = math.nan
        if imax > 0:
            # beam centre: intensity centroid of the -6 dB region (IEC 61828 beam axis)
            w = np.where(imap >= 0.25 * imax, imap, 0.0)
            tot = w.sum()
            cx = float(g.x @ w.sum(axis=1)) / tot
            cy = float(g.y @ w.sum(axis=0)) / tot
        rows["cen_x"].append(cx)
        rows["cen_y"].append(cy)

        if self._source is None:
            self._source = f.astype(self._dtype)
        d_face = abs(z - (rows["z"][0] + s.face_distance))
        if d_face < self._face["distance"]:
            self._face.update(distance=d_face, index=r)
            self._face["field"] = self._keep(self._face["field"], f)

        pl_pp = pl_pm = math.nan
        if self._max_pp is not None:
            # |p(tau)| <= sum_n |P_n|, so only points where that bound exceeds both the running
            # maximum there and the extremes already found on the two lines through the axis
            # can change the maps or the plane extremes
            bound = np.abs(f[:na]).sum(axis=0)
            floor = min(max(xz_pp.max(), yz_pp.max()), -min(xz_pm.min(), yz_pm.min()))
            cand = (bound > self._max_pp) | (bound > floor)
            cand_pp, cand_pm = peak_pressures(f[:na, cand], s.workers)
            pmax = np.full((g.nx, g.ny), -np.inf)
            pmax[cand] = cand_pp
            np.maximum(self._max_pp, pmax, out=self._max_pp)
            pl_pp = max(pmax.max(), xz_pp.max(), yz_pp.max())
            pl_pm = min(cand_pm.min(initial=0.0), xz_pm.min(), yz_pm.min())
            if pl_pp > self._peak["value"]:
                if pmax.max() >= pl_pp:
                    i, j = np.unravel_index(int(np.argmax(pmax)), pmax.shape)
                elif xz_pp.max() >= yz_pp.max():
                    i, j = int(np.argmax(xz_pp)), iy
                else:
                    i, j = ix, int(np.argmax(yz_pp))
                self._peak.update(value=float(pl_pp), index=r, ij=(int(i), int(j)))
                self._peak["field"] = self._keep(self._peak["field"], f)
        rows["pl_pp"].append(pl_pp)
        rows["pl_pm"].append(pl_pm)

        on_axis = {"p_positive": xz_pp[ix], "p_negative": -xz_pm[ix], "first_harmonic": abs(f[0, ix, iy])}[
            s.focal_quantity
        ]
        if on_axis > self._focus["value"]:
            self._focus.update(value=float(on_axis), index=r)
            self._focus["field"] = self._keep(self._focus["field"], f)
        for p in self._points.values():
            d = abs(z - p["pos"][2])
            if d < p["distance"]:
                i, j = p["ij"]
                p.update(distance=d, z=float(z), value=f[:, i, j].astype(complex))
        if self._slice_cols and self._z_range[0] <= z <= self._z_range[1]:
            self._slice_z.append(float(z))
            for name, cols in self._slice_cols.items():
                line = f[:, cols, iy] if name == "xz" else f[:, ix, cols]
                self._slices[name].append(line.astype(self._dtype))
        if self._xy_cols is not None and s.xy_z is not None:
            d = abs(z - s.xy_z)
            if d < self._xy["distance"]:
                self._xy.update(distance=d, z=float(z))
                self._xy["field"] = self._keep(self._xy["field"], f[:, self._xy_cols[0], self._xy_cols[1]])
        self._last = f

    def _keep(self, buffer: ComplexArray | None, f: ComplexArray) -> ComplexArray:
        """Copy ``f`` into ``buffer`` (allocated on first use, at the planes' precision)."""
        if buffer is None:
            return f.astype(self._dtype)
        buffer[...] = f
        return buffer

    def finish(self, **extra: Any) -> Results:
        """Return the :class:`Results` of the planes recorded.

        Args:
            extra: further :class:`Results` fields (``alpha``, ``focusing_depth``,
                ``godunov_from``, ``timings``, ...).
        """
        if self._last is None:
            raise ValueError("no plane was recorded")
        rows, s, g = self._rows, self.settings, self.grid
        arr = {k: np.asarray(v) for k, v in rows.items()}
        spacing = float(np.max(np.diff(arr["z"]))) if arr["z"].size > 1 else 0.0
        points = {}
        for name, p in self._points.items():
            if p["distance"] > spacing / 2 + 1e-12 * max(1.0, abs(p["z"])):
                log.warning("output point %s at %s is beyond the recorded planes: not written", name, p["pos"])
                continue
            i, j = p["ij"]
            points[name] = ((float(g.x[i]), float(g.y[j]), p["z"]), p["value"])
        peak = s.plane_peaks
        assert self._source is not None
        kept: dict[str, Any] = {}
        if self._slice_cols:
            if not self._slice_z:
                log.warning("no recorded plane lies in the slice extent %s: no slices kept", s.slice_extent)
            else:
                kept["slice_z"] = np.asarray(self._slice_z)
                for name, cols in self._slice_cols.items():
                    # (K, nz, n): harmonics first, as the planes
                    kept[f"{name}_harmonics"] = np.stack(self._slices[name], axis=1)
                    kept[f"{name}_coordinate"] = (g.x if name == "xz" else g.y)[cols].copy()
        if self._xy_cols is not None:
            cx, cy = self._xy_cols
            if s.xy_z is None:  # the plane of the focus
                assert self._focus["field"] is not None
                kept["xy_harmonics"] = np.ascontiguousarray(self._focus["field"][:, cx, cy])
                kept["xy_z"] = float(rows["z"][self._focus["index"]])
            else:
                kept["xy_harmonics"] = self._xy["field"]
                kept["xy_z"] = self._xy["z"]
            kept["xy_x"] = g.x[cx].copy()
            kept["xy_y"] = g.y[cy].copy()
        return Results(
            grid=g,
            z=arr["z"],
            axis=np.asarray(rows["axis"], dtype=complex).reshape(-1, self.k),
            axis_p_plus=arr["ax_pp"],
            axis_p_minus=arr["ax_pm"],
            xz_p_plus=arr["xz_pp"],
            xz_p_minus=arr["xz_pm"],
            yz_p_plus=arr["yz_pp"],
            yz_p_minus=arr["yz_pm"],
            xz_amplitude=arr["xz_a"],
            yz_amplitude=arr["yz_a"],
            plane_p_plus=arr["pl_pp"].astype(float),
            plane_p_minus=arr["pl_pm"].astype(float),
            power=arr["power"],
            plane_intensity=arr["pl_i"],
            max_p_plus=self._max_pp,
            final=self._last.astype(self._dtype),  # a copy: the caller may reuse its buffer
            points=points,
            focal_index=int(self._focus["index"]),
            focal_plane=self._focus["field"],
            peak_index=int(self._peak["index"]) if peak else -1,
            peak_xy_index=self._peak["ij"] if peak else self.axis_index,
            peak_plane=self._peak["field"] if peak else None,
            focal_quantity=s.focal_quantity,
            centroid_x=arr["cen_x"],
            centroid_y=arr["cen_y"],
            source_plane=self._source,
            face_index=int(self._face["index"]),
            face_plane=self._face["field"],
            sound_speed=self.sound_speed,
            levels_db=list(s.levels_db),
            metrics_harmonics=s.metrics_harmonics,
            hydrophone_diameter=s.hydrophone_diameter,
            derating=s.derating,
            impedance=self.impedance,
            source_amplitude=float(np.abs(self._source[0]).max()),
            frequency=self.frequency,
            axis_index=self.axis_index,
            **kept,
            **extra,
        )
