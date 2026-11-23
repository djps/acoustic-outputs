"""Output of BabelBrain (transcranial focused-ultrasound planning on BabelViscoFDTD).

BabelBrain's acoustic step runs BabelViscoFDTD, a viscoelastic FDTD solver, to a steady state.
It saves the field in ``<target>_<f>kHz_<ppw>PPW_..._DataForSim.h5``, written with
BabelViscoFDTD's ``H5pySimple.SaveToH5py`` (h5py datasets, blosc-compressed by default, so
reading them needs ``hdf5plugin``). The keys used here are:

* ``p_complex``: complex amplitude of the fundamental on the (x, y, z) grid, z along the
  beam. BabelBrain gets it from an FFT of the steady sensor data, so p = Re(P exp(+i omega t))
  in absolute phase. ``p_complex_refocus`` and ``p_complex_water`` hold the refocused and the
  water-only fields; ``p_amp`` and its variants hold |P| only.
* ``x_vec``, ``y_vec``, ``z_vec``: coordinates [m]; ``TargetLocation``: voxel indices of the
  target.
* ``Material``: rows of (density [kg/m^3], longitudinal speed [m/s], shear speed, longitudinal
  attenuation [Np/m], shear attenuation); ``MaterialMap``: the material index of each voxel.

The frequency is not stored in the file. :func:`from_babelbrain` reads it from the
``_<f>kHz_`` part of the file name, or takes it as an argument.

Only the fundamental is simulated, so p+ = |p-| = |P_1|, and there are no nonlinear
parameters. The plane-wave intensity and the power use one impedance, rho c of the material at
the target (brain tissue, by default), and so are approximate in skull and skin. The beam axis
runs through the target. The planes run from the transducer: if the field shows that the beam
travels towards -z (more power on the last plane than the first), the planes are reversed and
z becomes -z.

The normalised ``*_NORM.nii.gz`` files that BabelBrain also writes are scaled to a maximum of 1,
so they carry no absolute values; use ``DataForSim.h5``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from acoustic_outputs._types import RealArray
from acoustic_outputs.preprocess.spectral import from_harmonics
from acoustic_outputs.recorder import RecordSettings
from acoustic_outputs.results import Results

log = logging.getLogger(__name__)

FIELDS = ("p_complex", "p_complex_refocus", "p_complex_water")
Direction = Literal["auto", "+z", "-z"]


def frequency_from_name(name: str | Path) -> float | None:
    """The frequency [Hz] from BabelBrain's ``_<f>kHz_`` file-name part (None if absent)."""
    m = re.search(r"_(\d+(?:\.\d+)?)kHz_", Path(name).name)
    return float(m.group(1)) * 1e3 if m else None


def is_babelbrain(data: Mapping[str, Any]) -> bool:
    """Whether the keys look like BabelBrain's ``DataForSim.h5``."""
    return "MaterialMap" in data and "x_vec" in data and any(k in data for k in (*FIELDS, "p_amp"))


def from_babelbrain(
    data: Mapping[str, Any] | str | Path,
    *,
    frequency: float | None = None,
    field: str = "p_complex",
    density: float | None = None,
    sound_speed: float | None = None,
    direction: Direction = "auto",
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of a BabelBrain acoustic simulation (its ``DataForSim.h5``).

    Args:
        data: the file, or its keys (e.g. from :func:`acoustic_outputs.datasets.load_dataset`).
        frequency: the frequency [Hz]; default from the file name.
        field: which field: ``"p_complex"``, ``"p_complex_refocus"`` or ``"p_complex_water"``.
            If only the magnitude (``p_amp...``) is stored, it is used with zero phase, which
            is enough for magnitudes but makes the exact intensity wrong.
        density, sound_speed: rho0 and c0 for the intensities; default the material at the
            target.
        direction: the direction of propagation along z (``"auto"`` decides from the power on
            the first and last planes).
        settings, axis_xy, alpha, focusing_depth: as in
            :func:`acoustic_outputs.preprocess.spectral.from_harmonics`; ``axis_xy`` defaults to
            the target.
    """
    if isinstance(data, (str, Path)):
        from acoustic_outputs.datasets import load_dataset

        frequency = frequency if frequency is not None else frequency_from_name(data)
        data = load_dataset(data)
    if frequency is None:
        raise ValueError("the frequency is not in the file name (_<f>kHz_): give it")
    if field in data:
        p = np.asarray(data[field])
    else:
        amp = field.replace("p_complex", "p_amp")
        if amp not in data:
            raise KeyError(f"no {field!r} (or {amp!r}) in the BabelBrain data")
        log.warning("only %s is stored: the field has no phase, so the exact intensity is not valid", amp)
        p = np.asarray(data[amp]).astype(complex)
    x, y, z = (np.asarray(data[f"{c}_vec"], dtype=float).ravel() for c in "xyz")
    if p.shape != (x.size, y.size, z.size):
        raise ValueError(f"{field} of shape {p.shape} does not fit the coordinates ({x.size}, {y.size}, {z.size})")
    if z.size > 1 and z[1] < z[0]:  # coordinates stored decreasing
        z, p = z[::-1], p[:, :, ::-1]

    target = None
    if "TargetLocation" in data:
        target = tuple(round(float(v)) for v in np.asarray(data["TargetLocation"]).ravel()[:3])
        if not all(0 <= t < n for t, n in zip(target, p.shape, strict=True)):
            log.warning("TargetLocation %s is outside the grid %s: ignored", target, p.shape)
            target = None
    rho, c = _target_medium(data, target)
    density = density if density is not None else rho
    sound_speed = sound_speed if sound_speed is not None else c

    if direction == "auto":
        first, last = (float((np.abs(p[:, :, i]) ** 2).sum()) for i in (0, -1))
        direction = "-z" if last > first else "+z"
        log.info("BabelBrain field: beam taken to travel along %s (power on the end planes %.3g, %.3g)",
                 direction, first, last)  # fmt: skip
    if direction == "-z":
        z, p = -z[::-1], p[:, :, ::-1]
    if axis_xy is None and target is not None:
        axis_xy = (float(x[target[0]]), float(y[target[1]]))
    if target is not None:
        peak = np.unravel_index(int(np.argmax(np.abs(p))), p.shape)
        log.info(
            "target at (%.1f, %.1f) mm; field maximum at (%.1f, %.1f, %.1f) mm",
            *(1e3 * v for v in (x[target[0]], y[target[1]], x[peak[0]], y[peak[1]], z[peak[2]])),
        )
    return from_harmonics(
        p,
        layout="xyz",
        x=x,
        y=y,
        z=z,
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        phase="absolute",
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )


def _target_medium(data: Mapping[str, Any], target: tuple[int, ...] | None) -> tuple[float, float]:
    """(density, longitudinal speed) of the material at the target; water if unknown."""
    if "Material" in data and "MaterialMap" in data and target is not None:
        table = np.atleast_2d(np.asarray(data["Material"], dtype=float))
        index = int(np.asarray(data["MaterialMap"])[target])
        if 0 <= index < table.shape[0]:
            return float(table[index, 0]), float(table[index, 1])
    log.info("no material at the target: water (1000 kg/m^3, 1500 m/s) is used for the intensities")
    return 1000.0, 1500.0
