"""Error norms between simulated and reference values (IEC TS 63587, 3.43–3.46).

The norms compare a set of predicted values x_i with reference values x̂_i at the same
positions: complex amplitudes, magnitudes, or any field quantity on a line, plane or volume.
Bringing the two sets to the same positions (registration, interpolation, axis orientation) is
the caller's task (TS 63587, 5.3.2); :func:`on_common_points` interpolates one sampled line
onto the points of another.

* :func:`l2`: L2 = sqrt(sum |x_i - x̂_i|^2);
* :func:`relative_l2`: L2 / sqrt(sum |x̂_i|^2);
* :func:`linf`: L∞ = max |x_i - x̂_i|;
* :func:`relative_linf`: L∞ / max |x̂_i|.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from acoustic_outputs._types import RealArray


def _pair(predicted: Any, reference: Any) -> tuple[Any, Any]:
    x, ref = np.asarray(predicted), np.asarray(reference)
    if x.shape != ref.shape:
        raise ValueError(f"predicted {x.shape} and reference {ref.shape} values do not match")
    if x.size == 0:
        raise ValueError("no values to compare")
    return x.ravel(), ref.ravel()


def l2(predicted: Any, reference: Any) -> float:
    """L2 error norm (TS 63587, 3.43)."""
    x, ref = _pair(predicted, reference)
    return float(np.sqrt(np.sum(np.abs(x - ref) ** 2)))


def relative_l2(predicted: Any, reference: Any) -> float:
    """Relative L2 error norm (TS 63587, 3.44): L2 over the root-sum-square of the reference."""
    x, ref = _pair(predicted, reference)
    norm = float(np.sqrt(np.sum(np.abs(ref) ** 2)))
    if not norm > 0:
        raise ValueError("the reference values are all zero")
    return l2(x, ref) / norm


def linf(predicted: Any, reference: Any) -> float:
    """L-infinity error norm (TS 63587, 3.45): the largest element-by-element difference."""
    x, ref = _pair(predicted, reference)
    return float(np.max(np.abs(x - ref)))


def relative_linf(predicted: Any, reference: Any) -> float:
    """Relative L-infinity error norm (TS 63587, 3.46): L∞ over the largest reference magnitude."""
    x, ref = _pair(predicted, reference)
    peak = float(np.max(np.abs(ref)))
    if not peak > 0:
        raise ValueError("the reference values are all zero")
    return linf(x, ref) / peak


def error_norms(predicted: Any, reference: Any) -> dict[str, float]:
    """All four norms, keyed ``L2``, ``L2_rel``, ``Linf``, ``Linf_rel``."""
    return {
        "L2": l2(predicted, reference),
        "L2_rel": relative_l2(predicted, reference),
        "Linf": linf(predicted, reference),
        "Linf_rel": relative_linf(predicted, reference),
    }


def on_common_points(
    positions: Any, values: Any, reference_positions: Any, reference_values: Any
) -> tuple[RealArray, Any, Any]:
    """Two sampled lines on the reference positions that both cover: (positions, values, reference).

    ``values`` are interpolated linearly (real and imaginary parts) onto the reference
    positions inside the range of ``positions``; reference points outside it are dropped.
    """
    s, v = np.asarray(positions, dtype=float).ravel(), np.asarray(values).ravel()
    sr, vr = np.asarray(reference_positions, dtype=float).ravel(), np.asarray(reference_values).ravel()
    if s.size != v.size or sr.size != vr.size:
        raise ValueError("positions and values differ in length")
    order = np.argsort(s)
    s, v = s[order], v[order]
    keep = (sr >= s[0] - 1e-12) & (sr <= s[-1] + 1e-12)
    if not keep.any():
        raise ValueError("the lines do not overlap")
    sr, vr = sr[keep], vr[keep]
    if np.iscomplexobj(v):
        vi = np.interp(sr, s, v.real) + 1j * np.interp(sr, s, v.imag)
    else:
        vi = np.interp(sr, s, v)
    return sr, vi, vr
