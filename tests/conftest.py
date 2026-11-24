"""Shared fixtures: a synthetic focused Gaussian beam (known metrics) and its results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs.synthetic import GaussianBeam


def assert_metrics_equal(a: Any, b: Any, rtol: float, path: str = "") -> None:
    """Every number of two metrics dicts agrees to ``rtol`` (with a small absolute floor)."""
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            assert_metrics_equal(a[k], b[k], rtol, f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b, strict=True)):
            assert_metrics_equal(u, v, rtol, f"{path}[{i}]")
    elif isinstance(a, (bool, str)) or a is None:
        assert a == b, (path, a, b)
    else:
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), (path, a, b)
        else:
            assert b == pytest.approx(a, rel=rtol, abs=rtol * 1e-3), (path, a, b)


def metrics_close(ref: Any, got: Any, rtol: float) -> None:
    """As assert_metrics_equal, less the quantities that are zero by symmetry (round-off)."""
    zero = ("x", "y", "x0", "y0", "slope_x", "slope_y", "tilt")

    def strip(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k not in zero}
        return d

    assert_metrics_equal(strip(ref), strip(got), rtol)


def write_dolfinx_xdmf(
    path: Path, points: np.ndarray, cells: np.ndarray, functions: dict, topology: str = "Triangle"
) -> Path:
    """An XDMF file laid out as DOLFINx's XDMFFile writes a mesh and functions.

    ``functions`` maps a name to a list of (time, values); complex values are written as
    ``real_<name>`` and ``imag_<name>``.
    """
    import h5py

    h5 = path.with_suffix(".h5")
    n, (m, per) = len(points), cells.shape
    geom = points if points.shape[1] == 3 else np.column_stack([points, np.zeros(n)])
    atts: list[tuple[str, float, np.ndarray]] = []
    for name, items in functions.items():
        for t, v in items:
            if np.iscomplexobj(v):
                atts += [(f"real_{name}", t, v.real), (f"imag_{name}", t, v.imag)]
            else:
                atts.append((name, t, v))
    with h5py.File(h5, "w") as f:
        f["Mesh/mesh/geometry"] = geom
        f["Mesh/mesh/topology"] = cells.astype(np.int64)
        for name, t, v in atts:
            f[f"Function/{name}/{str(t).replace('.', '_')}"] = v[:, None]
    times = sorted({t for _, t, _ in atts})
    grids = []
    for t in times:
        a = "".join(
            f'<Attribute Name="{name}" AttributeType="Scalar" Center="Node"><DataItem Dimensions="{n} 1" '
            f'Format="HDF">{h5.name}:/Function/{name}/{str(t).replace(".", "_")}</DataItem></Attribute>'
            for name, tt, _ in atts
            if tt == t
        )
        grids.append(
            f'<Grid Name="f" GridType="Uniform"><xi:include xpointer="xpointer(/Xdmf/Domain/'
            f"Grid[@GridType='Uniform'][1]/*[self::Topology or self::Geometry])\" />"
            f'<Time Value="{t!r}" />{a}</Grid>'
        )
    path.write_text(
        '<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
        '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude"><Domain>'
        f'<Grid Name="mesh" GridType="Uniform"><Topology TopologyType="{topology}" NumberOfElements="{m}" '
        f'NodesPerElement="{per}"><DataItem Dimensions="{m} {per}" NumberType="Int" Format="HDF">{h5.name}:'
        f'/Mesh/mesh/topology</DataItem></Topology><Geometry GeometryType="XYZ"><DataItem '
        f'Dimensions="{n} 3" Format="HDF">{h5.name}:/Mesh/mesh/geometry</DataItem></Geometry></Grid>'
        '<Grid Name="f" GridType="Collection" CollectionType="Temporal">' + "".join(grids) + "</Grid>"
        "</Domain></Xdmf>"
    )
    return path


@pytest.fixture(scope="session")
def beam() -> GaussianBeam:
    """The default beam: 81 x 81 points of 0.25 mm, 31 planes from 15 to 45 mm, focus at 30 mm."""
    return GaussianBeam()


@pytest.fixture(scope="session")
def coarse() -> GaussianBeam:
    """The beam on 33 x 33 points and 11 planes, small enough to hold as time series."""
    return GaussianBeam().coarse()


@pytest.fixture(scope="session")
def coarse_ref(coarse: GaussianBeam) -> ao.Results:
    """The coarse beam's results from its harmonics: what every time-series route must give."""
    return ao.from_harmonics(coarse.harmonics(), x=coarse.x, y=coarse.y, z=coarse.z, **coarse.medium())


@pytest.fixture(scope="session")
def flat() -> GaussianBeam:
    """A 2D beam (no y), narrow enough that its -6 dB length fits the planes."""
    return GaussianBeam(waist=1.5e-3, x=np.linspace(-16e-3, 16e-3, 129), y=None, z=np.linspace(5e-3, 55e-3, 51))
