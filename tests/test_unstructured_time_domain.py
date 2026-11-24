"""Time-domain fields on unstructured 3D (and planar 2D) meshes: time series at nodes and in XDMF files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs import synthetic
from acoustic_outputs.datasets import from_dataset
from acoustic_outputs.preprocess.unstructured_frequency_domain import plane_points
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal, write_dolfinx_xdmf


@pytest.fixture(scope="module")
def box() -> GaussianBeam:
    return GaussianBeam(x=np.linspace(-5e-3, 5e-3, 21), y=np.linspace(-5e-3, 5e-3, 21), z=np.linspace(20e-3, 40e-3, 41))


@pytest.fixture(scope="module")
def record(box: GaussianBeam) -> tuple[dict, float, np.ndarray]:
    """A tetrahedral mesh and the time series at its nodes, noise until the wave arrives."""
    mesh = synthetic.unstructured_mesh(box)
    dt = 1 / (box.frequency * 19.3)
    return mesh, dt, synthetic.mesh_time_series(mesh["points"], mesh["values"], box, dt, int(box.duration() / dt))


@pytest.fixture(scope="module")
def harmonics_ref(box: GaussianBeam, record: tuple) -> dict:
    mesh = record[0]
    return ao.from_unstructured_harmonics(mesh["points"], mesh["values"], cells=mesh["cells"], spacing=0.5e-3,
                                          phase="absolute", time_sign=1, **box.medium()).metrics()  # fmt: skip


def test_time_series_at_nodes(box: GaussianBeam, record: tuple, harmonics_ref: dict) -> None:
    """Fitted at the nodes, the series give the metrics of the nodes' harmonics."""
    mesh, dt, series = record
    got = ao.from_unstructured_time_series(mesh["points"], series, dt, cells=mesh["cells"], spacing=0.5e-3,
                                           **box.medium()).metrics()  # fmt: skip
    assert_metrics_equal(harmonics_ref, got, 1e-6)
    data = {**mesh, "values": series, "dt": dt, "spacing": 0.5e-3}
    assert_metrics_equal(got, from_dataset(data, "unstructured").metrics(), 1e-12)
    with pytest.raises(ValueError, match="real"):
        ao.from_unstructured_time_series(mesh["points"], mesh["values"], dt, **box.medium())


def test_planar_2d_time_series() -> None:
    narrow = GaussianBeam(waist=1.5e-3, x=np.linspace(-6e-3, 6e-3, 49), y=None, z=np.linspace(15e-3, 45e-3, 61))
    m = synthetic.unstructured_mesh(narrow, spacing=0.25e-3)
    dt = 1 / (narrow.frequency * 21.1)
    series = synthetic.mesh_time_series(m["points"], m["values"], narrow, dt, int(narrow.duration() / dt))
    ref = ao.from_unstructured_harmonics(m["points"], m["values"], cells=m["cells"], propagation="y", spacing=0.5e-3,
                                         **narrow.medium()).metrics()  # fmt: skip
    got = ao.from_unstructured_time_series(m["points"], series, dt, cells=m["cells"], propagation="y",
                                           spacing=0.5e-3, **narrow.medium()).metrics()  # fmt: skip
    assert_metrics_equal(ref, got, 1e-6)


def test_xdmf_time_series(tmp_path: Path, box: GaussianBeam, record: tuple, harmonics_ref: dict) -> None:
    mesh, dt, series = record
    every = series
    funcs = {"p": [(i * dt, every[:, i]) for i in range(every.shape[1])]}
    path = write_dolfinx_xdmf(tmp_path / "td3d.xdmf", mesh["points"], mesh["cells"], funcs, "Tetrahedron")
    got = ao.from_unstructured_xdmf_time_series(path, spacing=0.5e-3, **box.medium()).metrics()
    assert_metrics_equal(harmonics_ref, got, 1e-6)
    assert_metrics_equal(got, from_dataset(path, spacing=0.5e-3, **box.medium()).metrics(), 1e-12)
    # a window from a time: only its steps are read, the earlier ones may be missing
    start = (every.shape[1] - round(2 / (box.frequency * dt))) * dt  # the last two periods
    with h5py.File(path.with_suffix(".h5"), "a") as f:
        for k in list(f["Function/p"]):
            if float(k.replace("_", ".")) < start - dt / 2:
                del f[f"Function/p/{k}"]
    got2 = ao.from_unstructured_xdmf_time_series(path, window_start=start, spacing=0.5e-3, **box.medium()).metrics()
    assert_metrics_equal(got, got2, 1e-9)


def test_plane_points_order() -> None:
    """The probe points of the planes fill (nz, nx, ny) in C order, the beam along any axis."""
    x, y, z = np.arange(3.0), np.arange(4.0) + 10, np.arange(5.0) + 20
    pts, shape = plane_points(x, y, z)
    assert shape == (5, 3, 4)
    grid = pts.reshape(*shape, 3)
    assert np.all(grid[2, 1, 3] == [1, 13, 22])
    pts, shape = plane_points(x, y, z, propagation="x")  # planes along x: their axes are y, z
    assert shape == (3, 4, 5) and np.all(pts.reshape(*shape, 3)[2, 1, 3] == [2, 11, 23])
    pts, shape = plane_points(x, None, z, propagation="y")  # 2D mesh (x, y = z vector): planes along y
    assert shape == (5, 3, 1) and np.all(pts.reshape(*shape, 3)[4, 2, 0] == [2, 24, 0])
