"""Time-domain axisymmetric fields: time series on (r, z) grids, at mesh nodes and in XDMF files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial import Delaunay

import acoustic_outputs as ao
from acoustic_outputs import synthetic
from acoustic_outputs.datasets import from_dataset
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal, metrics_close, write_dolfinx_xdmf


@pytest.fixture(scope="module")
def radial(coarse: GaussianBeam) -> tuple[GaussianBeam, float, np.ndarray]:
    """The coarse beam on (r, z), and its time series (r, z, t): noise until the wave arrives."""
    rb = synthetic.radial_beam(coarse)
    dt = 1 / (coarse.frequency * 20.3)
    return rb, dt, rb.time_series(dt, int(coarse.duration() / dt))[:, 0]


@pytest.fixture(scope="module")
def radial_ref(coarse: GaussianBeam, radial: tuple) -> dict:
    rb = radial[0]
    return ao.from_axisymmetric_harmonics(np.transpose(rb.harmonics()[..., 0], (2, 0, 1)), r=rb.x, z=rb.z,
                                          layout="rzk", phase="retarded", time_sign=1,
                                          **coarse.medium()).metrics()  # fmt: skip


def test_grid_time_series(coarse: GaussianBeam, radial: tuple, radial_ref: dict) -> None:
    rb, dt, p = radial
    got = ao.from_axisymmetric_time_series(p, dt, r=rb.x, z=rb.z, **coarse.medium()).metrics()
    metrics_close(radial_ref, got, 1e-6)
    # another layout, and as a dataset
    tzr = np.transpose(p, (2, 1, 0))
    got2 = ao.from_axisymmetric_time_series(tzr, dt, r=rb.x, z=rb.z, layout="tzr", **coarse.medium()).metrics()
    assert_metrics_equal(got, got2, 1e-12)
    data = {"p": p, "dt": dt, "r": rb.x, "z": rb.z, **coarse.medium()}
    assert_metrics_equal(got, from_dataset(data, "axisymmetric").metrics(), 1e-12)


def test_too_short_a_record_warns(coarse: GaussianBeam, radial: tuple, caplog: pytest.LogCaptureFixture) -> None:
    rb, dt, p = radial
    early = p[..., : int(2.5 / (coarse.frequency * dt))]  # the far planes are still noise
    ao.from_axisymmetric_time_series(early, dt, r=rb.x, z=rb.z, **coarse.medium())
    assert "not a steady periodic field" in caplog.text


def mesh_record(beam: GaussianBeam, dt: float) -> tuple[dict, np.ndarray]:
    """The axisymmetric mesh of ``beam`` and the time series at its nodes."""
    mesh = synthetic.axisymmetric_mesh(beam)
    nt = int(beam.duration() / dt)
    return mesh, synthetic.mesh_time_series(mesh["points"], mesh["values"], beam, dt, nt)


def test_mesh_time_series(beam: GaussianBeam) -> None:
    """Time series at the nodes: fitted there, they give the metrics of the nodes' harmonics."""
    dt = 1 / (beam.frequency * 19.7)
    mesh, series = mesh_record(beam, dt)
    ref = ao.from_axisymmetric_mesh_harmonics(mesh["points"], mesh["values"], cells=mesh["cells"], spacing=0.25e-3,
                                              phase="absolute", time_sign=1, **beam.medium()).metrics()  # fmt: skip
    got = ao.from_axisymmetric_mesh_time_series(mesh["points"], series, dt, cells=mesh["cells"], spacing=0.25e-3,
                                                **beam.medium()).metrics()  # fmt: skip
    assert_metrics_equal(ref, got, 1e-6)
    with pytest.raises(ValueError, match="real"):
        ao.from_axisymmetric_mesh_time_series(mesh["points"], mesh["values"], dt, **beam.medium())


def test_xdmf_time_series(tmp_path: Path, coarse: GaussianBeam, radial: tuple, radial_ref: dict) -> None:
    """A real function written at every step, as a time-domain FE solver writes it."""
    rb, dt, p = radial
    R, Z = np.meshgrid(rb.x, rb.z, indexing="ij")
    pts = np.column_stack([R.ravel(), Z.ravel()])
    series = p.reshape(-1, p.shape[-1])
    funcs = {"p": [(i * dt, series[:, i]) for i in range(series.shape[1])]}
    path = write_dolfinx_xdmf(tmp_path / "td.xdmf", pts, Delaunay(pts).simplices, funcs)
    ref = ao.from_axisymmetric_time_series(p, dt, r=rb.x, z=rb.z, **coarse.medium()).metrics()
    got = ao.from_axisymmetric_xdmf_time_series(path, r=rb.x, z=rb.z, **coarse.medium()).metrics()
    assert_metrics_equal(ref, got, 1e-6)
    # recognised as a time series (one real function at many times) from the file name
    assert_metrics_equal(ref, from_dataset(path, r=rb.x, z=rb.z, **coarse.medium()).metrics(), 1e-6)
    # only the steps of the window are read: the earlier ones may be missing from the file
    keep = round(2 / (coarse.frequency * dt))
    with h5py.File(path.with_suffix(".h5"), "a") as f:
        steps = sorted(f["Function/p"], key=lambda k: float(k.replace("_", ".")))
        for k in steps[: len(steps) - keep]:
            del f[f"Function/p/{k}"]
    got2 = ao.from_axisymmetric_xdmf_time_series(path, r=rb.x, z=rb.z, **coarse.medium()).metrics()
    assert_metrics_equal(ref, got2, 1e-12)
    metrics_close(radial_ref, got, 1e-6)
