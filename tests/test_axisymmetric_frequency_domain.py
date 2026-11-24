"""Frequency-domain axisymmetric fields: harmonics on (r, z) grids, at mesh nodes and in XDMF files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs import synthetic
from acoustic_outputs.cli import main
from acoustic_outputs.datasets import from_dataset
from acoustic_outputs.preprocess.axisymmetric_frequency_domain import Revolver, resample
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal, metrics_close, write_dolfinx_xdmf


@pytest.fixture(scope="module")
def ref(beam: GaussianBeam) -> dict:
    return ao.from_harmonics(beam.harmonics(), x=beam.x, y=beam.y, z=beam.z, **beam.medium()).metrics()


@pytest.fixture(scope="module")
def mesh(beam: GaussianBeam) -> dict:
    return synthetic.axisymmetric_mesh(beam)


def test_revolver_profile() -> None:
    """A smooth even profile is revolved to its values at sqrt(x^2 + y^2); zero beyond r_max."""
    r = np.linspace(0, 1, 21)
    rev = Revolver(r, dx=0.05)
    plane = rev(np.exp(-(r**2) / 0.1)[None].astype(complex))[0]
    rho = np.hypot(rev.x[:, None], rev.x[None, :])
    inside = rho <= 1
    assert np.allclose(plane[inside], np.exp(-(rho[inside] ** 2) / 0.1), atol=2e-4)
    assert np.all(plane[~inside] == 0)
    assert rev.x.size == 41 and rev.x[20] == 0
    with pytest.raises(ValueError, match="beyond"):
        Revolver(r, r_max=2.0)
    with pytest.raises(ValueError, match="increasing"):
        Revolver(r[::-1])


def test_grid_matches_3d(beam: GaussianBeam, ref: dict) -> None:
    """The beam's harmonics on (r, z), revolved, give the metrics of the same beam on (x, y, z)."""
    data = synthetic.axisymmetric_dataset(beam)
    res = ao.from_axisymmetric_harmonics(data["harmonics"], r=data["r"], z=data["z"], layout="rzk", **beam.medium())
    assert res.grid.nx == res.grid.ny == beam.x.size
    metrics_close(ref, res.metrics(), 2e-4)
    assert res.metrics()["power_source"] == pytest.approx(beam.power, rel=1e-4)


def test_absolute_phase_read() -> None:
    fine = GaussianBeam(z=np.linspace(15e-3, 45e-3, 121))
    fine_ref = ao.from_harmonics(fine.harmonics(), x=fine.x, y=fine.y, z=fine.z, **fine.medium()).metrics()
    data = synthetic.axisymmetric_dataset(fine, absolute=True)
    del data["phase"]  # read from the field
    metrics_close(fine_ref, from_dataset(data, "axisymmetric").metrics(), 2e-4)
    conj = {**data, "harmonics": data["harmonics"].conj()}  # exp(-i omega t)
    metrics_close(fine_ref, from_dataset(conj, "axisymmetric").metrics(), 2e-4)


def test_single_frequency_and_layout(beam: GaussianBeam) -> None:
    data = synthetic.axisymmetric_dataset(beam)
    p1 = data["harmonics"][..., 0]  # (r, z)
    res = ao.from_axisymmetric_harmonics(p1.T, r=data["r"], z=data["z"], layout="zr", **beam.medium())
    assert res.grid.n_harmonics == 1
    assert res.metrics()["focus"]["p_positive"] == pytest.approx(beam.amplitudes[0], rel=1e-9)
    with pytest.raises(ValueError, match="radii"):
        ao.from_axisymmetric_harmonics(p1.T, r=data["r"][:-1], z=data["z"], layout="zr", **beam.medium())


def test_mesh_harmonics(beam: GaussianBeam, ref: dict, mesh: dict) -> None:
    """Harmonics at the nodes of an unstructured mesh, resampled linearly: close to the 3D beam."""
    m = ao.from_axisymmetric_mesh_harmonics(mesh["points"], mesh["values"], cells=mesh["cells"], spacing=0.25e-3,
                                            **beam.medium()).metrics()  # fmt: skip
    assert m["focus"]["z"] == pytest.approx(beam.focus, abs=0.3e-3)
    assert m["focus"]["p_positive"] == pytest.approx(ref["focus"]["p_positive"], rel=1e-3)
    assert m["power_source"] == pytest.approx(beam.power, rel=1e-2)
    assert m["focal_width_x_6dB"] == pytest.approx(beam.focal_width(-6), rel=2e-2)
    assert m["focal_width_y_6dB"] == pytest.approx(m["focal_width_x_6dB"], rel=1e-12)
    # a Delaunay triangulation of the nodes (no cells) gives the same here: the domain is convex
    delaunay = ao.from_axisymmetric_mesh_harmonics(mesh["points"], mesh["values"], spacing=0.25e-3,
                                                   **beam.medium()).metrics()  # fmt: skip
    assert_metrics_equal(m, delaunay, 1e-9)
    with pytest.raises(ValueError, match="time_domain"):
        ao.from_axisymmetric_mesh_harmonics(mesh["points"], mesh["values"].real, **beam.medium())


def test_resample_follows_the_mesh() -> None:
    """Points outside the cells are zero, and a linear field is reproduced exactly inside them."""
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]], dtype=float)
    cells = np.array([[0, 1, 3, 2]])  # one quadrilateral, x <= 1; the nodes reach x = 2
    vals = 2 * pts[:, 0] + 3 * pts[:, 1] + 1j
    out = resample(pts, vals, np.array([0.25, 0.75, 1.5]), np.array([0.5]), cells=cells, cell_type="quadrilateral")
    assert np.allclose(out[:2, 0], [2 * 0.25 + 1.5 + 1j, 2 * 0.75 + 1.5 + 1j])
    assert out[2, 0] == 0
    # the third column of 3D geometry (DOLFINx writes x, y, 0) and the zr order
    zr = np.column_stack([pts[:, 1], pts[:, 0], np.zeros(6)])
    out2 = resample(zr, vals, np.array([0.25, 0.75, 1.5]), np.array([0.5]), cells=cells,
                    cell_type="Quadrilateral", coordinates="zr")  # fmt: skip
    assert np.allclose(out, out2)
    with pytest.raises(ValueError, match="r < 0"):
        resample(pts - 1, vals, np.array([0.5]), np.array([0.5]))


def test_xdmf_harmonics(tmp_path: Path, beam: GaussianBeam, mesh: dict) -> None:
    funcs = {f"p{n + 1}": [(0.0, mesh["values"][:, n])] for n in range(beam.n_harmonics)}
    path = write_dolfinx_xdmf(tmp_path / "hifu.xdmf", mesh["points"], mesh["cells"], funcs)
    data = ao.read_xdmf(path)
    assert data.points.shape == (len(mesh["points"]), 3)
    assert sorted(data.functions) == ["p1", "p2", "p3", "p4"]
    assert data.functions["p2"].is_complex
    assert np.allclose(data.functions["p2"].load()[0], mesh["values"][:, 1])
    names = ["p1", "p2", "p3", "p4"]
    ref = ao.from_axisymmetric_mesh_harmonics(mesh["points"], mesh["values"], cells=mesh["cells"], spacing=0.25e-3,
                                              **beam.medium()).metrics()  # fmt: skip
    got = ao.from_axisymmetric_xdmf_harmonics(path, names, spacing=0.25e-3, **beam.medium()).metrics()
    assert_metrics_equal(ref, got, 1e-12)
    # from a file name, recognised as axisymmetric (a 2D mesh) and as harmonics (complex)
    got2 = from_dataset(path, field=names, spacing=0.25e-3, **beam.medium()).metrics()
    assert_metrics_equal(ref, got2, 1e-12)
    with pytest.raises(ValueError, match="name the field"):
        ao.from_axisymmetric_xdmf_harmonics(data, **beam.medium())


def test_cli_example(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    data = tmp_path / "axi.npz"
    assert main(["--example", "axisymmetric", str(data)]) == 0
    with caplog.at_level(logging.INFO):
        assert main([str(data), "-o", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "metrics.json").exists()
