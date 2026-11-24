"""Frequency-domain fields on unstructured 3D (and planar 2D) meshes: harmonics at nodes and in XDMF files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs import synthetic
from acoustic_outputs.cli import main
from acoustic_outputs.datasets import from_dataset
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal, write_dolfinx_xdmf


@pytest.fixture(scope="module")
def box() -> GaussianBeam:
    """The beam on a box small enough to mesh: 10 x 10 x 20 mm, 0.5 mm (a third of a wavelength)."""
    return GaussianBeam(x=np.linspace(-5e-3, 5e-3, 21), y=np.linspace(-5e-3, 5e-3, 21), z=np.linspace(20e-3, 40e-3, 41))


@pytest.fixture(scope="module")
def box_ref(box: GaussianBeam) -> dict:
    return ao.from_harmonics(box.harmonics(), x=box.x, y=box.y, z=box.z, **box.medium()).metrics()


@pytest.fixture(scope="module")
def tets(box: GaussianBeam) -> dict:
    return synthetic.unstructured_mesh(box)


def close_to_beam(ref: dict, got: dict, rel: float) -> None:
    for k in ("p_positive", "p_negative", "intensity"):
        assert got["focus"][k] == pytest.approx(ref["focus"][k], rel=rel), k
    assert got["focus"]["z"] == pytest.approx(ref["focus"]["z"], abs=0.6e-3)
    for k in ("power_source", "focal_width_x_6dB", "focal_width_y_6dB"):
        assert got[k] == pytest.approx(ref[k], rel=4 * rel), k


# -- the beam on a mesh ----------------------------------------------------------------------------


def test_tetrahedral_mesh(box: GaussianBeam, box_ref: dict, tets: dict) -> None:
    """Absolute-phase harmonics at the nodes of a tetrahedral mesh: the beam, to the mesh's accuracy."""
    res = ao.from_unstructured_harmonics(
        tets["points"], tets["values"], cells=tets["cells"], spacing=0.5e-3, **box.medium()
    )
    assert (res.grid.nx, res.grid.ny, res.z.size) == (21, 21, 41)
    close_to_beam(box_ref, res.metrics(), 1e-2)
    # without the cells (a Delaunay triangulation of the same nodes), and with the other time sign
    conj = ao.from_unstructured_harmonics(tets["points"], tets["values"].conj(), spacing=0.5e-3, **box.medium())
    assert_metrics_equal(res.metrics(), conj.metrics(), 1e-9)


def test_grid_and_propagation(box: GaussianBeam, box_ref: dict, tets: dict) -> None:
    """The same mesh with the beam along x: the axes are (z, x, y) -> planes along x."""
    pts = tets["points"][:, [2, 0, 1]]  # x is the beam
    res = ao.from_unstructured_harmonics(pts, tets["values"], cells=tets["cells"], propagation="x", x=box.z, y=box.x,
                               z=box.y, phase="absolute", time_sign=1, **box.medium())  # fmt: skip
    assert np.allclose(res.z, box.z) and np.allclose(res.grid.x, box.x)
    close_to_beam(box_ref, res.metrics(), 1e-2)


def test_planar_2d_mesh() -> None:
    """A planar 2D mesh (x, z), the beam along z given as the second axis: per metre, as a 2D grid."""
    narrow = GaussianBeam(waist=1.5e-3, x=np.linspace(-6e-3, 6e-3, 49), y=None, z=np.linspace(15e-3, 45e-3, 121))
    ref = ao.from_harmonics(narrow.harmonics(), x=narrow.x, z=narrow.z, **narrow.medium()).metrics()
    m = synthetic.unstructured_mesh(narrow, spacing=0.25e-3)
    pts3 = np.column_stack([m["points"], np.zeros(len(m["points"]))])  # DOLFINx's (x, y, 0)
    res = ao.from_unstructured_harmonics(pts3, m["values"], cells=m["cells"], propagation="y", spacing=0.25e-3,
                               **narrow.medium())  # fmt: skip
    assert res.grid.ny == 1
    got = res.metrics()
    for k in ("p_positive", "p_negative"):
        assert got["focus"][k] == pytest.approx(ref["focus"][k], rel=5e-3)
    assert got["power_source"] == pytest.approx(ref["power_source"], rel=1e-2)
    assert got["focal_width_x_6dB"] == pytest.approx(ref["focal_width_x_6dB"], rel=1e-2)


def test_errors(tets: dict, box: GaussianBeam) -> None:
    with pytest.raises(ValueError, match="time_domain"):
        ao.from_unstructured_harmonics(tets["points"], tets["values"].real, **box.medium())
    with pytest.raises(ValueError, match="propagation"):
        ao.from_unstructured_harmonics(tets["points"][:, :2], tets["values"], propagation="z", **box.medium())
    with pytest.raises(ValueError, match="values for"):
        ao.from_unstructured_harmonics(tets["points"], tets["values"][1:], **box.medium())


# -- files -------------------------------------------------------------------------------------------


def test_xdmf_tetrahedra(tmp_path: Path, box: GaussianBeam, tets: dict) -> None:
    funcs = {f"p{n + 1}": [(0.0, tets["values"][:, n])] for n in range(box.n_harmonics)}
    path = write_dolfinx_xdmf(tmp_path / "bowl.xdmf", tets["points"], tets["cells"], funcs, "Tetrahedron")
    names = [f"p{n + 1}" for n in range(box.n_harmonics)]
    ref = ao.from_unstructured_harmonics(
        tets["points"], tets["values"], cells=tets["cells"], spacing=0.5e-3, **box.medium()
    )
    got = ao.from_unstructured_xdmf_harmonics(path, names, spacing=0.5e-3, **box.medium())
    assert_metrics_equal(ref.metrics(), got.metrics(), 1e-12)
    # a 3D file is recognised as unstructured
    got2 = from_dataset(path, field=names, spacing=0.5e-3, **box.medium())
    assert_metrics_equal(ref.metrics(), got2.metrics(), 1e-12)
    with pytest.raises(ValueError, match="3D"):
        ao.from_axisymmetric_xdmf_harmonics(path, names, **box.medium())


def test_dataset_and_cli(tmp_path: Path, box: GaussianBeam, tets: dict) -> None:
    ref = ao.from_unstructured_harmonics(tets["points"], tets["values"], cells=tets["cells"], **box.medium()).metrics()
    assert_metrics_equal(ref, from_dataset(tets, "unstructured").metrics(), 1e-12)
    data = tmp_path / "mesh.npz"
    assert main(["--example", "unstructured", str(data)]) == 0
    assert main([str(data), "-o", str(tmp_path / "out"), "-q"]) == 0
    assert (tmp_path / "out" / "metrics.json").exists()
