"""What the mesh readers share: cells in either node order, point location, and the phase convention."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay

from acoustic_outputs import synthetic
from acoustic_outputs.preprocess._mesh import Mesh, node_phase_convention, simplices
from acoustic_outputs.synthetic import GaussianBeam


@pytest.fixture(scope="module")
def box() -> GaussianBeam:
    return GaussianBeam(x=np.linspace(-5e-3, 5e-3, 21), y=np.linspace(-5e-3, 5e-3, 21), z=np.linspace(20e-3, 40e-3, 41))


@pytest.fixture(scope="module")
def tets(box: GaussianBeam) -> dict:
    return synthetic.unstructured_mesh(box)


def linear(p: np.ndarray) -> np.ndarray:
    return 1 + 2 * p[:, 0] - 3 * p[:, 1] + 0.5 * p[:, -1] + 1j


def unit_hex() -> np.ndarray:
    """The corners of [0, 1]^3 in tensor-product (DOLFINx) order: x fastest."""
    return np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=float)


@pytest.mark.parametrize("order", ["vtk", "dolfinx"])
def test_hexahedra_either_order(order: str) -> None:
    """Two hexahedra side by side: a linear field is exact inside, zero outside, in either node order."""
    corners = unit_hex()
    pts = np.concatenate([corners, corners + np.array([1.0, 0, 0])])
    perm = [0, 1, 3, 2, 4, 5, 7, 6] if order == "vtk" else list(range(8))
    cells = np.array([perm, [p + 8 for p in perm]])
    q = np.array([[0.3, 0.6, 0.2], [1.7, 0.1, 0.9], [2.5, 0.5, 0.5], [0.5, -0.1, 0.5]])
    loc = Mesh(pts, cells, "hexahedron").locate(q)
    assert loc.found.tolist() == [True, True, False, False]
    out = loc(linear(pts))
    assert np.allclose(out[:2], linear(q[:2]))
    assert np.all(out[2:] == 0)
    assert simplices(cells, "hexahedron", pts).shape == (12, 4)


def test_tetrahedra_and_delaunay_agree() -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (300, 3))
    cells = Delaunay(pts).simplices
    q = rng.uniform(0.2, 0.8, (500, 3))
    a = Mesh(pts, cells, "Tetrahedron").locate(q)
    b = Mesh(pts, max_edge=np.inf).locate(q)
    assert a.found.all() and b.found.all()
    assert np.allclose(a(linear(pts)), linear(q)) and np.allclose(b(linear(pts)), linear(q))
    # high-order cells: the corners come first, the rest are ignored
    p2 = np.concatenate([cells, np.zeros((len(cells), 6), dtype=int)], axis=1)
    assert np.allclose(Mesh(pts, p2, "Tetrahedron_10").locate(q)(linear(pts)), linear(q))


def test_delaunay_does_not_bridge_a_gap() -> None:
    """Nodes on two slabs with a gap between: the long Delaunay tetrahedra across it are dropped."""
    g = np.linspace(0, 1, 6)
    slab = np.stack(np.meshgrid(g, g, np.linspace(0, 0.4, 3), indexing="ij"), -1).reshape(-1, 3)
    pts = np.concatenate([slab, slab + np.array([0, 0, 2.6])])
    loc = Mesh(pts).locate(np.array([[0.5, 0.5, 0.3], [0.5, 0.5, 1.5]]))
    assert loc.found.tolist() == [True, False]


def test_phase_convention_at_nodes(box: GaussianBeam, tets: dict) -> None:
    pts, h = tets["points"], tets["values"][:, 0]
    f, c = box.frequency, box.sound_speed
    assert node_phase_convention(pts, h, 2, f, c) == ("absolute", 1)
    assert node_phase_convention(pts, h.conj(), 2, f, c) == ("absolute", -1)
    retarded = box.harmonics_at(pts[:, 0], pts[:, 1], pts[:, 2])[0]
    assert node_phase_convention(pts, retarded, 2, f, c) == ("retarded", 0)
    assert node_phase_convention(pts * 10, h, 2, f, c) is None  # 5 mm apart: too far to tell
