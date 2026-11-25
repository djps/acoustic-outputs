"""Datasets of the three kinds of solver read from .npz, .mat and HDF5 files, and the command line."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

import acoustic_outputs as ao
from acoustic_outputs import cli, synthetic
from acoustic_outputs.datasets import KINDS, from_dataset, load_dataset, save_dataset
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal


def datasets(coarse: GaussianBeam, flat: GaussianBeam) -> dict[str, dict]:
    small_flat = GaussianBeam(waist=flat.waist, x=flat.x[::2], y=None, z=flat.z[::5])
    return {
        "harmonics": synthetic.harmonics_dataset(coarse),
        "time-series": synthetic.time_series_dataset(coarse),
        "kwave": synthetic.kwave_dataset(coarse),
        "fdtd": synthetic.fdtd_dataset(small_flat),
        "fullwave": synthetic.fullwave_dataset(coarse),
        "babelbrain": synthetic.babelbrain_dataset(coarse),
        "msound": synthetic.msound_dataset(coarse),
        "focus": synthetic.focus_dataset(coarse, "transient"),
        "axisymmetric": synthetic.axisymmetric_dataset(coarse),
        "unstructured": synthetic.unstructured_mesh(coarse, spacing=0.625e-3),
        "optimus": {**synthetic.optimus_dataset(coarse), **coarse.medium()},
    }


@pytest.fixture(scope="module")
def data(coarse: GaussianBeam, flat: GaussianBeam) -> dict[str, dict]:
    return datasets(coarse, flat)


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_from_npz(kind: str, data: dict, coarse_ref: ao.Results, tmp_path: Path) -> None:
    path = save_dataset(tmp_path / f"{kind}.npz", kind, **data[kind])
    res = from_dataset(path)
    if kind == "fdtd":
        assert res.grid.ny == 1 and res.metrics()["focus"]["z"] == pytest.approx(30e-3)
    elif kind == "babelbrain":  # the fundamental only, attenuated, in brain tissue (see test_babelbrain.py)
        assert res.impedance == pytest.approx(1041 * 1562) and res.metrics()["focus"]["z"] == pytest.approx(30e-3)
    elif kind == "msound":  # the fundamental and the second harmonic (see test_msound_focus.py)
        assert res.grid.n_harmonics == 2 and res.metrics()["focus"]["z"] == pytest.approx(30e-3)
    elif kind == "optimus":  # the fundamental only (see test_optimus.py)
        assert res.grid.n_harmonics == 1
        assert res.metrics()["focus"]["p_positive"] == pytest.approx(GaussianBeam().amplitudes[0], rel=1e-9)
    elif kind == "unstructured":  # resampled from a mesh 0.625 mm apart (see test_unstructured.py)
        ref, got = coarse_ref.metrics(), res.metrics()
        assert got["focus"]["p_positive"] == pytest.approx(ref["focus"]["p_positive"], rel=3e-2)
        assert got["focus"]["z"] == pytest.approx(30e-3, abs=1e-3)
    elif kind == "axisymmetric":  # revolved, so interpolated between radii (see test_axisymmetric.py)
        ref, got = coarse_ref.metrics(), res.metrics()
        assert got["focus"]["p_positive"] == pytest.approx(ref["focus"]["p_positive"], rel=1e-9)
        assert got["power_source"] == pytest.approx(ref["power_source"], rel=1e-3)
    else:
        assert_metrics_equal(coarse_ref.metrics(), res.metrics(), 1e-8)


def test_mat_file_with_kwave_structs(data: dict, coarse_ref: ao.Results, tmp_path: Path) -> None:
    """A .mat file as MATLAB k-Wave leaves it: kgrid and sensor structs, sensor_data.p (column-major)."""
    d = data["kwave"]
    mask = d["sensor_mask"]
    block = tuple(slice(int(i.min()), int(i.max()) + 1) for i in np.nonzero(mask))
    sub = tuple(s.stop - s.start for s in block)
    p_f = d["p"].reshape(*sub, -1).reshape(-1, d["p"].shape[1], order="F")
    path = tmp_path / "kwave.mat"
    savemat(path, {
        "kind": "kwave",
        "kgrid": {**{f"{c}_vec": d[f"{c}_vec"][:, None] for c in "xyz"}, "dt": d["dt"]},
        "sensor": {"mask": mask.astype(np.uint8), "record_start_index": d["record_start_index"]},
        "sensor_data": {"p": p_f},
        "medium": {"sound_speed": 1500.0, "density": 1000.0},
        "f0": 1e6, "order": "F", "n_harmonics": 4,
    })  # fmt: skip
    loaded = load_dataset(path)
    assert {"x_vec", "mask", "p", "sound_speed"} <= set(loaded)
    res = from_dataset(path)
    assert_metrics_equal(coarse_ref.metrics(), res.metrics(), 1e-8)


def test_hdf5_file(data: dict, coarse_ref: ao.Results, tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    d = data["fullwave"]
    path = tmp_path / "fullwave.h5"
    with h5py.File(path, "w") as f:
        f.attrs["kind"] = "fullwave"
        for k, v in d.items():
            if isinstance(v, str):
                f.attrs[k] = v  # strings as attributes, as HDF5 writers usually keep them
            else:
                f.create_dataset(k, data=np.asarray(v))
    res = from_dataset(path)
    assert_metrics_equal(coarse_ref.metrics(), res.metrics(), 1e-8)


def test_overrides_and_settings(data: dict, coarse_ref: ao.Results) -> None:
    d = dict(data["harmonics"])
    d.pop("density")
    with pytest.raises(KeyError, match="density"):
        from_dataset(d, "harmonics")
    res = from_dataset(d, "harmonics", density=1000.0, levels_db=[-3.0], focal_quantity="first_harmonic",
                       points=[("p.txt", (0.0, 0.0, 0.03))])  # fmt: skip
    m = res.metrics()
    assert m["focal_quantity"] == "first_harmonic" and "focal_width_x_3dB" in m and "focal_width_x_6dB" not in m
    assert "p.txt" in res.points
    with pytest.raises(ValueError, match="kind"):
        from_dataset(d, density=1000.0)
    with pytest.raises(ValueError, match="unknown kind"):
        from_dataset(d, "optical", density=1000.0)


def test_cli(data: dict, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = save_dataset(tmp_path / "kw.npz", "kwave", **data["kwave"])
    out = tmp_path / "out"
    assert (
        cli.main([str(path), "-o", str(out), "--set", "levels_db=[-6, -3]", "--point", "p.txt", "0", "0", "0.03"]) == 0
    )
    m = json.loads((out / "metrics.json").read_text())
    assert m["focus"]["z"] == pytest.approx(0.03) and "focal_width_x_3dB" in m
    assert (out / "p.txt").exists() and (out / "axial.txt").exists()
    # a missing key is an error message, not a traceback
    bad = save_dataset(tmp_path / "bad.npz", "kwave", **{k: v for k, v in data["kwave"].items() if k != "dt"})
    assert cli.main([str(bad), "-o", str(out), "-q"]) == 1
    assert "dt" in capsys.readouterr().err


@pytest.mark.parametrize("kind", KINDS)
def test_cli_examples(kind: str, tmp_path: Path) -> None:
    path = tmp_path / f"{kind}.npz"
    assert cli.main(["--example", kind, str(path), "-q"]) == 0
    assert cli.main([str(path), "-o", str(tmp_path / "out"), "-q"]) == 0
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert m["focus"]["z"] == pytest.approx(0.03, abs=1e-3)
