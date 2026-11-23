"""BabelBrain's DataForSim.h5: the complex fundamental on the grid, read as BabelBrain writes it."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest

import acoustic_outputs as ao
from acoustic_outputs import synthetic
from acoustic_outputs.datasets import from_dataset
from acoustic_outputs.preprocess.babelbrain import frequency_from_name, from_babelbrain
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal


@pytest.fixture(scope="module")
def data(beam: GaussianBeam) -> dict:
    return synthetic.babelbrain_dataset(beam)


@pytest.fixture(scope="module")
def ref(beam: GaussianBeam, data: dict) -> ao.Results:
    """The same fundamental given as harmonics on planes, with the target's (brain) medium."""
    return ao.from_harmonics(
        data["p_complex"].astype(complex), layout="xyz", x=beam.x, y=beam.y, z=beam.z, phase="absolute",
        frequency=beam.frequency, sound_speed=1562.0, density=1041.0,
    )  # fmt: skip


def test_matches_the_harmonics(data: dict, ref: ao.Results) -> None:
    res = from_babelbrain(data, frequency=1e6)
    assert res.impedance == pytest.approx(1041.0 * 1562.0)
    assert_metrics_equal(ref.metrics(), res.metrics(), 1e-9)
    m = res.metrics()
    # one frequency: p+ = |p-| = |P1|
    assert m["focus"]["p_positive"] == pytest.approx(-m["focus"]["p_negative"], rel=1e-9)
    assert m["focus"]["p_positive"] == pytest.approx(m["focus"]["harmonics"][0], rel=1e-9)
    assert abs(m["focus"]["z"] - 0.03) <= 1e-3


def test_beam_towards_minus_z(data: dict, ref: ao.Results, caplog: pytest.LogCaptureFixture) -> None:
    """The transducer at the high-z end: the planes are reversed and z becomes -z."""
    flipped = dict(data, p_complex=data["p_complex"][:, :, ::-1], MaterialMap=data["MaterialMap"][:, :, ::-1])
    flipped["TargetLocation"] = data["TargetLocation"] * [1, 1, -1] + [0, 0, data["z_vec"].size - 1]
    with caplog.at_level(logging.INFO, logger="acoustic_outputs.preprocess.babelbrain"):
        res = from_babelbrain(flipped, frequency=1e6)
    assert "along -z" in caplog.text
    a, b = ref.metrics(), res.metrics()
    assert b["focus"]["z"] == pytest.approx(-(data["z_vec"][0] + data["z_vec"][-1]) + a["focus"]["z"])
    for key in ("focal_width_x_6dB", "focal_width_y_6dB", "focal_length_6dB", "power_source"):
        assert b[key] == pytest.approx(a[key], rel=1e-9, nan_ok=True)
    assert b["focus"]["p_positive"] == pytest.approx(a["focus"]["p_positive"], rel=1e-9)


def write_like_babelbrain(path: Path, data: dict) -> None:
    """What BabelViscoFDTD's H5pySimple.SaveToH5py writes: one dataset per key with a 'type'
    attribute, larger arrays compressed (gzip here; blosc by default, read with hdf5plugin), and
    dicts as groups."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            if k == "frequency":
                continue  # BabelBrain does not store it
            v = np.asarray(v)
            kw = {"compression": "gzip", "chunks": True, "shuffle": True} if v.nbytes > 1024 else {}
            ds = f.create_dataset(k, data=v, **kw)
            ds.attrs["type"] = "ndarray" if v.ndim else "scalar"
        f.create_dataset("TxSystem", data="H317").attrs["type"] = "<class 'str'>"
        g = f.create_group("TxMechanicalAdjustment")
        g.attrs["type"] = "dict"
        g.create_dataset("X", data=0.0).attrs["type"] = "scalar"


def test_file_as_babelbrain_writes_it(tmp_path: Path, data: dict, ref: ao.Results) -> None:
    path = tmp_path / "target_1000kHz_6PPW_DataForSim.h5"
    write_like_babelbrain(path, data)
    res = from_dataset(path)  # the kind and the frequency are recognised
    assert res.frequency == 1e6
    assert_metrics_equal(ref.metrics(), res.metrics(), 1e-6)  # stored in single precision
    assert from_babelbrain(path).metrics()["focus"]["z"] == res.metrics()["focus"]["z"]


def test_blosc_file(tmp_path: Path, data: dict, ref: ao.Results) -> None:
    """BabelBrain's default compression."""
    h5py = pytest.importorskip("h5py")
    hdf5plugin = pytest.importorskip("hdf5plugin")
    path = tmp_path / "target_1000kHz_6PPW_DataForSim.h5"
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            if k != "frequency":
                v = np.asarray(v)
                f.create_dataset(k, data=v, **(hdf5plugin.Blosc(clevel=9) if v.nbytes > 1024 else {}))
    assert_metrics_equal(ref.metrics(), from_dataset(path).metrics(), 1e-6)


def test_choices_and_errors(data: dict, caplog: pytest.LogCaptureFixture) -> None:
    assert frequency_from_name("a/b/T1_250kHz_6PPW_DataForSim.h5") == 250e3
    assert frequency_from_name("DataForSim.h5") is None
    with pytest.raises(ValueError, match="frequency"):
        from_babelbrain(data)
    # magnitude only: usable, with a warning
    amp = {k: v for k, v in data.items() if k != "p_complex"}
    with caplog.at_level(logging.WARNING, logger="acoustic_outputs.preprocess.babelbrain"):
        res = from_babelbrain(amp, frequency=1e6)
    assert "no phase" in caplog.text
    assert res.metrics()["focus"]["p_positive"] == pytest.approx(float(data["p_amp"].max()), rel=0.05)
    # medium given, and the water-only field missing
    res = from_babelbrain(data, frequency=1e6, density=1000.0, sound_speed=1500.0)
    assert res.impedance == 1.5e6
    with pytest.raises(KeyError, match="p_complex_water"):
        from_babelbrain(data, frequency=1e6, field="p_complex_water")
    # no target: the plane centre and water
    res = from_babelbrain({k: v for k, v in data.items() if k != "TargetLocation"}, frequency=1e6)
    assert res.impedance == 1.5e6 and not math.isnan(res.metrics()["focus"]["p_positive"])
