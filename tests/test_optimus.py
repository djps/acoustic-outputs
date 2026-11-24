"""OptimUS output: fields at points from postprocess objects or a .mat export, 3D or a plane through the axis."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat, savemat

import acoustic_outputs as ao
from acoustic_outputs import synthetic
from acoustic_outputs.cli import main
from acoustic_outputs.datasets import from_dataset
from acoustic_outputs.preprocess.optimus import beam_frame, gridded
from acoustic_outputs.synthetic import GaussianBeam
from conftest import assert_metrics_equal, metrics_close


@pytest.fixture(scope="module")
def fundamental(beam: GaussianBeam) -> dict:
    """The metrics of the beam's fundamental on its 3D grid: what OptimUS's field must give."""
    return ao.from_harmonics(beam.harmonics()[:, :1], x=beam.x, y=beam.y, z=beam.z, **beam.medium()).metrics()


def model_of(beam: GaussianBeam, axis: tuple = (1, 0, 0), location: tuple = (0, 0, 0), roc: float | None = None):
    """An object with the attributes of an OptimUS model that the reader uses."""
    source = SimpleNamespace(frequency=beam.frequency, source_axis=np.array(axis, dtype=float),
                             location=np.array(location, dtype=float))  # fmt: skip
    if roc is not None:
        source.radius_of_curvature = roc
    material = SimpleNamespace(density=beam.density, speed_of_sound=beam.sound_speed)
    return SimpleNamespace(source=source, frequency=beam.frequency, material_exterior=material)


def planes_of(data: dict, n_planes: int) -> list:
    """The export split into postprocess-like objects, one per plane across the beam."""
    pts, tot, inc = (np.array_split(data[k], n_planes, axis=-1) for k in ("points", "total_field", "incident_field"))
    return [SimpleNamespace(points=p, field=SimpleNamespace(total_field=t, incident_field=i))
            for p, t, i in zip(pts, tot, inc, strict=True)]  # fmt: skip


def test_beam_frame() -> None:
    assert beam_frame("x") == (0, 1) and beam_frame("-z") == (2, -1) and beam_frame((0, -2, 0)) == (1, -1)
    with pytest.raises(ValueError, match="not along"):
        beam_frame((1, 1, 0))
    with pytest.raises(ValueError, match="regular grid"):
        gridded(np.random.default_rng(0).uniform(size=(3, 20)), np.zeros(20), 0, 1, np.zeros(3))


def test_volume_from_postprocess_objects(beam: GaussianBeam, fundamental: dict) -> None:
    """VisualisePlane objects at each offset along +x (OptimUS's default axis): the 3D beam, exactly."""
    data = synthetic.optimus_dataset(beam)
    res = ao.from_optimus(planes_of(data, beam.z.size), model=model_of(beam))
    assert res.grid.n_harmonics == 1 and (res.grid.nx, res.grid.ny) == (beam.x.size, beam.y.size)
    assert_metrics_equal(fundamental, res.metrics(), 1e-9)


@pytest.mark.parametrize("propagation", ["-z", "y"])
def test_other_axes_and_location(beam: GaussianBeam, fundamental: dict, propagation: str) -> None:
    loc = (2e-3, -1e-3, 5e-3)
    data = synthetic.optimus_dataset(beam, propagation=propagation, location=loc)
    axis = np.zeros(3)
    axis["xyz".index(propagation[-1])] = -1 if propagation.startswith("-") else 1
    res = ao.from_optimus(points=data["points"], field=data["total_field"], model=model_of(beam, axis, loc))
    assert_metrics_equal(fundamental, res.metrics(), 1e-9)


def test_plane_through_the_axis(beam: GaussianBeam, fundamental: dict, caplog: pytest.LogCaptureFixture) -> None:
    """One VisualisePlane through the axis: revolved, close to the 3D beam; its halves agree."""
    data = synthetic.optimus_dataset(beam, "plane")
    pp = SimpleNamespace(points=data["points"], field=SimpleNamespace(total_field=data["total_field"]))
    with caplog.at_level(logging.INFO):
        res = ao.from_optimus(pp, model=model_of(beam))
    assert "halves agree" in caplog.text
    metrics_close(fundamental, res.metrics(), 2e-4)
    # read as a 2D field instead: per metre, the y widths NaN
    flat = ao.from_optimus(pp, model=model_of(beam), geometry="planar")
    assert flat.grid.ny == 1 and np.isnan(flat.metrics()["focal_width_y_6dB"])


def test_asymmetric_plane_warns(beam: GaussianBeam, caplog: pytest.LogCaptureFixture) -> None:
    data = synthetic.optimus_dataset(beam, "plane")
    x = data["points"][1]
    field = data["total_field"] * (1 + 0.2 * (x > 0))  # one half 20% stronger
    ao.from_optimus(points=data["points"], field=field, model=model_of(beam))
    assert "not axisymmetric" in caplog.text


def test_planes_behind_the_source_left_out(beam: GaussianBeam, fundamental: dict) -> None:
    """A source at x = 20 mm: the planes from 15 to 20 mm, behind it, are dropped."""
    data = synthetic.optimus_dataset(beam, location=(-0.0, 0, 0))
    model = model_of(beam, location=(20e-3, 0, 0))
    res = ao.from_optimus(points=data["points"], field=data["total_field"], model=model)
    assert res.z[0] == pytest.approx(0.0, abs=1e-12) and res.z.size == 26  # z from the source
    assert res.metrics()["focus"]["z"] == pytest.approx(10e-3)
    # the focusing depth of a bowl: its radius of curvature
    bowl = ao.from_optimus(points=data["points"], field=data["total_field"], model=model_of(beam, roc=30e-3))
    assert bowl.metrics()["focal_shift"] == pytest.approx(0.0, abs=1e-12)  # focus at 30 mm from the source


def test_errors(beam: GaussianBeam) -> None:
    data = synthetic.optimus_dataset(beam)
    n = beam.x.size * beam.y.size
    one = SimpleNamespace(points=data["points"][:, :n], field=SimpleNamespace(total_field=data["total_field"][:n]))
    with pytest.raises(ValueError, match="one plane across"):
        ao.from_optimus(one, model=model_of(beam))
    with pytest.raises(ValueError, match="no frequency"):
        ao.from_optimus(points=data["points"], field=data["total_field"])
    with pytest.raises(ValueError, match="compute_fields"):
        ao.from_optimus(SimpleNamespace(points=data["points"], field=SimpleNamespace()), model=model_of(beam))
    plane = synthetic.optimus_dataset(beam, "plane", location=(0, 0, 1e-3))  # the plane z = 1 mm, off the axis
    with pytest.raises(ValueError, match="from the beam axis"):
        ao.from_optimus(points=plane["points"], field=plane["total_field"], model=model_of(beam))


def test_mat_export(tmp_path: Path, beam: GaussianBeam, fundamental: dict) -> None:
    """A .mat file as export_to_file writes it (no frequency or medium): recognised, with them given."""
    data = synthetic.optimus_dataset(beam, scattered=0.25)
    path = tmp_path / "optimus_exported_data.mat"
    savemat(path, {k: np.atleast_2d(v) for k, v in data.items()})
    assert set(loadmat(path)) >= {"points", "total_field", "incident_field"}
    res = from_dataset(path, frequency=beam.frequency, sound_speed=beam.sound_speed, density=beam.density)
    assert_metrics_equal(fundamental, res.metrics(), 1e-9)
    inc = from_dataset(path, field="incident", **beam.medium()).metrics()
    assert inc["focus"]["p_positive"] == pytest.approx(0.75 * fundamental["focus"]["p_positive"], rel=1e-9)
    out = tmp_path / "out"
    assert main([str(path), "-o", str(out), "-q", "--frequency", "1e6", "--sound-speed", "1500",
                 "--density", "1000"]) == 0  # fmt: skip
    assert (out / "metrics.json").exists()
