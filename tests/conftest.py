"""Shared fixtures: a synthetic focused Gaussian beam (known metrics) and its results."""

from __future__ import annotations

import math
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
