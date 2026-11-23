"""The grid of the recorded planes: transverse (x, y) points, z planes and harmonics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from acoustic_outputs._types import RealArray


@dataclass(frozen=True)
class Grid:
    """A regular grid of ``nx`` x ``ny`` points on planes z = z0 + j dz.

    ``nt`` is the number of samples per period whose harmonics 1 .. nt/2 are held. A 2D field
    (one transverse axis) has ``ny = 1`` and ``dy = 1`` m, so that plane quantities are per metre
    in y. The recorded planes need not be uniformly spaced: :class:`~acoustic_outputs.results.Results`
    keeps their z, and ``dz`` is only their mean spacing.

    Attributes:
        nx, ny: transverse points.
        nz: planes.
        nt: samples per period; K = nt / 2 harmonics.
        dx, dy, dz: spacings [m].
        x0, y0, z0: the first point [m].
    """

    nx: int
    ny: int
    nz: int
    nt: int
    dx: float
    dy: float
    dz: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 1 or self.nz < 1:
            raise ValueError("the grid needs at least 2 points in x and 1 in y and z")
        if self.nt < 2 or self.nt % 2:
            raise ValueError("nt (samples per period) must be even and >= 2")
        if min(self.dx, self.dy, self.dz) <= 0:
            raise ValueError("grid spacings must be positive")

    @cached_property
    def x(self) -> RealArray:
        """x coordinates [m], shape (nx,)."""
        return self.x0 + self.dx * np.arange(self.nx)

    @cached_property
    def y(self) -> RealArray:
        """y coordinates [m], shape (ny,)."""
        return self.y0 + self.dy * np.arange(self.ny)

    @cached_property
    def z(self) -> RealArray:
        """Uniform z planes [m], shape (nz,) (the recorded z are in ``Results.z``)."""
        return self.z0 + self.dz * np.arange(self.nz)

    @cached_property
    def kx(self) -> RealArray:
        """Angular wavenumbers in x [rad/m], in FFT order, shape (nx,)."""
        return 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)

    @cached_property
    def ky(self) -> RealArray:
        """Angular wavenumbers in y [rad/m], in FFT order, shape (ny,)."""
        return 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)

    @property
    def n_harmonics(self) -> int:
        """Number of harmonics, K = nt / 2."""
        return self.nt // 2

    @property
    def centre_index(self) -> tuple[int, int]:
        """Indices (nx/2, ny/2) of the plane centre, the default beam axis."""
        return self.nx // 2, self.ny // 2

    @property
    def centre(self) -> tuple[float, float]:
        """Coordinates of the plane centre [m]."""
        i, j = self.centre_index
        return float(self.x[i]), float(self.y[j])

    def nearest(self, x: float, y: float, z: float | None = None) -> tuple[int, ...]:
        """Indices of the grid point nearest to (x, y) or (x, y, z), clamped to the grid."""
        i = int(np.clip(round((x - self.x0) / self.dx), 0, self.nx - 1))
        j = int(np.clip(round((y - self.y0) / self.dy), 0, self.ny - 1))
        if z is None:
            return i, j
        k = int(np.clip(round((z - self.z0) / self.dz), 0, self.nz - 1))
        return i, j, k
