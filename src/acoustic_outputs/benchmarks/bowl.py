"""Benchmark C.1 of IEC TS 63587: the linear field of a uniformly vibrating spherically focused bowl.

A bowl of radius of curvature D and aperture radius a (sin φ = a/D, inset h = D(1 - cos φ))
vibrates uniformly, with normal velocity v0, and radiates into a homogeneous medium, which may be
absorbing (complex wavenumber k̂ = k + iα). The origin of z is the centre of the bowl's surface
(its apex), on the axis. Evaluating the Rayleigh integral on the axis gives, in the time
convention exp(-i omega t) used by the TS (C.1):

    p(z) = p0 (exp(i k̂ z) - exp(i k̂ R_m)) / (1 - z/D),   p0 = rho c v0,

with R_m = sqrt(a^2 + (z - h)^2) the distance to the rim, and at the geometric focus (C.2, C.3)

    p(D) = -i p0 k̂ h exp(i k̂ D),   |p(D)| = p0 h exp(-alpha D) sqrt(k^2 + alpha^2).

The singularity at z = D is removable: with a^2 + h^2 = 2 D h, R_m - z = 2 h (D - z) / (R_m + z),
so that

    p(z) = -i k̂ p0 exp(i k̂ z) E(i k̂ (R_m - z)) 2 h D / (R_m + z),   E(x) = (exp(x) - 1) / x,

which :meth:`FocusedBowl.axial_pressure` evaluates everywhere, the focus included.

The benchmark (TS 63587, Clause 6 and Annex C) compares a simulated field with this solution.
:class:`FocusedBowl` also gives the whole field, off the axis too, by a numerical quadrature of
the same Rayleigh integral over the bowl (:meth:`FocusedBowl.rayleigh`), so that the field can be
run through this package like any solver's output. :func:`compare_axis` compares any
:class:`~acoustic_outputs.results.Results` (from any reader) with the analytic axis;
:func:`run_bowl_benchmark` does the whole exercise, with a convergence study, and writes the
report that Annex C asks for: the inputs, how they were put into the model, the discrepancies,
and the convergence.

Fields here are returned in the TS's convention, exp(-i omega t), unless ``time_sign=+1`` asks
for this package's exp(+i omega t) (the complex conjugate).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_outputs import __version__
from acoustic_outputs._types import ComplexArray, RealArray
from acoustic_outputs.compare import error_norms, on_common_points
from acoustic_outputs.results import Results

log = logging.getLogger(__name__)


def _expm1_over_x(x: ComplexArray) -> ComplexArray:
    """(exp(x) - 1) / x, equal to 1 at x = 0."""
    out = np.ones_like(x, dtype=complex)
    nz = x != 0
    out[nz] = np.expm1(x[nz]) / x[nz]
    return out


@dataclass(frozen=True)
class FocusedBowl:
    """A uniformly vibrating spherically focused bowl in a homogeneous medium (TS 63587, C.1).

    The defaults are the transducer and water of benchmark C.2 (aperture 100 mm, radius of
    curvature 90 mm, 1 MHz, c = 1485 m/s, rho = 998 kg/m^3), at its linear level p0 = 15 kPa.

    Attributes:
        frequency: f [Hz].
        sound_speed, density: c [m/s] and rho [kg/m^3] of the medium.
        aperture: the aperture diameter 2a [m], in the aperture plane.
        radius_of_curvature: D [m].
        p0: rho c v0, the characteristic pressure at the surface [Pa].
        attenuation: alpha at f [Np/m] (k̂ = k + i alpha).
    """

    frequency: float = 1.0e6
    sound_speed: float = 1485.0
    density: float = 998.0
    aperture: float = 100e-3
    radius_of_curvature: float = 90e-3
    p0: float = 15e3
    attenuation: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.aperture / 2 <= self.radius_of_curvature:
            raise ValueError("the aperture radius must be positive and at most the radius of curvature")
        if min(self.frequency, self.sound_speed, self.density, self.p0) <= 0 or self.attenuation < 0:
            raise ValueError("frequency, sound speed, density and p0 must be positive, attenuation >= 0")

    # -- geometry ----------------------------------------------------------------------------

    @property
    def a(self) -> float:
        """Aperture radius [m]."""
        return self.aperture / 2

    @property
    def D(self) -> float:
        """Radius of curvature [m]."""
        return self.radius_of_curvature

    @property
    def phi(self) -> float:
        """Half-angle of the bowl, sin phi = a / D."""
        return math.asin(self.a / self.D)

    @property
    def h(self) -> float:
        """Inset: from the apex to the aperture plane, D (1 - cos phi) [m]."""
        return self.D * (1 - math.cos(self.phi))

    @property
    def wavelength(self) -> float:
        return self.sound_speed / self.frequency

    @property
    def k(self) -> float:
        """Real wavenumber [rad/m]."""
        return 2 * math.pi * self.frequency / self.sound_speed

    @property
    def k_hat(self) -> complex:
        """Complex wavenumber k + i alpha."""
        return complex(self.k, self.attenuation)

    @property
    def velocity(self) -> float:
        """Normal velocity amplitude v0 = p0 / (rho c) [m/s]."""
        return self.p0 / (self.density * self.sound_speed)

    @property
    def f_number(self) -> float:
        return self.D / self.aperture

    def medium(self) -> dict[str, float]:
        return {"frequency": self.frequency, "sound_speed": self.sound_speed, "density": self.density}

    # -- the analytic solution on the axis ------------------------------------------------------

    def axial_pressure(self, z: Any, *, time_sign: int = -1) -> ComplexArray:
        """The complex pressure on the axis at distances ``z`` from the apex (C.1, exact at z = D)."""
        zin = np.asarray(z, dtype=float)
        zv = np.atleast_1d(zin)
        kh = self.k_hat
        rm = np.sqrt(self.a**2 + (zv - self.h) ** 2)
        x = 1j * kh * (2 * self.h * (self.D - zv) / (rm + zv))  # i k̂ (R_m - z)
        p = -1j * kh * self.p0 * np.exp(1j * kh * zv) * _expm1_over_x(x) * (2 * self.h * self.D / (rm + zv))
        p = p.reshape(zin.shape)
        return p if time_sign == -1 else np.conj(p)

    def focal_pressure(self, *, time_sign: int = -1) -> complex:
        """p(D) = -i p0 k̂ h exp(i k̂ D) (C.2)."""
        p = -1j * self.p0 * self.k_hat * self.h * np.exp(1j * self.k_hat * self.D)
        return complex(p if time_sign == -1 else np.conj(p))

    def focal_magnitude(self) -> float:
        """|p(D)| = p0 h exp(-alpha D) sqrt(k^2 + alpha^2) (C.3)."""
        return self.p0 * self.h * math.exp(-self.attenuation * self.D) * math.hypot(self.k, self.attenuation)

    def beam_maximum(self) -> tuple[float, float]:
        """The beam maximum point on the axis and |p| there (the maximum of ppsi, i.e. of |p|^2, for a CW field).

        It lies before the geometric focus (the focal shift of a weakly focused bowl).
        """
        from scipy.optimize import minimize_scalar

        z = np.linspace(self.h + 0.5 * self.wavelength, 2 * self.D, 4001)
        mag = np.abs(self.axial_pressure(z))
        i = int(np.argmax(mag))
        lo, hi = z[max(i - 1, 0)], z[min(i + 1, z.size - 1)]
        opt = minimize_scalar(lambda s: -abs(complex(self.axial_pressure(s))), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-9 * self.D})  # fmt: skip
        return float(opt.x), float(-opt.fun)

    # -- the whole field, by quadrature of the Rayleigh integral --------------------------------

    def _nodes(self, points_per_wavelength: float) -> tuple[RealArray, RealArray, RealArray, RealArray]:
        lam = self.wavelength
        n_t = max(8, math.ceil(points_per_wavelength * self.D * self.phi / lam))
        n_p = max(8, math.ceil(points_per_wavelength * math.pi * self.a / lam))
        xt, wt = np.polynomial.legendre.leggauss(n_t)
        xp, wp = np.polynomial.legendre.leggauss(n_p)
        theta = 0.5 * self.phi * (xt + 1)
        wt = 0.5 * self.phi * wt
        psi = 0.5 * math.pi * (xp + 1)
        wp = 0.5 * math.pi * wp * 2  # the field point at azimuth 0: the integrand is even in psi
        rho = self.D * np.sin(theta)
        zs = self.D * (1 - np.cos(theta))
        w = (self.D**2 * np.sin(theta) * wt)[:, None] * wp[None, :]  # dS = D^2 sin(theta) dtheta dpsi
        return rho, zs, np.cos(psi), w

    def rayleigh(self, r: Any, z: Any, *, points_per_wavelength: float = 6.0, time_sign: int = -1) -> ComplexArray:
        """The complex pressure at radii ``r`` and distances ``z`` from the apex: (nr, nz).

        p(x) = -(i k̂ p0 / (2 pi)) ∫ exp(i k̂ R) / R dS over the bowl, by Gauss-Legendre quadrature
        in the polar angle and the azimuth, ``points_per_wavelength`` nodes per wavelength of arc.
        On the axis it converges to :meth:`axial_pressure`.
        """
        rv = np.atleast_1d(np.asarray(r, dtype=float)).ravel()
        zv = np.atleast_1d(np.asarray(z, dtype=float)).ravel()
        rho, zs, cpsi, w = self._nodes(points_per_wavelength)
        R, Z = np.meshgrid(rv, zv, indexing="ij")
        rf, zf = R.ravel(), Z.ravel()
        out = np.empty(rf.size, dtype=complex)
        nodes = w.size
        chunk = max(1, int(4e6 // nodes))
        kh = self.k_hat
        for s in range(0, rf.size, chunk):
            e = slice(s, s + chunk)
            r2 = (
                rf[e, None, None] ** 2
                + (rho**2)[None, :, None]
                - 2 * rf[e, None, None] * rho[None, :, None] * cpsi[None, None, :]
                + (zf[e, None, None] - zs[None, :, None]) ** 2
            )
            dist = np.sqrt(r2)
            out[e] = np.einsum("tp,ctp->c", w, np.exp(1j * kh * dist) / dist)
        p = (-1j * kh * self.p0 / (2 * math.pi)) * out.reshape(rv.size, zv.size)
        return p if time_sign == -1 else np.conj(p)


# -- comparing a simulation with the benchmark -------------------------------------------------------


def _refined_peak(z: RealArray, v: RealArray) -> tuple[float, float]:
    """The maximum of sampled ``v``, refined by the parabola through it and its neighbours."""
    i = int(np.argmax(v))
    if 0 < i < z.size - 1:
        y0, y1, y2 = v[i - 1], v[i], v[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            t = 0.5 * (y0 - y2) / den
            h = 0.5 * (z[i + 1] - z[i - 1])
            return float(z[i] + t * h), float(y1 - 0.25 * (y0 - y2) * t)
    return float(z[i]), float(v[i])


def compare_axis(results: Results, bowl: FocusedBowl, *, apex_z: float = 0.0) -> dict[str, Any]:
    """Compare the axis of ``results`` with the analytic solution of ``bowl`` (TS 63587, 6.3).

    Args:
        results: the simulated field (from any reader), its first harmonic on the beam axis.
        bowl: the benchmark transducer and medium.
        apex_z: the position of the bowl's apex in the ``results``' z [m].

    Returns:
        The discrepancies: error norms (3.43–3.46) of |p| and of the complex pressure on the
        axis (in retarded time, as the package keeps it), |p| at the geometric focus, and the
        beam maximum point and value (sampled, and refined by a parabola).
    """
    z = results.z - apex_z
    p_sim = results.axis[:, 0]
    ref = np.conj(bowl.axial_pressure(z))  # exp(+i omega t)
    ref = ref * np.exp(1j * bowl.k * (z - z[0]))  # retarded, as Results keep the planes
    mag = error_norms(np.abs(p_sim), np.abs(ref))
    cplx = error_norms(p_sim, ref)
    out: dict[str, Any] = {
        "axis": {
            "z_from_apex": [float(z[0]), float(z[-1])],
            "points": int(z.size),
            "magnitude": mag,
            "complex": cplx,
        }
    }
    if z[0] <= bowl.D <= z[-1]:
        _, at_d, _ = on_common_points(z, np.abs(p_sim), np.array([bowl.D]), np.array([bowl.focal_magnitude()]))
        out["geometric_focus"] = {
            "z_from_apex": bowl.D,
            "analytic": bowl.focal_magnitude(),
            "simulated": float(at_d[0]),
            "relative_error": float(at_d[0] / bowl.focal_magnitude() - 1),
        }
    z_bm, p_bm = bowl.beam_maximum()
    zs, ps = _refined_peak(z, np.abs(p_sim))
    i = int(np.argmax(np.abs(p_sim)))
    out["beam_maximum"] = {
        "analytic": {"z_from_apex": z_bm, "p": p_bm},
        "simulated_sampled": {"z_from_apex": float(z[i]), "p": float(np.abs(p_sim[i]))},
        "simulated_refined": {"z_from_apex": zs, "p": ps},
        "position_error": zs - z_bm,
        "relative_error": ps / p_bm - 1,
        "plane_spacing": float(np.max(np.diff(z))) if z.size > 1 else 0.0,
    }
    return out


def compare_longitudinal(
    results: Results,
    bowl: FocusedBowl,
    *,
    apex_z: float = 0.0,
    max_planes: int = 60,
    points_per_wavelength: float = 6.0,
) -> dict[str, Any]:
    """Compare |p| on the longitudinal plane of ``results`` (the x line through the axis of every
    plane, ``Results.xz_amplitude``) with the Rayleigh integral of ``bowl`` (TS 63587, 6.2).

    At most ``max_planes`` planes, evenly chosen, are compared, as the reference costs a
    quadrature per point. Returns the error norms of |p| and the planes and radii compared.
    """
    z = results.z - apex_z
    sel = np.unique(np.linspace(0, z.size - 1, min(max_planes, z.size)).round().astype(int))
    x = results.grid.x - results.grid.x[results.axis_index[0]]
    radii, inverse = np.unique(np.round(np.abs(x), 12), return_inverse=True)
    ref = np.abs(bowl.rayleigh(radii, z[sel], points_per_wavelength=points_per_wavelength))  # (nr, nsel)
    ref_xz = ref[inverse].T  # (nsel, nx)
    sim_xz = results.xz_amplitude[sel]
    return {
        "planes": int(sel.size),
        "z_from_apex": [float(z[sel[0]]), float(z[sel[-1]])],
        "x_half_width": float(np.max(np.abs(x))),
        "magnitude": error_norms(sim_xz, ref_xz),
    }


# -- the whole benchmark ---------------------------------------------------------------------------


@dataclass
class BenchmarkRun:
    """One simulation of the benchmark: its grid, the quadrature of the reference field, and the comparison."""

    spacing: float
    points_per_wavelength: float
    r_max: float
    z_range: tuple[float, float]
    comparison: dict[str, Any]
    results: Results | None = field(default=None, repr=False)


def _grid(bowl: FocusedBowl, spacing: float, r_max: float, z_range: tuple[float, float] | None):
    if z_range is None:
        z_range = (bowl.h + 4 * bowl.wavelength, 1.3 * bowl.D)
    r = spacing * np.arange(int(np.floor(r_max / spacing + 1e-9)) + 1)
    z = z_range[0] + spacing * np.arange(int(np.floor((z_range[1] - z_range[0]) / spacing + 1e-9)) + 1)
    return r, z


def source_plane(
    bowl: FocusedBowl, z0: float, half_width: float, spacing: float, *, points_per_wavelength: float = 8.0
) -> tuple[RealArray, ComplexArray]:
    """The bowl's field on the plane z0 (from the apex), on a square grid of ``spacing`` and
    half-width ``half_width``: x (n,) and p (n, n), exp(-i omega t).

    The field is axisymmetric, so the Rayleigh integral is evaluated at radii a sixteenth of a
    wavelength apart and revolved onto the grid; it is zero beyond ``half_width``.
    """
    from acoustic_outputs.preprocess.axisymmetric_frequency_domain import Revolver

    dr = bowl.wavelength / 16
    radii = dr * np.arange(int(np.ceil(half_width / dr)) + 2)
    profile = bowl.rayleigh(radii, [z0], points_per_wavelength=points_per_wavelength)[:, 0]
    rev = Revolver(radii, dx=spacing, r_max=half_width)
    return rev.x, rev(profile[None])[0]


def simulate(
    bowl: FocusedBowl,
    *,
    spacing: float,
    r_max: float,
    z_range: tuple[float, float] | None = None,
    method: str = "angular_spectrum",
    points_per_wavelength: float = 6.0,
    margin: float | None = None,
    pad: float = 4.0,
) -> Results:
    """Simulate the bowl's field and record it as any solver's output would be.

    ``method``:

    * ``"angular_spectrum"`` (a numerical model, the default): the field on a source plane, the
      first plane (by the Rayleigh integral, :func:`source_plane`), sampled at ``spacing`` over
      the aperture plus ``margin`` (default max(20 lambda, a / 2)), projected to each plane by
      the angular spectrum method with ``pad`` times zero padding
      (:func:`acoustic_outputs.propagation.angular_spectrum_planes`), and recorded in 3D on
      |x|, |y| <= ``r_max`` (:func:`~acoustic_outputs.preprocess.frequency_domain.from_harmonics`).
      Its accuracy depends on the sampling, the window and the padding, as a solver's does;
    * ``"rayleigh"`` (the reference): the Rayleigh integral at radii 0 .. ``r_max`` and on the
      planes, revolved onto (x, y)
      (:func:`~acoustic_outputs.preprocess.axisymmetric_frequency_domain.from_axisymmetric_harmonics`).
      It agrees with the analytic axis to the accuracy of the quadrature, and so checks the
      processing.

    The planes are ``spacing`` apart over ``z_range`` (distances from the apex; default from
    four wavelengths beyond the aperture plane to 1.3 D); the results' z are distances from the apex.
    """
    from acoustic_outputs.preprocess.axisymmetric_frequency_domain import from_axisymmetric_harmonics
    from acoustic_outputs.preprocess.frequency_domain import from_harmonics
    from acoustic_outputs.propagation import angular_spectrum_planes

    r, z = _grid(bowl, spacing, r_max, z_range)
    if method == "rayleigh":
        log.info("bowl benchmark: Rayleigh integral at %d x %d points (spacing %.4g mm)", r.size, z.size,
                 spacing * 1e3)  # fmt: skip
        p = bowl.rayleigh(r, z, points_per_wavelength=points_per_wavelength)
        return from_axisymmetric_harmonics(p, r=r, z=z, layout="rz", phase="absolute", time_sign=-1,
                                           focusing_depth=bowl.D - float(z[0]), **bowl.medium())  # fmt: skip
    if method != "angular_spectrum":
        raise ValueError(f"unknown method {method!r}: use 'angular_spectrum' or 'rayleigh'")
    margin = max(20 * bowl.wavelength, bowl.a / 2) if margin is None else margin
    half = max(bowl.a + margin, r_max)
    z0 = float(z[0])
    x, plane = source_plane(bowl, z0, half, spacing, points_per_wavelength=max(points_per_wavelength, 8.0))
    keep = np.abs(x) <= r_max * (1 + 1e-9)
    log.info("bowl benchmark: angular spectrum from a %d x %d plane at z = %.4g mm to %d planes (spacing %.4g mm)",
             x.size, x.size, z0 * 1e3, z.size, spacing * 1e3)  # fmt: skip
    planes = np.stack([q[np.ix_(keep, keep)] for q in angular_spectrum_planes(plane, spacing, spacing, z - z0,
                                                                              bowl.k_hat, pad=pad)])  # fmt: skip
    return from_harmonics(planes, layout="zxy", x=x[keep], y=x[keep], z=z, phase="absolute", time_sign=-1,
                          focusing_depth=bowl.D - z0, **bowl.medium())  # fmt: skip


def run_bowl_benchmark(
    bowl: FocusedBowl | None = None,
    *,
    spacings: tuple[float, ...] | None = None,
    r_max: float | None = None,
    z_range: tuple[float, float] | None = None,
    method: str = "angular_spectrum",
    points_per_wavelength: float = 6.0,
    longitudinal: bool = True,
) -> dict[str, Any]:
    """Run benchmark C.1 at several grid spacings and report it as Annex C asks.

    Args:
        bowl: the transducer and medium (default C.2's transducer in water, linear).
        spacings: grid spacings [m], coarsest first (default lambda/2, lambda/4, lambda/8).
        r_max: radius of the grid [m] (default 6 focal -6 dB widths, at least 4 wavelengths).
        z_range: planes, as distances from the apex [m] (see :func:`simulate`).
        method: the model, ``"angular_spectrum"`` or ``"rayleigh"`` (see :func:`simulate`).
        points_per_wavelength: nodes of the Rayleigh quadrature.
        longitudinal: also compare |p| on the longitudinal plane (:func:`compare_longitudinal`).

    Returns:
        The report: the inputs, how they are put into the model, for each spacing the
        comparison (:func:`compare_axis`) and the metrics of the finest run, and the convergence
        of the discrepancies with the spacing.
    """
    bowl = bowl or FocusedBowl()
    lam = bowl.wavelength
    spacings = spacings or (lam / 2, lam / 4, lam / 8)
    if r_max is None:
        r_max = max(4 * lam, 6 * 1.41 * lam * bowl.f_number)
    runs = []
    last: Results | None = None
    for s in spacings:
        res = simulate(bowl, spacing=s, r_max=r_max, z_range=z_range, method=method,
                       points_per_wavelength=points_per_wavelength)  # fmt: skip
        comparison = compare_axis(res, bowl)
        if longitudinal:
            lon = compare_longitudinal(res, bowl, points_per_wavelength=points_per_wavelength)
            comparison["longitudinal_plane"] = lon
        runs.append(BenchmarkRun(s, points_per_wavelength, r_max, (float(res.z[0]), float(res.z[-1])), comparison))
        last = res
    assert last is not None
    on_axis = bowl.rayleigh(0.0, last.z, points_per_wavelength=points_per_wavelength)[0]
    quad = error_norms(on_axis, bowl.axial_pressure(last.z))
    report = {
        "benchmark": "IEC TS 63587 C.1: linear field of a uniformly vibrating spherically focused transducer",
        "inputs": {
            **asdict(bowl),
            "derived": {
                "aperture_radius": bowl.a,
                "half_angle_deg": math.degrees(bowl.phi),
                "inset_h": bowl.h,
                "wavelength": lam,
                "velocity_v0": bowl.velocity,
                "f_number": bowl.f_number,
            },
        },
        "model": {
            "method": method,
            "description": {
                "angular_spectrum": "the Rayleigh field on a source plane four wavelengths beyond the aperture plane, "
                "sampled at the grid spacing over the aperture plus max(20 wavelengths, a/2), projected plane by "
                "plane by the angular spectrum method (4x zero padding), recorded in 3D "
                "(frequency_domain.from_harmonics; absolute phase, time_sign -1)",
                "rayleigh": "the Rayleigh integral on the (r, z) grid, revolved onto x, y "
                "(axisymmetric_frequency_domain.from_axisymmetric_harmonics; absolute phase, time_sign -1)",
            }[method],
            "reference_field": "Rayleigh integral over the bowl surface, Gauss-Legendre quadrature "
            f"({points_per_wavelength:g} nodes per wavelength of arc), time convention exp(-i omega t)",
            "reference_axis_vs_analytic": quad,
            "software": {"acoustic_outputs": __version__, "numpy": np.__version__},
            "grid": {"r_max": r_max, "z_range_from_apex": list(runs[-1].z_range)},
        },
        "runs": [
            {"spacing": r.spacing, "points_per_wavelength_of_grid": lam / r.spacing, **r.comparison} for r in runs
        ],
        "convergence": {
            "spacing": [r.spacing for r in runs],
            "axis_magnitude_L2_rel": [r.comparison["axis"]["magnitude"]["L2_rel"] for r in runs],
            "axis_complex_L2_rel": [r.comparison["axis"]["complex"]["L2_rel"] for r in runs],
            "beam_maximum_position_error": [r.comparison["beam_maximum"]["position_error"] for r in runs],
            "beam_maximum_relative_error": [r.comparison["beam_maximum"]["relative_error"] for r in runs],
        },
        "metrics_finest": last.metrics(),
    }
    if longitudinal:
        report["convergence"]["longitudinal_magnitude_L2_rel"] = [
            r.comparison["longitudinal_plane"]["magnitude"]["L2_rel"] for r in runs
        ]
    report["_results"] = last
    return report


def save_report(report: dict[str, Any], directory: str | Path) -> list[Path]:
    """Write ``benchmark.json``, ``benchmark_axis.txt`` (z, analytic and simulated |p|) and the
    finest run's outputs (``metrics.json`` etc.) to ``directory``."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    res: Results = report["_results"]
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    written = [out / "benchmark.json"]
    written[0].write_text(json.dumps(clean, indent=2, default=float) + "\n")
    bowl = FocusedBowl(**{k: v for k, v in clean["inputs"].items() if k != "derived"})
    z = res.z
    table = np.column_stack([z, np.abs(bowl.axial_pressure(z)), np.abs(res.axis[:, 0])])
    path = out / "benchmark_axis.txt"
    np.savetxt(path, table, header="z_from_apex[m] |p|_analytic[Pa] |p|_simulated[Pa]", fmt="%.9e")
    written.append(path)
    written += res.save(out, harmonics=1)
    return written


def summary(report: dict[str, Any]) -> list[str]:
    """A few lines on the benchmark result."""
    lines = [report["benchmark"]]
    ref = report["model"]["reference_axis_vs_analytic"]["L2_rel"]
    lines.append(f"model: {report['model']['method']}; reference field on the axis vs analytic: relative L2 {ref:.2e}")
    for r in report["runs"]:
        bm = r["beam_maximum"]
        lon = r.get("longitudinal_plane")
        lines.append(
            f"spacing {r['spacing'] * 1e3:.4g} mm ({r['points_per_wavelength_of_grid']:.1f} ppw): axis |p| "
            f"L2_rel {r['axis']['magnitude']['L2_rel']:.2e}, Linf_rel {r['axis']['magnitude']['Linf_rel']:.2e}"
            + (f"; xz plane |p| L2_rel {lon['magnitude']['L2_rel']:.2e}" if lon else "")
            + f"; beam maximum {bm['simulated_refined']['z_from_apex'] * 1e3:.3f} mm "
            f"(analytic {bm['analytic']['z_from_apex'] * 1e3:.3f}), p error {bm['relative_error']:+.2e}"
        )
    return lines
