"""Turn the output of other acoustic solvers into :class:`~acoustic_outputs.results.Results`.

Every field parameter of :class:`~acoustic_outputs.results.Results` (see ``docs/iec_outputs.md``) is
computed from transverse planes holding the harmonics of a periodic field. Each module here
reduces one kind of solver output to such planes and passes them, in increasing z, to a
:class:`~acoustic_outputs.recorder.PlaneRecorder`. A frequency-domain solver gives the harmonics
(complex amplitudes) directly; a time-domain solver gives the pressure in time, whose harmonics
are fitted over a steady window (:class:`HarmonicFit`).

Frequency domain (complex harmonics):

* :mod:`~acoustic_outputs.preprocess.frequency_domain`: harmonics on planes of a regular grid
  (marching, angular-spectrum and KZK pseudo-spectral solvers);
* :mod:`~acoustic_outputs.preprocess.axisymmetric_frequency_domain`: harmonics P_n(r, z) on (r, z)
  grids, at the nodes of (r, z) meshes or in DOLFINx XDMF files, revolved onto (x, y) planes;
* :mod:`~acoustic_outputs.preprocess.unstructured_frequency_domain`: harmonics at the nodes of
  unstructured 3D (or planar 2D) meshes, or in DOLFINx XDMF files, resampled on planes.

Time domain (pressure time series, fitted for their harmonics):

* :mod:`~acoustic_outputs.preprocess.time_domain`: time series on a regular grid, and k-wave-python
  (time-domain pseudo-spectral, k-space) sensor data;
* :mod:`~acoustic_outputs.preprocess.fdtd`: finite-difference time-domain solvers such as
  fullwave25, with the drift of the absorbing layers removed;
* :mod:`~acoustic_outputs.preprocess.axisymmetric_time_domain`: time series p(r, z, t) on (r, z)
  grids, at the nodes of (r, z) meshes or in DOLFINx XDMF files;
* :mod:`~acoustic_outputs.preprocess.unstructured_time_domain`: time series at the nodes of
  unstructured 3D (or planar 2D) meshes, or in DOLFINx XDMF files.

Particular solvers, in their own formats:

* :mod:`~acoustic_outputs.preprocess.babelbrain`: BabelBrain (FDTD by BabelViscoFDTD), whose
  ``DataForSim.h5`` holds the complex fundamental (frequency domain, from a steady FDTD run);
* :mod:`~acoustic_outputs.preprocess.msound`: mSOUND, its frequency-specific fundamental and
  second harmonic (frequency domain) or its transient sensor series (time domain);
* :mod:`~acoustic_outputs.preprocess.focus`: FOCUS, its CW and KZK pressures (frequency domain)
  or its transient pressures (time domain);
* :mod:`~acoustic_outputs.preprocess.optimus`: OptimUS (boundary elements), its total, incident
  or scattered field at one frequency (frequency domain), on planes across the beam or a plane
  through it.

The finite-element modules share :mod:`~acoustic_outputs.preprocess._mesh` (cells, point
location, XDMF files, the DOLFINx probe). ``spectral`` is the former name of ``frequency_domain``.

The result is a :class:`~acoustic_outputs.results.Results`, with ``metrics()``
(including the IEC parameters) and ``save()``. A 2D field (one transverse axis) gives a grid with
``ny = 1`` and ``dy = 1 m``: its power and areas are per metre in y and its y widths are NaN.
"""

from acoustic_outputs.preprocess import (
    axisymmetric_frequency_domain,
    axisymmetric_time_domain,
    babelbrain,
    fdtd,
    focus,
    frequency_domain,
    msound,
    optimus,
    spectral,
    time_domain,
    unstructured_frequency_domain,
    unstructured_time_domain,
)
from acoustic_outputs.preprocess._common import HarmonicFit
from acoustic_outputs.preprocess._mesh import DolfinxProbe, read_xdmf

__all__ = [
    "DolfinxProbe",
    "HarmonicFit",
    "axisymmetric_frequency_domain",
    "axisymmetric_time_domain",
    "babelbrain",
    "fdtd",
    "focus",
    "frequency_domain",
    "msound",
    "optimus",
    "read_xdmf",
    "spectral",
    "time_domain",
    "unstructured_frequency_domain",
    "unstructured_time_domain",
]
