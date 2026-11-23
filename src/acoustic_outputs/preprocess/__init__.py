"""Turn the output of other acoustic solvers into :class:`~acoustic_outputs.results.Results`.

Every field parameter of :class:`~acoustic_outputs.results.Results` (see ``docs/iec_outputs.md``) is
computed from transverse planes holding the harmonics of a periodic field. Each module here
reduces one kind of solver output to such planes and passes them, in increasing z, to a
:class:`~acoustic_outputs.recorder.PlaneRecorder`:

* :mod:`~acoustic_outputs.preprocess.spectral`: frequency-domain pseudo-spectral (marching,
  angular-spectrum, KZK) solvers, which already give harmonics on planes;
* :mod:`~acoustic_outputs.preprocess.time_domain`: time-domain pseudo-spectral (k-space) solvers such as
  k-wave-python, whose sensor time series are fitted for harmonics over a steady window;
* :mod:`~acoustic_outputs.preprocess.fdtd`: finite-difference time-domain solvers such as fullwave25,
  as the time-domain case with the drift of the absorbing layers removed and the sensor indices
  of fullwave25;
* :mod:`~acoustic_outputs.preprocess.babelbrain`: BabelBrain's transcranial simulations (FDTD by
  BabelViscoFDTD), whose ``DataForSim.h5`` holds the complex fundamental on the grid;
* :mod:`~acoustic_outputs.preprocess.msound`: mSOUND's mixed-domain method, its frequency-specific
  fundamental and second harmonic on planes, or its time-domain sensor series;
* :mod:`~acoustic_outputs.preprocess.focus`: FOCUS's CW, KZK and transient pressures on a coordinate
  grid, with the phase convention read from the field.

The result is a :class:`~acoustic_outputs.results.Results`, with ``metrics()``
(including the IEC parameters) and ``save()``. A 2D field (one transverse axis) gives a grid with
``ny = 1`` and ``dy = 1 m``: its power and areas are per metre in y and its y widths are NaN.
"""

from acoustic_outputs.preprocess import babelbrain, fdtd, focus, msound, spectral, time_domain
from acoustic_outputs.preprocess._common import HarmonicFit

__all__ = ["HarmonicFit", "babelbrain", "fdtd", "focus", "msound", "spectral", "time_domain"]
