"""acoustic_outputs: acoustic field parameters (IEC and beam metrics) from the output of any solver.

The field parameters are computed from transverse planes z = const that hold the harmonics
P_n(x, y), n = 1 .. K, of a periodic (continuous-wave) field, p = Re sum P_n exp(+i n omega tau).
:class:`PlaneRecorder` takes those planes in increasing z and makes the :class:`Results`, whose
``metrics()`` holds the focus, widths, power and the IEC parameters (``iec`` section), and whose
``save()`` writes them to files.

:mod:`acoustic_outputs.preprocess` reduces the output of three kinds of solver to such planes:

* frequency-domain pseudo-spectral solvers (spatial marching, angular spectrum, KZK), which
  already give harmonics on planes: :func:`from_harmonics`;
* time-domain pseudo-spectral solvers such as k-wave-python: :func:`from_time_series`,
  :func:`from_kwave`;
* FDTD solvers such as fullwave25: :func:`from_fdtd_time_series`, :func:`from_fullwave`;
* BabelBrain (transcranial planning on BabelViscoFDTD): :func:`from_babelbrain`.

:mod:`acoustic_outputs.datasets` reads the same data from ``.npz``, ``.mat`` or HDF5 files, and the
``acoustic-outputs`` command runs it from the shell.

Every public function and method is checked at run time with beartype, unless the
environment variable ``ACOUSTIC_OUTPUTS_NO_BEARTYPE`` is set before the package is imported.
"""

import os

if not os.environ.get("ACOUSTIC_OUTPUTS_NO_BEARTYPE"):
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # is_pep484_tower: an int is accepted where a float is expected, and a float where a complex is
    beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))

from acoustic_outputs.datasets import from_dataset, load_dataset
from acoustic_outputs.grid import Grid
from acoustic_outputs.preprocess import HarmonicFit
from acoustic_outputs.preprocess.babelbrain import from_babelbrain
from acoustic_outputs.preprocess.fdtd import from_fullwave
from acoustic_outputs.preprocess.fdtd import from_time_series as from_fdtd_time_series
from acoustic_outputs.preprocess.focus import from_focus
from acoustic_outputs.preprocess.msound import from_msound, from_msound_time_series
from acoustic_outputs.preprocess.spectral import from_harmonics
from acoustic_outputs.preprocess.time_domain import from_kwave, from_time_series
from acoustic_outputs.recorder import PlaneRecorder, RecordSettings
from acoustic_outputs.results import Results
from acoustic_outputs.waveform import harmonics_of, waveform

__version__ = "0.1.0"

__all__ = [
    "Grid",
    "HarmonicFit",
    "PlaneRecorder",
    "RecordSettings",
    "Results",
    "__version__",
    "from_babelbrain",
    "from_dataset",
    "from_fdtd_time_series",
    "from_focus",
    "from_fullwave",
    "from_harmonics",
    "from_kwave",
    "from_msound",
    "from_msound_time_series",
    "from_time_series",
    "harmonics_of",
    "load_dataset",
    "waveform",
]
