# acoustic-outputs

Acoustic field parameters from the output of any acoustic solver: beam metrics, and the parameters of IEC 62127-1, 61828, TS 62556, 62359 and 61689. Three kinds of solver output are supported, and the outputs of BabelBrain, mSOUND and FOCUS are read as those programs write them:

| Kind of solver | Examples | What is done to its output |
| --- | --- | --- |
| frequency-domain pseudo-spectral (marching, angular spectrum, KZK) | pyWave3D, Wave3D, k-Wave `angularSpectrumCW` | harmonics on planes are taken as they are, in any axis order and phase convention |
| mixed-domain marching | mSOUND `Forward*_fund`, `Forward*_sec` | the fundamental and second harmonic on every plane |
| | mSOUND `Forward2D`, `Forward3D` (TMDM) | time series at the sensor mask, fitted as for k-Wave |
| Rayleigh / fast nearest method, KZK | FOCUS `fnm_cw`, `cw_pressure`, `kzk_cw`, `fnm_tsd` | the CW pressure or KZK harmonics on the coordinate grid, phase convention read from the field; transients fitted |
| time-domain pseudo-spectral (k-space) | k-wave-python `kspaceFirstOrder` | the harmonics are fitted to a steady window of the sensor time series |
| finite-difference time-domain | fullwave25 | the same fit, plus removal of the slow drift that absorbing layers leave; sensors are placed by grid index |
| transcranial planning (FDTD by BabelViscoFDTD) | BabelBrain | its `DataForSim.h5` is read as it is: the complex fundamental on the grid, with the medium of the target |

This code started as the recording code of [pyWave3D](https://github.com/djps/Wave3D). The metrics and file formats are the same, and so are the keys of `metrics.json`.

## Installation

Requires Python 3.10 or later.

```bash
pip install -e ".[test]"        # numpy, scipy, beartype (+ pytest, h5py)
pip install -e ".[hdf5,kwave]"  # optional: HDF5 files (h5py, hdf5plugin for blosc), k-wave-python
```

## Quick start

From the shell, with a dataset file (`.npz`, `.mat` or HDF5):

```bash
acoustic-outputs --example kwave kwave.npz      # write a synthetic k-Wave-style dataset to try
acoustic-outputs kwave.npz -o outputs           # metrics.json, axial.txt, planes.txt, focal_plane.txt, results.npz
acoustic-outputs data.mat --kind kwave --set order=F --density 998   # MATLAB k-Wave output
acoustic-outputs run.npz --kind fullwave --point p.txt 0 0 0.03       # also the waveform at (0, 0, 30 mm)
acoustic-outputs T1_500kHz_6PPW_DataForSim.h5 -o outputs               # BabelBrain: kind and frequency recognised
acoustic-outputs msound.mat -o outputs                                 # mSOUND: P_fundamental (and P_second), struct(mgrid)
acoustic-outputs focus.mat --kind focus --density 1000 -o outputs      # FOCUS: pressure, x, y, z, f0
```

From Python:

```python
import acoustic_outputs as ao

# k-wave-python: sensor_data = kspaceFirstOrder(kgrid, medium, source, sensor, ...)
res = ao.from_kwave(sensor_data, kgrid, frequency=1e6, sound_speed=1500, density=1000,
                    sensor_mask=sensor.mask, propagation="x")
# fullwave25: sensor_data = fullwave.Solver(...).run()
res = ao.from_fullwave(sensor_data, sensor, grid, density=1000)
# harmonics on planes, e.g. (nx, ny, nz) from angular_spectrum_cw
res = ao.from_harmonics(p, layout="xyz", x=x, y=y, z=z, frequency=1e6, sound_speed=1500,
                        density=1000, phase="absolute")

m = res.metrics()          # focus, widths, lengths, areas, power, and the IEC parameters under "iec"
res.save("outputs")
```

## Datasets

A dataset holds named arrays and scalars.

- **Kind:** its `kind` is stored in the file under `kind`, or given with `--kind`.
- **Structs:** `kgrid`, `sensor`, `sensor_data`, `medium`, `grid` and `mgrid` (or `mgrid_s`) saved whole are read field by field, so a MATLAB k-Wave `.mat` file works as it is. mSOUND's `set_grid` object can't be read outside MATLAB: save `mgrid_s = struct(mgrid)`.
- **Medium:** every kind needs `frequency` (or `f0`), `sound_speed` (or `c0`) and `density` (or `rho0`), from the file or the command line. For BabelBrain, the frequency comes from the `_<f>kHz_` part of the file name, and rho0 and c0 from the material at the target. mSOUND's `fc`, `medium.c0` and `medium.rho0`, and FOCUS's `soundspeed`, are also recognised.
- **Recognised without `kind`:** BabelBrain and mSOUND files. FOCUS files need `--kind focus`.

| `kind` | Required keys | Optional keys (defaults) |
| --- | --- | --- |
| `harmonics` | `harmonics` (complex), `x`, `z` | `y` (left out in 2D), `layout` (`"zkxy"`; leave out `k` for a single frequency), `phase` (`"retarded"` or `"absolute"`), `time_sign` (+1 for exp(+iωt), −1 for exp(−iωt)) |
| `time-series` | `p`, `dt`, the coordinates named in `layout` | `layout` (`"xyzt"`), `propagation` (`"z"`), `start_time`, `detrend` |
| `kwave` | `p` (n_sensor, Nt), `dt`, `x_vec` (and `y_vec`, `z_vec`) | `sensor_mask` (a rectangular block), `record_start_index` (1-based), `propagation` (`"x"`), `order` (`"C"`; `"F"` for MATLAB) |
| `fdtd` | as `time-series` | defaults change to `layout` `"xyt"`, `propagation` `"x"`, `detrend` true |
| `fullwave` | `sensor_data`, `outcoords`, `dt`, `dx` (and `dy`, `dz`) | `sampling_modulus_time`, `origin`, `propagation` (`"x"`), `detrend` (true) |
| `babelbrain` | BabelBrain's `DataForSim.h5` as written: `p_complex`, `x_vec`, `y_vec`, `z_vec` | `Material`, `MaterialMap`, `TargetLocation`, `field` (`"p_complex"`, `"p_complex_refocus"`, `"p_complex_water"`), `direction` (`"auto"`, `"+z"`, `"-z"`) |
| `msound` | `P_fundamental` (nx, ny, planes) or (nx, planes), or `p_time` (num_t, n_sensor) with `sensor_mask`; the plane spacing `dz` (`dy` in 2D); `dt` for `p_time` | `P_second`, `x`, `y` (default centred, from `dx`, `dy`), `z0` (0: the source plane), `fc` or `omega_c` |
| `focus` | `pressure`: complex (nx, ny, nz) (CW) or (nx, ny, nz, n) (KZK), or real (nx, ny, nz, nt) (transient); `x`, `y`, `z`, or `delta` and `xmin` … `zmax` | `phase` (`"auto"`), `time_sign`, `scale` (1); transient: `deltat` (or `dt`), `tmin` |

Every kind also takes these optional keys:

- **Harmonic fit:** `n_harmonics` (4), `periods` (2) and `window_start`.
- **Physics and geometry:** `alpha` (Np/m per harmonic), `focusing_depth` and `axis_xy`.
- **What is recorded:** `focal_quantity`, `levels_db`, `plane_peaks`, `metrics_harmonics`, `hydrophone_diameter`, `derating` and `face_distance`.

`--set KEY=VALUE` sets any key on the command line. The value is read as JSON, e.g. `--set levels_db=[-6,-3]`.

## Limits

- **Continuous waves only.** The field must be steady and periodic: CW, or a tone burst long enough to reach steady state. Pulse quantities are per period.
- **Fit warning.** The fit warns when more than 1% of the signal is left unexplained. That means the record is still transient, or it holds more harmonics than were asked for.
- **Sensor blocks.** Sensors must cover a rectangular block of grid points, such as a plane or a volume. fullwave25's sparse-grid whole-domain recording is not supported; use a sensor mask instead.
- **Exact intensity.** The exact (angular-spectrum) intensity assumes a one-way field. Where a time-domain record holds reflections, it is approximate.
- **BabelBrain.**
  - **One frequency:** BabelBrain simulates only the fundamental, so p+ = |p−| and there are no nonlinear parameters.
  - **Intensities:** they use rho c of the material at the target, so they are approximate in skull and skin.
  - **Normalised files:** the `*_NORM.nii.gz` files are normalised to 1, so they carry no absolute values. Use `DataForSim.h5`.
  - **Direction:** the beam's direction along z is decided from the power on the end planes (`direction` overrides it). A beam travelling towards −z is reported with z negated.
  - **Checked against:** the reader is tested on files written as BabelViscoFDTD's `SaveToH5py` writes them, including blosc compression. It has not yet been run on a real BabelBrain simulation.
- **mSOUND.**
  - **Conventions, from its source:** the frequency-domain fields are p = Re P exp(+iωt) in absolute phase (the propagator is exp(iK dz) with K = −√(ω²/c0² − k⊥²)), |P| is the peak amplitude, and plane i is (i − 1) dz from the source plane. The time-domain records start at t = 0 and are periodic over the window.
  - **z:** mgrid.z is centred on 0 and is not the distance from the source, so it is ignored; give `z0` for the source plane's position.
  - **Medium:** a heterogeneous `medium.rho` can't be used for the intensities: `medium.rho0`, or `--density`, is needed.
  - **Checked against:** mSOUND v0.1's source (`Forward3D_fund.m`, `Forward3D.m`, `set_grid.m`). Version 0.2 changed `Forward3D_fund` and `Forward3D_sec`, and moved the extra plane into `set_grid`'s `num_z`. The reader takes the number of planes from the array, so that should not matter, but version 0.2's source has not been read. Not yet run on real mSOUND output.
- **FOCUS.**
  - **Conventions:** FOCUS's documentation does not give its time convention, whether `kzk_cw`'s harmonics are in retarded time, or the scaling of its CW pressure. The reader reads the convention from how fast the phase turns along z (about ±k per metre in absolute phase, far less in retarded time). That needs planes less than 0.45 wavelengths apart; otherwise KZK harmonics are taken as retarded, exp(+iωt). A single frequency doesn't depend on the convention.
  - **Scale:** the pressure is taken to be in Pa; `scale` multiplies it if it isn't (for instance, a field computed for a unit surface velocity).
  - **Transients** (`fnm_tsd`) are lossless and must be tone bursts long enough to reach a steady state.
  - **Checked against:** the documentation only. Not yet run on real FOCUS output.
- **2D fields.** A 2D field (one transverse axis) gives power and areas per metre. Its y widths and its thermal and mechanical indices are NaN.

## Layout

```
src/acoustic_outputs/
  grid.py, waveform.py      plane grid; waveforms and peak pressures from harmonics
  recorder.py, results.py   PlaneRecorder builds Results from planes of harmonics; metrics and files
  iec.py                    the IEC parameter functions
  preprocess/               spectral.py, time_domain.py (from_kwave), fdtd.py (from_fullwave), babelbrain.py,
                            msound.py (from_msound, from_msound_time_series), focus.py (from_focus)
  datasets.py, cli.py       dataset files and the acoustic-outputs command
  synthetic.py              a focused Gaussian beam with known metrics, and datasets of it
docs/iec_outputs.md         every IEC parameter, how it is computed, and its key in metrics.json
tests/
```

## Tests

```bash
pytest          # about 40 s; tests/test_kwave.py runs k-wave-python if it is installed
```

- **Analytic values:** a synthetic focused Gaussian beam with harmonics has known metrics:
  - focus, widths and lengths at several levels;
  - area, power and the hydrophone average;
  - in 3D, in 2D and off axis.
- **Round trips:** every preprocessor is given the same beam, as harmonics and as time series in each solver's format, and must give its metrics back to 1e-8. The time series cover:
  - a time step that does not fit the period, and a transient at the start;
  - decimation, drift and shuffled sensors;
  - both k-Wave orders.
- **BabelBrain:** the fundamental of the beam, attenuated and stored as BabelBrain stores it, gives the same metrics as the same field given as harmonics. This holds for a beam travelling towards −z, for gzip- and blosc-compressed files, and for a field stored as magnitude only.
- **mSOUND and FOCUS:** the beam written as each of their functions writes it gives the metrics of the same harmonics:
  - mSOUND in 3D and 2D, with the transverse grid from the spacings alone, a `.mat` file holding `struct(mgrid)` and `medium`, and time-domain series on a sensor mask;
  - FOCUS CW with either time convention (read from the field), KZK harmonics (read as retarded; on coarse planes, taken as retarded with a warning), transients, grids given by `delta` and limits, and xz and yz 2D grids.
- **Files and command line:** the `.npz`, `.mat` and HDF5 readers, and the `acoustic-outputs` command.
- **Real k-Wave run:** a k-wave-python CW run is compared with an angular-spectrum projection of its own first plane. They agree to within 6% of the peak.

Set `ACOUSTIC_OUTPUTS_NO_BEARTYPE=1` to turn off run-time type checking.

## Differences from pyWave3D's results

- **Peak pressures:** they are refined by Newton steps between samples (`waveform.peak_pressures`). pyWave3D takes the largest of 4 samples per period per harmonic, which can miss a peak by up to 1 − cos(π/4K) of the fundamental: 2% for 4 harmonics, 0.1% for 16.
- **Side lobes:** secondary maxima below −120 dB are ignored as round-off.
- **Everything else** is computed the same way.
