# Acoustic Outputs from Simulations

<<<<<<< README.md
[![output](https://github.com/djps/acoustic-outputs/actions/workflows/pytest.yml/badge.svg)](https://github.com/djps/acoustic-outputs/actions/workflows/pytest.yml)

This repository tries to derive acoustic field parameters from the output of any acoustic solver: beam metrics, and the parameters defined in standards IEC 62127-1, 61828, TS 62556, 62359 and 61689. 

They are computed from the harmonics of a steady periodic field on planes across the beam. 

Frequency-domain output gives those harmonics directly. 

Time-domain output gives the pressure in time, and its harmonics are fitted over a steady window. 

Both are supported, on regular grids, on axisymmetric (r, z) grids and meshes, and on unstructured 3D meshes. 

The outputs of 
* [BabelBrain](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD),
* [mSOUND](https://github.com/m-SOUND/mSOUND),
* [fullwave](https://github.com/pinton-lab/fullwave25)
* [FOCUS](https://www.egr.msu.edu/~fultras-web/)
* [k-wave-python](https://github.com/waltsims/k-wave-python)
* [optimus](https://github.com/optimuslib/optimus)
* [FEniCSx](https://docs.fenicsproject.org/)

are read as those programs write them. The output data is then preprocessed into a format that can be easily processed. 
=======
Acoustic field parameters from the output of any acoustic solver: beam metrics, and the parameters of IEC 62127-1, 61828, TS 62556, 62359 and 61689. They are computed from the harmonics of a steady periodic field on planes across the beam. Frequency-domain output gives those harmonics directly. Time-domain output gives the pressure in time, and its harmonics are fitted over a steady window. Both are supported, on regular grids, on axisymmetric (r, z) grids and meshes, and on unstructured 3D meshes. The outputs of BabelBrain, mSOUND, FOCUS and OptimUS are read as those programs write them:
>>>>>>> /tmp/stash.md

| Domain | Kind of solver | Examples | What is done to its output |
| --- | --- | --- | --- |
| frequency | pseudo-spectral marching, angular spectrum, KZK | pyWave3D, k-Wave `angularSpectrumCW` | harmonics on planes are taken as they are, in any axis order and phase convention |
| frequency | mixed-domain marching | mSOUND `Forward*_fund`, `Forward*_sec` | the fundamental and second harmonic on every plane |
| frequency | Rayleigh / fast nearest method, KZK | FOCUS `fnm_cw`, `cw_pressure`, `kzk_cw` | the CW pressure or KZK harmonics on the coordinate grid, phase convention read from the field |
| frequency | boundary elements (Helmholtz, with scatterers) | OptimUS pistons, bowls and arrays; `VisualisePlane`, `VisualiseCloudPoints`, `export_to_file` | the total, incident or scattered field on planes across the beam (3D), or on a plane through the axis (revolved, after a symmetry check); convention exp(−iωt) |
| frequency | transcranial planning (steady FDTD by BabelViscoFDTD) | BabelBrain | its `DataForSim.h5` is read as it is: the complex fundamental on the grid, with the medium of the target |
| frequency | axisymmetric finite element (Helmholtz), KZK | FEniCSx / DOLFINx models in (r, z); any (r, z) grid | harmonics on an (r, z) grid, at mesh nodes or in a DOLFINx `.xdmf` file; each plane revolved onto (x, y) |
| frequency | finite element on unstructured meshes (Helmholtz) | FEniCSx / DOLFINx in 3D (tetrahedra, hexahedra) or planar 2D | harmonics at the nodes resampled, plane by plane, by linear interpolation over the cells |
| time | pseudo-spectral (k-space) | k-wave-python `kspaceFirstOrder` | the harmonics are fitted to a steady window of the sensor time series |
| time | finite difference (FDTD) | fullwave25 | the same fit, plus removal of the slow drift that absorbing layers leave; sensors are placed by grid index |
| time | mixed-domain marching (TMDM) | mSOUND `Forward2D`, `Forward3D` | time series at the sensor mask, fitted as for k-Wave |
| time | Rayleigh transients | FOCUS `fnm_tsd` | tone bursts fitted as for k-Wave |
| time | axisymmetric finite element (wave, Westervelt) | FEniCSx / DOLFINx models in (r, z); any (r, z) grid | time series on an (r, z) grid, at mesh nodes or in an `.xdmf` file, fitted, then revolved |
| time | finite element on unstructured meshes (wave, Westervelt) | FEniCSx / DOLFINx in 3D or planar 2D, e.g. FEniCSx-FUS | time series at the nodes fitted there, then resampled on planes |

The metrics and file formats are the same, and so are the keys of `metrics.json`.

## Installation

Requires Python 3.10 or later.

```bash
pip install -e ".[test]"        # numpy, scipy, beartype (+ pytest, h5py)
pip install -e ".[hdf5,kwave]"  # optional: HDF5 files (h5py, hdf5plugin for blosc), k-wave-python
pip install -e ".[examples]"    # the examples: k-wave-python, pywave3d, matplotlib
```

## Quick start

From the shell, with a dataset file (`.npz`, `.mat` or `.h5`):

```bash
acoustic-outputs --example kwave kwave.npz      # write a synthetic k-Wave-style dataset to try
acoustic-outputs kwave.npz -o outputs           # metrics.json, axial.txt, planes.txt, focal_plane.txt, results.npz
acoustic-outputs data.mat --kind kwave --set order=F --density 998   # MATLAB k-Wave output
acoustic-outputs run.npz --kind fullwave --point p.txt 0 0 0.03       # also the waveform at (0, 0, 30 mm)
acoustic-outputs T1_500kHz_6PPW_DataForSim.h5 -o outputs               # BabelBrain: kind and frequency recognised
acoustic-outputs msound.mat -o outputs                                 # mSOUND: P_fundamental (and P_second), struct(mgrid)
acoustic-outputs focus.mat --kind focus --density 1000 -o outputs      # FOCUS: pressure, x, y, z, f0
acoustic-outputs optimus_exported_data.mat --frequency 5e5 --sound-speed 1500 --density 1000  # OptimUS export
acoustic-outputs hifu.xdmf --frequency 1e6 --sound-speed 1500 --density 1000 --set 'field=["p1","p2"]'  # DOLFINx
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

# OptimUS: postprocess objects after compute_fields() (planes across the beam, or one plane through it)
res = ao.from_optimus(planes, model=model)       # frequency, medium, source axis and location from the model

# axisymmetric, frequency domain: harmonics (nr, nz, K) on an (r, z) grid, at FE nodes, or in an XDMF file
res = ao.from_axisymmetric_harmonics(H, r=r, z=z, layout="rzk", frequency=1e6, sound_speed=1500, density=1000)
res = ao.from_axisymmetric_mesh_harmonics(V.tabulate_dof_coordinates(), u.x.array, frequency=1e6,
                                          sound_speed=1500, density=1000)
res = ao.from_axisymmetric_xdmf_harmonics("hifu.xdmf", ["p1", "p2"], frequency=1e6, sound_speed=1500, density=1000)
# axisymmetric, time domain: p (nr, nz, nt) on a grid, (n, nt) at FE nodes, or a function saved every step
res = ao.from_axisymmetric_time_series(p, dt, r=r, z=z, frequency=1e6, sound_speed=1500, density=1000)
res = ao.from_axisymmetric_mesh_time_series(points, series, dt, frequency=1e6, sound_speed=1500, density=1000)
res = ao.from_axisymmetric_xdmf_time_series("hifu_td.xdmf", "p", frequency=1e6, sound_speed=1500, density=1000)
# unstructured 3D mesh, frequency domain: nodes (n, 3), harmonics (n, K), cells; or a 3D XDMF file
res = ao.from_unstructured_harmonics(mesh.geometry.x, values, cells=mesh.geometry.dofmap, cell_type="tetrahedron",
                                     frequency=1e6, sound_speed=1500, density=1000)
res = ao.from_unstructured_xdmf_harmonics("bowl3d.xdmf", ["p1", "p2"], frequency=1e6, sound_speed=1500, density=1000)
# unstructured 3D mesh, time domain: series (n, nt) at the nodes; or a function saved every step
res = ao.from_unstructured_time_series(points, series, dt, cells=cells, frequency=1e6, sound_speed=1500, density=1000)
res = ao.from_unstructured_xdmf_time_series("bowl3d_td.xdmf", "p", frequency=1e6, sound_speed=1500, density=1000)

m = res.metrics()          # focus, widths, lengths, areas, power, and the IEC parameters under "iec"
res.save("outputs")
```

Inside a DOLFINx program, the field can be sampled exactly (any degree, in parallel) instead of read from files. In the frequency domain, `sample_dolfinx_rz` and `sample_dolfinx_planes` evaluate the harmonics on an (r, z) grid or on planes. In the time domain, a `DolfinxProbe` records the pressure at those points at every step:

```python
from acoustic_outputs.preprocess import DolfinxProbe
from acoustic_outputs.preprocess.axisymmetric_frequency_domain import rz_points

probe = DolfinxProbe(mesh, rz_points(r, z))
for step in range(n_steps):
    ...                     # advance u
    probe.record(u)
res = ao.from_axisymmetric_time_series(probe.series().reshape(r.size, z.size, -1), dt, r=r, z=z,
                                       start_time=t0, frequency=1e6, sound_speed=1500, density=1000)
```

## Benchmarks (IEC TS 63587)

IEC TS 63587 (measurement-based simulation) asks for a simulation model to be benchmarked against known results before it is used (Clause 6, Annex C). Benchmark **C.1** is the linear field of a uniformly vibrating spherically focused bowl, whose axial pressure is known in closed form. `acoustic_outputs.benchmarks` provides:

- **The analytic solution:** `FocusedBowl.axial_pressure`, in a form that stays finite at the geometric focus, plus the focal values (C.2, C.3) and the beam maximum point (the maximum of ppsi on the axis; it lies before the geometric focus).
- **The whole field:** the Rayleigh integral over the bowl by Gauss–Legendre quadrature (`FocusedBowl.rayleigh`), on and off the axis. It agrees with the analytic axis to 1e-14.
- **A comparison for any solver's output:** `compare_axis(results, bowl, apex_z=...)` takes the `Results` of any reader in this package (k-Wave, FOCUS, OptimUS, finite elements, ...). It reports the TS 63587 error norms (relative L2 and L∞, 3.43–3.46) of |p| and of the complex pressure on the axis, |p| at the geometric focus, and the beam maximum's position and value. `compare_longitudinal` does the same for |p| on the longitudinal plane.
- **A built-in run with a convergence study:** `run_bowl_benchmark` writes the report that Annex C asks for (inputs, how they enter the model, discrepancies, convergence). Its default model is an angular-spectrum projection from a Rayleigh source plane (`acoustic_outputs.propagation`), a genuine numerical model whose error depends on its sampling, window and padding. `method="rayleigh"` checks the processing alone, which should agree to round-off.

```bash
acoustic-outputs --benchmark bowl -o outputs/c1             # C.2's transducer (100 mm, 90 mm ROC, 1 MHz), linear, ~1 min
acoustic-outputs --benchmark bowl --set attenuation=5 --set "spacings=[0.00037]"
python examples/benchmark_c1_bowl.py --results my_solver.npz --apex-z 0 --frequency 1e6 --sound-speed 1485 --density 998
```

With the default C.2 transducer (F-number 0.9) at 2 and 4 grid points per wavelength, the angular-spectrum model's axial |p| agrees with the analytic solution to 5e-4 (relative L2), and its beam maximum to 2e-4 in pressure and 4 µm in position. For a small, strongly focused bowl (20 mm aperture, 30 mm ROC), the error falls from 5e-4 to 2e-4 as the grid goes from 2 to 4 points per wavelength. The outputs are `benchmark.json`, `benchmark_axis.txt` (z, analytic and simulated |p|) and the usual `metrics.json`, `axial.txt`, ... of the finest run.

## Datasets

A dataset holds named arrays and scalars.

- **Kind:** its `kind` is stored in the file under `kind`, or given with `--kind`.
- **Structs:** `kgrid`, `sensor`, `sensor_data`, `medium`, `grid` and `mgrid` (or `mgrid_s`) saved whole are read field by field, so a MATLAB k-Wave `.mat` file works as it is. mSOUND's `set_grid` object can't be read outside MATLAB, and Octave can't save `struct(mgrid)`: save a struct of the fields, e.g. `mgrid_s = struct('dx', mgrid.dx, 'dy', mgrid.dy, 'dz', mgrid.dz, 'dt', mgrid.dt, 'x', mgrid.x, 'y', mgrid.y)`. FOCUS's `coordinate_grid` and `medium` structs are read as they are.
- **Medium:** every kind needs `frequency` (or `f0`), `sound_speed` (or `c0`) and `density` (or `rho0`), from the file or the command line. For BabelBrain, the frequency comes from the `_<f>kHz_` part of the file name, and rho0 and c0 from the material at the target. mSOUND's `fc`, `medium.c0` and `medium.rho0`, and FOCUS's `soundspeed`, are also recognised.
- **Recognised without `kind`:** BabelBrain, mSOUND and OptimUS (`.mat` export) files, and `.xdmf` files (a 2D mesh as axisymmetric, a 3D one as unstructured; `--kind unstructured` reads a 2D mesh as a planar field). FOCUS files need `--kind focus`.

| `kind` | Required keys | Optional keys (defaults) |
| --- | --- | --- |
| `harmonics` | `harmonics` (complex), `x`, `z` | `y` (left out in 2D), `layout` (`"zkxy"`; leave out `k` for a single frequency), `phase` (`"retarded"` or `"absolute"`), `time_sign` (+1 for exp(+iωt), −1 for exp(−iωt)) |
| `time-series` | `p`, `dt`, the coordinates named in `layout` | `layout` (`"xyzt"`), `propagation` (`"z"`), `start_time`, `detrend` |
| `kwave` | `p` (n_sensor, Nt), `dt`, `x_vec` (and `y_vec`, `z_vec`) | `sensor_mask` (a rectangular block), `record_start_index` (1-based), `propagation` (`"x"`), `order` (`"C"`; `"F"` for MATLAB) |
| `fdtd` | as `time-series` | defaults change to `layout` `"xyt"`, `propagation` `"x"`, `detrend` true |
| `fullwave` | `sensor_data`, `outcoords`, `dt`, `dx` (and `dy`, `dz`) | `sampling_modulus_time`, `origin`, `propagation` (`"x"`), `detrend` (true) |
| `babelbrain` | BabelBrain's `DataForSim.h5` as written: `p_complex`, `x_vec`, `y_vec`, `z_vec` | `Material`, `MaterialMap`, `TargetLocation`, `field` (`"p_complex"`, `"p_complex_refocus"`, `"p_complex_water"`), `direction` (`"auto"`, `"+z"`, `"-z"`) |
| `msound` | `P_fundamental` (nx, ny, planes) or (nx, planes), or `p_time` (num_t, n_sensor) with `sensor_mask`; the plane spacing `dz` (`dy` in 2D); `dt` for `p_time` | `P_second`, `x`, `y` (default centred, from `dx`, `dy`), `z0` (0: the source plane), `fc` or `omega_c` |
| `axisymmetric` | `harmonics` (complex) with `r`, `z`; or `p` (real) with `r`, `z`, `dt`; or `points` (n, 2) and `values` (complex (n[, K]), or real (n, nt) with `dt`); or a DOLFINx `.xdmf` file | `layout` (`"zkr"`, or `"rzt"` for `p`), `cells`, `cell_type`, `spacing`, `field` (XDMF: one name, or a list of harmonics), `coordinates` (`"rz"`: x[0] = r; or `"zr"`), `phase` (`"auto"`), `time_sign`, `dx`, `r_max` |
| `optimus` | OptimUS's `.mat` export: `points` (3, N), `total_field` (or `incident_field`, `scattered_field`), and the frequency and medium (not in the file) | `field` (`"total"`), `propagation` (`"x"`, OptimUS's default; or `source_axis`), `axis_point` (or `location`), `z_min`, `geometry` (`"auto"`, `"3d"`, `"axisymmetric"`, `"planar"`), `dx`, `r_max` |
| `unstructured` | `points` (n, 3) or (n, 2) and `values` (complex (n[, K]), or real (n, nt) with `dt`); or a DOLFINx `.xdmf` file | `cells`, `cell_type` (tetrahedron / triangle), `max_edge`, `propagation` (`"z"` in 3D, `"x"` in 2D), `x`, `y`, `z` (grid vectors), `spacing` (median node spacing), `field`, `time`, `phase` (`"auto"`), `time_sign` |
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
  - **Checked against:** mSOUND v0.1 run in GNU Octave 10. `Forward2D_fund` and `Forward3D_fund` match an angular-spectrum propagation of their own source plane to round-off, and the fit of `Forward2D`'s and `Forward3D`'s CW records matches them to 1e-5 or better.
  - **Version 0.2** changed `Forward3D_fund` and `Forward3D_sec`, and moved the extra plane into `set_grid`'s `num_z`. The reader takes the number of planes from the array, so that should not matter, but version 0.2 has not been run.
- **FOCUS.**
  - **Conventions:** FOCUS's documentation does not give them, but run, `fnm_cw` gives p = Re P e^{+iωt} in absolute phase, in Pa for the transducer's surface velocity (1 m/s by default). Whether `kzk_cw`'s harmonics are in retarded time is unknown. The reader reads the convention from how fast the phase turns along z (about ±k per metre in absolute phase, far less in retarded time). That needs planes less than 0.45 wavelengths apart; otherwise KZK harmonics are taken as retarded, exp(+iωt). A single frequency doesn't depend on the convention.
  - **Scale:** the pressure is taken to be in Pa; `scale` multiplies it, for instance by the real surface velocity if the field was computed for 1 m/s.
  - **Transients** (`fnm_tsd`) are lossless and must be tone bursts long enough to reach a steady state.
  - **Checked against:** FOCUS 0.981's Octave build. A piston's `fnm_cw` axis matches the exact solution to 1e-9, and an `fnm_tsd` tone burst, fitted, matches it to 1e-4. That build has no `kzk_cw`, so the KZK route is tested only on synthetic data.
- **OptimUS.**
  - **What it gives:** the field at one frequency, p = Re(u e^{−iωt}) (BEMPP's convention), in absolute phase, in Pa for the sources' surface velocities (1 m/s by default). Only the fundamental, so p+ = |p−| and there are no nonlinear parameters. A pulse is a set of runs at single frequencies.
  - **Grids:** the points must lie on a regular grid aligned with the beam, which must run along x, y or z (either sign). Planes across the beam (several `VisualisePlane` objects at different offsets, or a cloud of points) give a 3D field. One `VisualisePlane` through the axis is revolved as an axisymmetric field: its two halves are compared, and a difference above 1% of the peak is warned of (a steered array, or an asymmetric scatterer). Its grid must have a point on the axis (an odd number of points across a centred box).
  - **Source:** with the model, the frequency, the exterior medium, the source axis and location and a bowl's radius of curvature (for the focal shift) are read from it. Planes behind the source are left out. The `.mat` export holds none of these: give them.
  - **Checked against:** the Gaussian beam's fundamental, written as OptimUS's postprocess objects and `.mat` export write it, along ±x, y and −z, from an offset source: the 3D metrics to 1e-9, and from a plane through the axis to 2e-4. OptimUS itself has not been run here (it runs in its Docker image).
- **Axisymmetric fields.**
  - **Revolution:** each plane is revolved onto a square (x, y) grid of spacing `dx` (default the smallest radial spacing) out to `r_max` (default the last radius), by a cubic spline of the profile's even extension, so the slope on the axis is zero. Power, areas, widths and the IEC indices are those of the 3D beam. The beam axis is r = 0.
  - **Cost:** a plane holds (2 r_max / dx + 1)² points. A large radius at a fine spacing is slow; use `r_max` and `dx` to bound it.
  - **Finite-element nodes** are resampled on an (r, z) grid (spacing `spacing`, default the median node spacing) by linear interpolation over the mesh cells (or a Delaunay triangulation of the nodes, less its triangles with an edge more than 5 times the median, which would bridge a concave part of the domain). An absolute-phase field has its carrier exp(−inkz) removed at the nodes first, so that only the smooth envelope is interpolated. Time series at the nodes are fitted there first. High-order solutions are best sampled in DOLFINx itself: `sample_dolfinx_rz` (frequency domain) or a `DolfinxProbe` (time domain) evaluates them exactly, in parallel, on the (r, z) grid.
  - **XDMF:** a frequency-domain file holds complex functions (one per harmonic); a time-domain file holds a real function written at every saved step, of which only the steps of the fitted window are read. DOLFINx writes the values at the geometry nodes, so a function must have the mesh's degree (for example P1 on a linear mesh); complex functions come as `real_<name>` and `imag_<name>`, which are joined. Triangles and quadrilaterals, of any degree, are read; ADIOS2 `.bp` (VTX) files are not.
  - **Checked against:** the Gaussian beam on an (r, z) grid gives the metrics of the same beam in 3D to 2e-4, and at the nodes of an unstructured mesh 0.25 mm apart to 0.5%. The XDMF reader is tested on files laid out as DOLFINx writes them, not on files written by DOLFINx itself. `sample_dolfinx_rz` and `DolfinxProbe` have not been run: DOLFINx is not a dependency.
- **Unstructured meshes.**
  - **Resampling:** the nodal values are interpolated linearly over the cells, split into tetrahedra (or triangles), on planes across the beam (`propagation`), one plane at a time. The grid defaults to the box of the nodes at the median node spacing, centred across the beam; `x`, `y`, `z` choose it. Points outside the mesh are zero. Without cells, a Delaunay triangulation of the nodes is used, less its simplices with an edge more than 5 times the median.
  - **Cells:** tetrahedra, hexahedra, triangles and quadrilaterals of any degree (their corners are used). Hexahedra and quadrilaterals are read in VTK order (XDMF files) or DOLFINx's tensor-product order (`mesh.geometry.dofmap`), told apart by their shape.
  - **Phase:** the convention is read at the nodes, from the gradient of the fundamental's phase along the beam (nodes less than 0.45 wavelengths apart are needed), and an absolute-phase carrier is removed before the interpolation.
  - **Accuracy:** linear interpolation between nodes. A high-order solution loses its accuracy this way; `sample_dolfinx_planes` (frequency domain) or a `DolfinxProbe` on `plane_points` (time domain) evaluates it exactly in DOLFINx on the planes, for `from_harmonics` or `from_time_series`.
  - **Checked against:** the Gaussian beam at the nodes of a jittered tetrahedral mesh 0.5 mm (λ/3) apart gives the focal pressures of the beam to 0.5% and its power to 2%, with the cells or triangulated, along any axis, and as time series. 2D planar meshes, hexahedra in both node orders, and XDMF files laid out as DOLFINx writes them are tested too. `sample_dolfinx_planes` and `DolfinxProbe` have not been run.
- **2D fields.** A 2D field (one transverse axis) gives power and areas per metre. Its y widths and its thermal and mechanical indices are NaN.

## Layout

```
src/acoustic_outputs/
  grid.py, waveform.py      plane grid; waveforms and peak pressures from harmonics
  recorder.py, results.py   PlaneRecorder builds Results from planes of harmonics; metrics and files
  iec.py                    the IEC parameter functions
  preprocess/               frequency domain: frequency_domain.py (from_harmonics; formerly spectral.py),
                              axisymmetric_frequency_domain.py, unstructured_frequency_domain.py
                            time domain: time_domain.py (from_time_series, from_kwave), fdtd.py (from_fullwave),
                              axisymmetric_time_domain.py, unstructured_time_domain.py
                            solver formats: babelbrain.py, msound.py, focus.py (both domains), optimus.py (frequency)
                            _mesh.py (cells, point location, XDMF reading, DolfinxProbe), _common.py (harmonic fit)
  benchmarks/               IEC TS 63587 benchmarks: bowl.py (C.1, the focused bowl)
  compare.py, propagation.py   error norms (TS 63587 3.43-3.46); angular spectrum projection
  datasets.py, cli.py       dataset files and the acoustic-outputs command
  synthetic.py              a focused Gaussian beam with known metrics, and datasets of it
docs/iec_outputs.md         every IEC parameter, how it is computed, and its key in metrics.json
examples/                   benchmark_c1_bowl.py: benchmark C.1, built in or against your own solver's output
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
- **OptimUS** (`test_optimus.py`): the beam's fundamental as postprocess objects and as a `.mat` export, in 3D along each axis and from a plane through the axis, against the beam on a 3D grid; planes behind the source dropped; asymmetric planes warned of.
- **Real FOCUS and mSOUND output:** `tests/data/` holds small outputs of FOCUS 0.981 and mSOUND v0.1, made by `tests/data/make_fixtures.m` in GNU Octave 10. They are checked against physics: a piston's exact axis (FOCUS) and an angular-spectrum propagation (mSOUND).
- **Meshes, frequency domain** (`test_axisymmetric_frequency_domain.py`, `test_unstructured_frequency_domain.py`): the beam's harmonics on an (r, z) grid (retarded or absolute phase, either time convention), at the nodes of (r, z) and tetrahedral meshes (with cells or triangulated, beam along any axis, planar 2D), and in XDMF files as DOLFINx writes them, against the beam on a 3D grid.
- **Meshes, time domain** (`test_axisymmetric_time_domain.py`, `test_unstructured_time_domain.py`): the same fields as time series with noise until the wave arrives, on (r, z) grids, at mesh nodes and in XDMF files written at every step, give the metrics of their harmonics; a record too short to be steady is warned of; only the steps of the window are read from a file.
- **Mesh mechanics** (`test_mesh.py`): exact linear interpolation in tetrahedra and hexahedra (both node orders), no bridging of gaps by the triangulation, and the phase convention read at the nodes.
- **Benchmark C.1** (`test_benchmark_bowl.py`): the regularised axial formula against C.1 away from the focus and against C.2/C.3 at it, the flat-piston limit, the Rayleigh quadrature against the analytic axis (with and without absorption), the processing chain to round-off, convergence of the angular-spectrum model, the report files and the command line; the angular spectrum projector (exact for a periodic plane wave, both time conventions, back-projection) and the error norms.
- **Files and command line:** the `.npz`, `.mat` and HDF5 readers, and the `acoustic-outputs` command.
- **Real k-Wave run:** a k-wave-python CW run is compared with an angular-spectrum projection of its own first plane. They agree to within 6% of the peak.

Set `ACOUSTIC_OUTPUTS_NO_BEARTYPE=1` to turn off run-time type checking.

