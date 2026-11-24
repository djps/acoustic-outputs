# TODO

The aim of this repository is to support **IEC TS 63587 ED1** (*Ultrasonics – Field characterization – Measurement-based simulation in water and other media*). The reference here is the committee draft ACD#2 of July 2026. The repository should compute the simulation outputs that the TS and the standards it cites expect, and give worked examples of them.

The TS does not fix which outputs to report. It says to derive them from the application standards (5.3.2):
- HITU: IEC TS 62556, TR 62649 and 60601-2-62;
- diagnostic: 60601-2-37, 61157 and 62359;
- physiotherapy: 61689 and 63009;
- lithotripsy: 61846.

It does require these simulation outputs (Clauses 5, 6 and 8, and Annex F):
- benchmarks;
- convergence checks;
- comparison with measurements;
- an uncertainty budget.

Clause and annex numbers below refer to the draft. Check them against the published edition.

Legend: **[ ]** to do, **[~]** partly done, **[x]** done.

## Already in place

- [x] Field quantities from planes of harmonics (CW / steady state): p+, p−, I_ta, I_spta, ppsi, pii, power, beam widths and areas, f_awf, MI, TIs, side lobes, focal gain (`results.py`, `iec.py`, `docs/iec_outputs.md`).
- [x] Readers for the solver families the TS names in A.3:
  - k-Wave, fullwave, mSOUND, FOCUS, BabelBrain (BabelVisco) and OptimUS;
  - FEniCSx / finite-element output: axisymmetric, unstructured 3D and planar 2D, in both frequency and time domain (`preprocess/`).
- [x] Frequency-domain and time-domain routines in separate modules. The TS distinguishes CW (frequency-domain) and transient (time-domain) source conditions and regimes (7.3.2, A.3).
- [x] **Benchmark C.1 (bowl)** — `acoustic_outputs.benchmarks.bowl`, `acoustic-outputs --benchmark bowl`, `examples/benchmark_c1_bowl.py`.
  - Analytic axis, regular at the focus, and the focal values C.2/C.3.
  - Rayleigh quadrature for the whole field.
  - `compare_axis` / `compare_longitudinal` for any reader's `Results`.
  - Built-in angular-spectrum model with a convergence study and an Annex C report.
- [x] **Error norms** L2, L2_rel, L∞, L∞_rel (3.43–3.46) — `acoustic_outputs.compare`.
- [x] **Angular spectrum projector** between parallel planes (forward and back, both time conventions) — `acoustic_outputs.propagation`. It is the building block of 7.4.2 and Annex D.
- [x] Steady-state check for time-domain records (the residual of the harmonic fit). The TS asks for this explicitly (F.1.2, "simulation duration").

## Simulation outputs (5.3.2, 3.x definitions)

- [ ] **Beam maximum per IEC 61828 / TS 3.15–3.18.** The beam maximum point is the maximum of **ppsi** on the beam axis. For CW, ppsi is replaced by the mean-square pressure. Add `focal_quantity="ppsi"` (or a separate `beam_maximum` section) and report the beam maximum point, the beam maximum depth L_bm (−6 dB of ppsi along the axis) and the beam maximum value. The current default focus is the maximum of p+.
- [ ] **Beam maximum volume V_bm (3.18):** the volume where ppsi is above −6 dB of its value at the beam maximum point, for both −6 dB and −3 dB (5.3.2 names both). It can be accumulated plane by plane in `PlaneRecorder` (area above the level × dz), with no need to hold the 3D field. Also report its centroid. The skull benchmark (C.3, F.3.1) uses the centre of mass of the −6 dB focal volume as the focus position.
- [ ] **Shape of the focal region (5.3.2):** eccentricity / distortion of the −6 dB region, the principal longitudinal plane (3.51), the beamwidth midpoint (3.20) and FWHM (3.35).
- [ ] **Overlap of the focal region with a target region of interest (5.3.2):** take a mask or a box and report the fraction of V_bm inside it.
- [ ] **Spatial-peak temporal-peak pressure p_sptp (3.79) and its location:** check the names against `spatial_peak_p_positive`.
- [ ] **Local distortion parameter σ_q and the quasi-linear check (3.49, 3.66).** σ_q = 2π f_awf β z p_m / (ρ c³ √F_a). It uses:
  - p_m = (p_c + p_r)/2 at the p_sptp point (3.52);
  - local area factor F_a = √(A_SAeff / A_b), with A_b the 1/e² ppsi area (3.48).

  Report σ_q and a `quasi_linear` flag (σ_q < 0.5). Clause 7 requires quasi-linear source measurements, and 6.3 relates expected discrepancies to σ_q.
- [ ] **Acoustic-working frequencies (3.6):** add the magnitude-weighted f_awf (3.6.3, pulses only), f_p (3.6.4) and f_t (3.6.5) beside the arithmetic-mean f_awf.
- [ ] **Shock amplitude A_s** at the beam maximum (C.2, A.1). It belongs with the nonlinear waveform metrics.
- [ ] **Intensities from particle velocity (5.3.2).** Instantaneous and temporal-average intensity from p·v when a solver gives velocity, beside the derived (plane-wave) intensity (3.42). The exact angular-spectrum intensity already exists for one-way fields.

## Pulsed and burst fields (7.3.2, A.3, C.3)

The TS allows time-domain source conditions with the whole pulse recorded. Its skull benchmark (C.3) is a 3-cycle burst.
- [ ] **Burst / duty factor:** add `duty_factor` and burst parameters and scale the time-average quantities (I_ta, I_spta, power, TIs). On-time quantities are unchanged (TS 62556, 61689).
- [ ] **Transient recorder:** compute the 62127-1 pulse quantities directly from p(t) on each plane for short pulses. These are p+, p−, ppsi and pii over the pulse, pulse duration t_d, I_pa, a spectrum-based f_awf (and its magnitude-weighted form), and the prp / prr time averages. It applies to the time-domain readers (k-Wave, fullwave, FE time stepping, FOCUS `fnm_tsd`).
- [ ] **Frequency-domain pulses:** combine single-frequency runs (OptimUS, FOCUS CW) into p(t) before the transient recorder.

## Comparison metrics (3.43–3.46, 5.3.2, 6.3, 7.4.2, 8.1.3)

- [~] **Error norms:** L2, relative L2, L∞ and relative L∞ are done (`compare.py`), with linear interpolation along a line (`on_common_points`). Still to do: registration on planes and volumes, with axis orientation and offsets (5.3.2).
- [ ] **Compare with hydrophone line scans (7.3.2, 7.4.2, 8.1.1):** read measured scans along three orthogonal axes through the beam maximum, sample the simulation at the same positions, and report the error norms plus the differences in p+, p−, beam maximum position and widths.
- [ ] **Waveform comparison at the beam maximum (8.1.3):** temporal and spectral comparison, as a function of output level.
- [ ] **Normalised error / gamma comparison with combined uncertainties (F.2, last step).**

## Benchmarks (Clause 6, Annex C)

- [x] **C.1 — linear focused bowl, analytic on-axis solution.** Done (see above).
  - [ ] Run it through each reader with real solver output (FOCUS `fnm_cw` bowl, OptimUS `create_bowl`, k-Wave `makeBowl`, an FE model) and add those as example reports.
  - [ ] Add the TS's focal-plane intensity and focal-size estimates (C.1.2, Ref. [80]) once they are in the published text.
- [ ] **C.2 — nonlinear focused field in water with shocks:**
  - bowl: aperture 100 mm, radius of curvature 90 mm, 1 MHz;
  - water: c = 1485 m/s, β = 3.5, ρ = 998 kg/m³, δ = 4.33e-6 m²/s;
  - drive levels: p0 from 0.015 to 1.45 MPa;
  - equivalent flat source for KZK: aperture 114 mm, focused at 98 mm, pressure × 0.917.

  The reference table (p+, p−, A_s, focal sizes) is still empty in ACD#2. Add the case as a dataset or example (e.g. from pyWave3D / HIFU beam / k-Wave) once it is published.
- [ ] **C.3 — linear propagation through skull phantoms** (Robertson et al., k-Wave, DRUM data). It uses a 60 mm aperture, 75 mm radius-of-curvature transducer at 1 MHz with a 3-cycle burst. The reported outputs are:
  - spatial-peak positive pressure;
  - focal position (centre of mass of the −6 dB volume);
  - −6 dB focal volume.

  Reproduce Table C.3 from the DRUM data as an end-to-end example. It needs V_bm, its centroid and the transient recorder.
- [~] **Benchmark harness:** `benchmarks/` and `--benchmark bowl` exist for C.1. Generalise them when C.2 / C.3 are added (common report format, one CLI entry per case).

## Convergence (5.3.1, F.1.2)

- [ ] **Convergence helper:** take results at several resolutions (spatial and temporal, or number of harmonics), report how each chosen output changes, and record the final discretisation and the residual error.
- [ ] **Check the highest frequency supported** by the grid (points per wavelength at the highest harmonic), and warn when harmonics are under-resolved. Assess this under the most taxing conditions (5.3.1).

## Uncertainty (Annex F, normative)

- [ ] **Uncertainty budget module (F.2):**
  - list of quantities of interest;
  - uncertainty components with distributions (normal / rectangular);
  - sensitivity coefficients by ± perturbation (central difference);
  - contributions combined in quadrature, a coverage factor k = 2 and expanded uncertainty;
  - a budget table written next to `metrics.json`.
- [ ] **Monte Carlo propagation (F.2 b):** a driver that resamples inputs and collects the distribution of each output (GUM Supplement 1).
- [ ] **Hybrid / autodiff sensitivities (F.2 c, F.3.2):** an interface for gradients from differentiable solvers (j-Wave).
- [ ] **Worked example:** an uncertainty budget for a benchmark case, following F.3.3 / G.7.

## Source definition and scaling (Clause 7, Annex D)

- [ ] **Output-level scaling (7.3.3):** scale a linear source definition or result by A_n/A_v (pressure) or √(P_n/P_v) (power) for each drive level. This is linear only; nonlinear runs must be repeated at each level.
- [~] **Axisymmetric equivalent source (7.4.1.1, D.2):** fit aperture, radius of curvature and amplitude of a uniform bowl to three orthogonal line scans. The axisymmetric readers already exist; the fitting does not.
- [~] **2D hologram back-projection (7.4.1.2, D.3):** the projector exists (`propagation.angular_spectrum`, negative distances with evanescent waves removed). Still to do: angular-spectrum (or Rayleigh) back-projection of a measured plane to the source plane. Include:
  - window selection and sampling (≤ λ/2);
  - a power check against an independent measurement;
  - re-projection to the line-scan positions to evaluate the source (7.4.2).
- [ ] **Steady-cycle extraction from hydrophone waveforms (7.3.2, D.3.2.3, G.4.1):** amplitude and phase from an integer number of steady cycles. `HarmonicFit` can be reused here.

## Thermal outputs (5.3.2, Annex G)

- [~] **Heat source:** the heating rate exists (`alpha`). Export the heat-source term q = 2αI on the grid for thermal solvers.
- [ ] **Temperature rise and thermal dose:** max ΔT and its location, the time course, and CEM43 (3.24) from a supplied temperature history or a simple bioheat solver. Validate against embedded thermocouples (G.6.2).

## Records and reporting (5.3, 6.2, 8)

- [ ] **Provenance in `metrics.json` (5.3, required):**
  - all inputs: source, boundary conditions, material properties, solver and its settings / function calls;
  - software versions (solver and this package);
  - hardware (processor, memory, parallel strategy).
- [ ] **Simulation report:** one document per case listing the simulation objectives (5.1), the model selection (5.2, A.1), the benchmark results (6.3), the validation comparisons (8.1, 8.2) and the uncertainty budget (F.2).
- [ ] **Cavitation note (5.2):** at or above the cavitation threshold, the simulated field is the substrate field without bubbles. State this in reports of histotripsy and boiling-histotripsy cases.

## Examples (the repository's role as a set of worked examples)

- [ ] `examples/` folder with one notebook or script per route:
  - C.1 bowl → each solver reader → metrics vs analytic;
  - C.2 nonlinear bowl in water at several drive levels (σ_q, quasi-linear flag, A_s);
  - C.3 skull phantom from the DRUM data;
  - an Annex G-style measurement-based thermal case (source from a planar scan → simulation → ΔT, CEM43 → uncertainty budget).
- [ ] Map every output to the standard it comes from (extend `docs/iec_outputs.md` with the TS 63587 terms: beam maximum volume, σ_q, error norms, CEM43, uncertainty).

## Housekeeping

- [ ] Keep the TS draft itself out of the repository (IEC copyright). Cite its clauses only.
- [ ] Re-check clause numbers and definitions when the TS is published (C.2 is marked as still to be updated).
