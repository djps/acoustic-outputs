# Acoustic field outputs under the IEC standards, and how to compute them from planes of harmonics

This note lists the acoustic field parameters that the IEC standards for medical and therapeutic ultrasound ask to be determined or declared. For each one, it explains how `acoustic_outputs` computes it from planes of harmonics, and where it is written. Every `iec.<key>` and metric named here is produced by `Results.metrics()`, whichever solver the planes came from. "The march" means the sequence of planes, in increasing $z$, that a solver's output is reduced to; a spatial-marching solver produces them directly. `godunov_from_z` is NaN unless a marching solver supplies it.

Every parameter that can be computed is recorded in every run. The exceptions are pulse duration and the scanned-mode indices, which a single periodic beam cannot represent.

The standards define these quantities for measurements with a hydrophone in water. A simulation gives the same quantities of the modelled field in the modelled medium. The definitions below are paraphrased. Edition, clause and exact wording should be checked against the current text of each standard before anything is declared from a simulation.

The equations are written in LaTeX for KaTeX: `$...$` inline and `$$...$$` for display.

## The standards

| Standard | Scope | Main field parameters |
| --- | --- | --- |
| IEC 62127-1 (hydrophone measurement and characterisation of medical ultrasonic fields up to 40 MHz) | the basic definitions used by all the others | $p_c$, $p_r$, intensities (instantaneous, pulse-average, temporal-average, spatial-peak), pulse-pressure-squared and pulse-intensity integrals, beam axis, beam widths and areas, acoustic working frequency, pulse duration |
| IEC 61828 (focusing transducers: definitions and measurement methods for the transmitted field) | focused fields | beam maximum, focal plane, focal length (geometric and measured), focal widths, focal area, focal depth (focal length along the beam), focal gain, beam-alignment axis |
| IEC TS 62556 (specification and measurement of high-intensity therapeutic ultrasound (HITU) fields) | HITU transducers, including nonlinear propagation | focal $p_c$ and $p_r$, focal intensities, −6 dB focal area and depth, power through the focal area, the conditions for extrapolating from low drive |
| IEC 62555 (power measurement for HITU transducers) | total acoustic power | acoustic power (radiation force or calorimetry), and its variation with drive level |
| IEC 61161 (ultrasonic power measurement in liquids up to 40 MHz) | radiation-force balances | total acoustic power |
| IEC 62359 (thermal and mechanical exposure indices) | diagnostic ultrasound | mechanical index $\mathrm{MI}$, thermal indices $\mathrm{TIS}$, $\mathrm{TIB}$, $\mathrm{TIC}$, with the derating of 0.3 dB cm⁻¹ MHz⁻¹ |
| IEC 61689 (physiotherapy systems, 0.5 to 5 MHz) | unfocused physiotherapy transducers | effective radiating area $A_{ER}$, beam non-uniformity ratio $R_{BN}$, beam type, effective intensity $I_e$ |
| IEC 60601-2-62 (safety of HITU equipment) | HITU equipment | the output parameters the manufacturer declares, from the standards above |
| IEC TS 63587 (measurement-based simulation in water and other media) | simulations of the fields above | the outputs of the application standards, beam maximum depth $L_{bm}$ and volume $V_{bm}$, error norms against a reference, the local distortion parameter $\sigma_q$, shock amplitude $A_s$ |

## How the field is represented

The field is held as the harmonics $P_n(x, y)$, $n = 1, \dots, K$, of a periodic waveform in retarded time $\tau = t - z/c_0$, on planes taken in turn along the beam. On each plane ($z = z_0 + j\,\Delta z$):

- the waveform at any point is the band-limited series $p(\tau) = \mathrm{Re} \sum_{n=1}^{K} P_n e^{i n \omega \tau}$. It is evaluated with an inverse FFT, oversampled (4 samples per period per active harmonic) so that peaks between samples are not missed;
- spatial derivatives and the angular spectrum $\hat{P}_n(k_x, k_y)$ come from one 2D FFT per harmonic;
- the axial component of the particle velocity of each plane-wave component is $\hat{v}_{z,n} = \hat{P}_n k_{z,n} / (\rho_0 n \omega)$, the exact one-way relation.

Everything below uses only these three operations on the planes, so a marching solver can record as it goes, at no extra propagation cost. The FFTs treat each plane as periodic in $x$ and $y$, and the exact intensity treats the field as one-way (no reflections), which limits where results are valid: the field must stay away from the lateral edges of the planes.

**Other solvers.** The same computations apply to any solver whose output is reduced to such planes of harmonics, and `acoustic_outputs.preprocess` does that reduction:
- frequency-domain output is already in this form (`frequency_domain`, and `axisymmetric_frequency_domain` and `unstructured_frequency_domain` for meshes);
- time-domain pseudo-spectral (k-wave-python) and FDTD (fullwave25) records are fitted for their harmonics over a steady window (`time_domain`, `fdtd`, and `axisymmetric_time_domain` and `unstructured_time_domain` for meshes).

`PlaneRecorder` then records every parameter below in the same way. The one-way exact intensity treats each plane's field as travelling in $+z$, which holds for a beam away from reflectors. Fields from a time-domain solver may also contain reflected waves, so the exact intensity is then an approximation.

**Continuous wave versus pulses.** The planes represent a periodic (CW or tone-burst steady-state) field. For pulsed fields, the "period" must cover the pulse repetition period, or at least a time window containing the whole pulse, with enough harmonics for its bandwidth. Then pulse quantities (pulse integrals, pulse duration, pulse-average intensity) follow from the waveform at each point. For CW fields, pulse-average equals temporal-average.

## Mathematical definitions

These are the formulae that `acoustic_outputs` evaluates. They are the standards' definitions applied to a periodic field held as $K$ harmonics on planes.

### Notation

| Symbol | Meaning |
| --- | --- |
| $f_0$, $\omega = 2\pi f_0$, $T = 1/f_0$ | fundamental frequency, angular frequency and period |
| $K$ | number of harmonics |
| $P_n(x, y; z)$ | complex amplitude of harmonic $n$ at $(x, y)$ on the plane at $z$ |
| $\hat{P}_n(k_x, k_y; z)$ | its angular spectrum (2D FFT over $x$, $y$) |
| $\rho_0$, $c_0$, $Z_0 = \rho_0 c_0$ | density, sound speed and characteristic impedance of the medium |
| $k_n = n\omega / c_0$ | wavenumber of harmonic $n$ |
| $\alpha_n$ | absorption of the medium at harmonic $n$ [Np/m] |
| $\Delta x$, $\Delta y$, $\Delta A = \Delta x\,\Delta y$ | grid steps and cell area on a plane |
| $z_0$ | position of the source plane |
| $\mathbf{x}_f = (x_f, y_f, z_f)$ | the focus (beam maximum) |
| $\psi$ | the focal quantity: $p_{+}$, $\lvert p_{-} \rvert$ or $\lvert P_1 \rvert$ (`Metrics.focal_quantity`) |
| $\psi_f = \psi(\mathbf{x}_f)$ | its value at the focus |
| $L$ | a level in dB below the focal value ($L < 0$, from `Metrics.levels_dB`) |
| $p_0$ | source pressure amplitude |
| $R$ | geometric focal length (radius of curvature) of the source |

The waveform at a point, and its inverse, are

$$
p(\tau) = \mathrm{Re} \sum_{n=1}^{K} P_n\, e^{i n \omega \tau},
\qquad
P_n = \frac{2}{T} \int_0^T p(\tau)\, e^{-i n \omega \tau}\, \mathrm{d}\tau .
$$

### Pressure

Peak-compressional and peak-rarefactional pressure at a point (IEC 62127-1):

$$
p_{+} = p_c = \max_{0 \le \tau < T} p(\tau),
\qquad
p_{-} = \min_{0 \le \tau < T} p(\tau),
\qquad
p_r = -p_{-} = \lvert p_{-} \rvert .
$$

The maxima are found on $4K$ samples per period, then refined by Newton steps on the series itself.

Spatial-peak values, over all recorded planes:

$$
p_{+,\mathrm{sp}} = \max_{x, y, z} p_{+}(x, y, z),
\qquad
p_{-,\mathrm{sp}} = \min_{x, y, z} p_{-}(x, y, z).
$$

Harmonic amplitudes: $\lvert P_n \rvert$, $n = 1, \dots, K$.

Hydrophone-averaged harmonics, for a sensitive element of diameter $d_h$ (a disc $D$ of $N_D$ grid points):

$$
\bar{P}_n(x, y) = \frac{1}{N_D} \sum_{(x', y') \in D(x, y)} P_n(x', y'),
\qquad
D(x, y) = \left\{ (x', y') : \sqrt{(x' - x)^2 + (y' - y)^2} \le d_h / 2 \right\},
$$

from which $p_c$, $p_r$ and $\lvert \bar{P}_n \rvert$ at the focus are computed as above.

### Acoustic working frequency

Arithmetic-mean frequency, from the lowest and highest harmonics $n_1$ and $n_2$ whose amplitude is within 3 dB of the largest (IEC 62127-1):

$$
f_{awf} = \frac{n_1 + n_2}{2}\, f_0,
\qquad
n_1 = \min \mathcal{N}, \quad n_2 = \max \mathcal{N}, \quad
\mathcal{N} = \left\{ n : \lvert P_n \rvert \ge \frac{\max_m \lvert P_m \rvert}{\sqrt{2}} \right\}.
$$

Zero-crossing frequency, with $N_{zc}$ the number of zero crossings of $p(\tau)$ in one period:

$$
f_{zc} = \frac{N_{zc}}{2}\, f_0 .
$$

Where it is used below (derating, $\mathrm{MI}$, $\mathrm{TIS}$), $f_{awf}$ is the arithmetic-mean frequency at the focus.

### Intensity, pulse integrals and power

Instantaneous intensity (plane-wave form):

$$
i(\tau) = \frac{p(\tau)^2}{Z_0} .
$$

Pulse-pressure-squared integral and pulse-intensity integral over one period, by Parseval:

$$
\mathrm{ppsi} = \int_0^T p(\tau)^2\, \mathrm{d}\tau = \frac{T}{2} \sum_{n=1}^{K} \lvert P_n \rvert^2,
\qquad
\mathrm{pii} = \frac{\mathrm{ppsi}}{Z_0} .
$$

Temporal-average intensity, plane-wave form (as a hydrophone measurement reports it):

$$
I_{ta} = \frac{1}{T} \int_0^T i(\tau)\, \mathrm{d}\tau = \frac{\mathrm{pii}}{T} = \sum_{n=1}^{K} \frac{\lvert P_n \rvert^2}{2 Z_0} .
$$

Exact time-averaged axial (active) intensity, with the axial particle velocity from the angular spectrum (evanescent components carry no power):

$$
I_{z} = \frac{1}{2} \sum_{n=1}^{K} \mathrm{Re}\left( P_n\, v_{z,n}^{\ast} \right),
\qquad
v_{z,n} = \mathcal{F}^{-1}\left[ \frac{k_{z,n}}{\rho_0\, n \omega}\, \hat{P}_n \right],
\qquad
k_{z,n} =
\begin{cases}
\sqrt{k_n^2 - k_x^2 - k_y^2}, & k_x^2 + k_y^2 \le k_n^2, \\
0, & \text{otherwise.}
\end{cases}
$$

For a single plane wave at angle $\theta$ to $z$, $I_z = I_{ta} \cos\theta$: the two forms differ by the obliquity of the waves.

Spatial-peak temporal-average intensity, and the pulse-average forms for a continuous wave:

$$
I_{spta} = \max_{x, y, z} I_{ta}(x, y, z),
\qquad
I_{pa} = I_{ta},
\qquad
I_{sppa} = I_{spta},
\qquad
\mathrm{pii}_{sp} = I_{spta}\, T .
$$

Acoustic output power through the plane at $z$, plane-wave and exact (IEC 61161, 62555):

$$
P(z) = \Delta A \sum_{i, j} I_{ta}(x_i, y_j, z),
\qquad
P_{\mathrm{exact}}(z) = \Delta A \sum_{i, j} I_{z}(x_i, y_j, z).
$$

Power through the focal area, with $\mathcal{R}_L$ the set of points of the focal plane above the first level $L$ of `levels_dB` (IEC TS 62556):

$$
P_{L} = \Delta A \sum_{(i, j) \in \mathcal{R}_L} I_{ta}(x_i, y_j, z_f),
\qquad
\mathcal{R}_L = \left\{ (i, j) : \psi(x_i, y_j, z_f) \ge 10^{L/20}\, \psi_f \right\}.
$$

Heating rate (heat deposited per unit volume, not an IEC index):

$$
q = \sum_{n=1}^{K} 2 \alpha_n I_n = \sum_{n=1}^{K} \frac{\alpha_n \lvert P_n \rvert^2}{Z_0},
\qquad
I_n = \frac{\lvert P_n \rvert^2}{2 Z_0} .
$$

### Beam axis and focus

Beam centre on each plane: the intensity centroid of the −6 dB region,

$$
x_c(z) = \frac{\sum_{i, j} w_{ij}\, x_i}{\sum_{i, j} w_{ij}},
\qquad
y_c(z) = \frac{\sum_{i, j} w_{ij}\, y_j}{\sum_{i, j} w_{ij}},
\qquad
w_{ij} =
\begin{cases}
I_{ta}(x_i, y_j, z), & I_{ta}(x_i, y_j, z) \ge \tfrac{1}{4} \max I_{ta}(\cdot, \cdot, z), \\
0, & \text{otherwise.}
\end{cases}
$$

Beam-alignment axis: the least-squares line through the centres, and its tilt from $z$:

$$
x = x_0 + a_x (z - z_0),
\qquad
y = y_0 + a_y (z - z_0),
\qquad
\theta = \arctan \sqrt{a_x^2 + a_y^2} .
$$

Focus (beam maximum): the point of the maximum of the focal quantity on the axis,

$$
z_f = \arg\max_{z} \psi(x_a, y_a, z),
$$

where $(x_a, y_a)$ is the transducer axis. The focal plane is the plane $z = z_f$.

Focal shift and focal gain:

$$
\Delta z_f = (z_f - z_0) - R,
\qquad
G = \frac{\lvert P_1(\mathbf{x}_f) \rvert}{p_0} .
$$

### Beam widths, focal depth and focal area

The level of the focal quantity at $L$ dB is

$$
\psi_L = 10^{L/20}\, \psi_f .
$$

On a line through the focus, the level crossings are found by linear interpolation between grid points. Along $x$ in the focal plane, with $\psi(x) = \psi(x, y_f, z_f)$:

- the beamwidth (IEC 62127-1, 61828) uses the outermost crossings, so side lobes above the level are inside it:

$$
w_x(L) = x_{\max} - x_{\min},
\qquad
x_{\min} = \min \{ x : \psi(x) \ge \psi_L \},
\quad
x_{\max} = \max \{ x : \psi(x) \ge \psi_L \};
$$

- the main-lobe width uses the crossings either side of the focus, $x^{-} < x_f < x^{+}$, of the contiguous region where $\psi \ge \psi_L$:

$$
w_x^{\mathrm{ml}}(L) = x^{+} - x^{-} .
$$

$w_y(L)$ and $w_y^{\mathrm{ml}}(L)$ are the same along $y$. Each is NaN if the region reaches an end of the line.

Focal depth (focal length along the axis) is the smallest distance between two crossings on the axis, either side of the focus. At −6 dB it is the beam maximum depth of IEC 61828 and TS 63587 (3.16):

$$
L_{bm} = d(-6\ \mathrm{dB}),
\qquad
d(L) = z^{+} - z^{-},
\qquad
z^{-} < z_f < z^{+},
\quad
\psi(z^{\pm}) = \psi_L .
$$

Focal (beam) area: all the points of the focal plane above the level, and the main lobe's area:

$$
A(L) = \Delta A\, \#\, \mathcal{R}_L,
\qquad
A^{\mathrm{ml}}(L) = \Delta A\, \#\, \mathcal{C}_L,
$$

where $\# $ counts points and $\mathcal{C}_L \subseteq \mathcal{R}_L$ is the 4-connected region of $\mathcal{R}_L$ that contains the focus.

### Side lobes and secondary maxima

On a profile $\psi_j$ through the focus (index $j_f$), the main lobe extends from $j_f$ on each side while $\psi_j \ge 10^{-6/20} \psi_f$, and then further while $\psi_j$ is still decreasing, up to its first minimum. A side lobe (or, along the axis, a prefocal or post-focal maximum) is a local maximum outside the main lobe,

$$
\psi_{j-1} < \psi_j \ge \psi_{j+1},
\qquad
\text{at the level}
\qquad
L_j = 20 \log_{10} \frac{\psi_j}{\psi_f}\ \mathrm{dB},
$$

with maxima below −120 dB left out as round-off.

### Exposure indices (IEC 62359)

Derating factor for pressure, with $a = 0.3$ dB cm⁻¹ MHz⁻¹ (`Metrics.derating`), $f_{awf}$ in MHz and $z - z_0$ in cm:

$$
\delta(z) = 10^{-a\, f_{awf}\, (z - z_0) / 20},
\qquad
p_{r,\alpha} = \delta\, p_r,
\qquad
I_{ta,\alpha} = \delta^2 I_{ta},
\qquad
\mathrm{pii}_{\alpha} = \delta^2\, \mathrm{pii},
\qquad
P_{\alpha} = \delta^2 P .
$$

Aperture and break-point depth, from the −12 dB output beam area of the source plane:

$$
A_{aprt} = \Delta A\, \#\left\{ (i, j) : I_{ta}(x_i, y_j, z_0) \ge 10^{-1.2} \max I_{ta}(\cdot, \cdot, z_0) \right\},
\qquad
D_{eq} = \sqrt{\frac{4 A_{aprt}}{\pi}},
\qquad
z_{bp} = 1.5\, D_{eq} .
$$

Mechanical index, at the depth of the maximum of the derated pulse-intensity integral beyond the break point:

$$
\mathrm{MI} = \frac{p_{r,\alpha}(z_{MI}) / (1\ \mathrm{MPa})}{\sqrt{f_{awf} / (1\ \mathrm{MHz})}},
\qquad
z_{MI} = \arg\max_{z - z_0 \ge z_{bp}} \mathrm{pii}_{\alpha}(z) .
$$

The value in water, without derating, at the focus is $p_r(\mathbf{x}_f) / \sqrt{f_{awf}}$ with the same units.

Thermal indices (unscanned mode), with $P = P(z_0)$ the source power in mW, $f_{awf}$ in MHz, $I_{ta,\alpha}$ in mW cm⁻² on the axis, and $C_{TIS} = 210$ mW MHz, $C_{TIB1} = 50$ mW cm⁻¹, $C_{TIB2} = 4.4$ mW, $C_{TIC} = 40$ mW cm⁻¹:

$$
\mathrm{TIS} =
\begin{cases}
\dfrac{P\, f_{awf}}{C_{TIS}}, & A_{aprt} \le 1\ \mathrm{cm}^2, \\[2ex]
\dfrac{f_{awf}}{C_{TIS}} \displaystyle\max_{z - z_0 \ge z_{bp}} \min\left( P_{\alpha}(z),\ I_{ta,\alpha}(z) \times 1\ \mathrm{cm}^2 \right), & A_{aprt} > 1\ \mathrm{cm}^2,
\end{cases}
$$

$$
\mathrm{TIB} = \max_{z - z_0 \ge z_{bp}} \min\left( \frac{\sqrt{P_{\alpha}(z)\, I_{ta,\alpha}(z)}}{C_{TIB1}},\ \frac{P_{\alpha}(z)}{C_{TIB2}} \right),
\qquad
\mathrm{TIC} = \frac{P}{D_{eq}\, C_{TIC}} .
$$

### Physiotherapy (IEC 61689)

On the plane at the face distance $z_{face}$ (`Metrics.face_distance`, default 0.3 cm), sort the values of $I_{ta}$ in decreasing order, $I_{(1)} \ge I_{(2)} \ge \dots$, and take the fewest that hold 75 % of the total:

$$
A_{BCS} = N_{75}\, \Delta A,
\qquad
N_{75} = \min \left\{ N : \sum_{m=1}^{N} I_{(m)} \ge 0.75 \sum_{m} I_{(m)} \right\}.
$$

Then

$$
A_{ER} = F_{AC}\, A_{BCS}, \quad F_{AC} = 1.333,
\qquad
I_e = \frac{P}{A_{ER}},
\qquad
R_{BN} = \frac{I_{spta}}{I_e} .
$$

### Nonlinear and shock parameters

Degree of nonlinear distortion at the focus:

$$
\frac{p_c}{p_r},
\qquad
\frac{\lvert P_2 \rvert}{\lvert P_1 \rvert} .
$$

Steepness, and the shock amplitude read from the waveform, with $\tau^{\ast}$ the time of the steepest rise:

$$
S = \frac{1}{\omega \lvert P_1 \rvert} \max_{\tau} \frac{\mathrm{d}p}{\mathrm{d}\tau},
\qquad
A_s^{\mathrm{wf}} = p\left( \tau^{\ast} + \frac{T}{2K} \right) - p\left( \tau^{\ast} - \frac{T}{2K} \right),
\qquad
\tau^{\ast} = \arg\max_{\tau} \frac{\mathrm{d}p}{\mathrm{d}\tau} .
$$

$S = 1$ for a sinusoid. The waveform is called shocked when $S > 3$ and $A_s^{\mathrm{wf}} > \lvert P_1 \rvert$. $A_s^{\mathrm{wf}}$ includes the Gibbs overshoot of the truncated series, up to about 18 %.

Shock parameters from the spectrum. A jump $\Delta p$ in the waveform gives $\lvert P_n \rvert \to \Delta p / (\pi n)$. Over the band $n_a \le n \le n_b$, with $n_a = \max(2, \lfloor K/8 \rfloor)$ and $n_b = \max(4, \lfloor K/2 \rfloor)$, the local spectral exponent $s$ is the least-squares slope

$$
\log \lvert P_n \rvert \approx c - s \log n,
\qquad
n_a \le n \le n_b,
$$

and the band level and the shock amplitude are

$$
\ell_b = \frac{n_a \lvert P_{n_a} \rvert}{\lvert P_1 \rvert},
\qquad
A_s = \pi\, \operatorname*{median}_{n_a \le n \le n_b} \left( n \lvert P_n \rvert \right).
$$

The point is shocked when $s \le 4/3$ and $\ell_b \ge 0.1$; $A_s$ is NaN otherwise. $s = 4/3$ where the Fubini solution forms a shock ($\sigma = 1$, $\lvert P_n \rvert = 2 J_n(n) / n \sim n^{-4/3}$), and $s = 1$ for a developed shock.

Shock-formation distance: the first $z$ on the axis where the point is shocked, interpolated linearly in $s$ between the plane before, $z_{j-1}$, and the first shocked plane, $z_j$:

$$
z_{sh} = z_{j-1} + t\,(z_j - z_{j-1}),
\qquad
t = \frac{s_{j-1} - 4/3}{s_{j-1} - s_j}, \quad 0 \le t \le 1,
$$

and its distance to the focus is $z_f - z_{sh}$.

### Error norms (IEC TS 63587, 3.43 to 3.46)

For simulated values $x_i$ and reference values $\hat{x}_i$ at the same $N$ positions (complex amplitudes, magnitudes, or any field quantity on a line, plane or volume), computed by `acoustic_outputs.compare`:

$$
L_2 = \sqrt{\sum_{i=1}^{N} \lvert x_i - \hat{x}_i \rvert^2},
\qquad
L_{2,\mathrm{rel}} = \frac{L_2}{\sqrt{\sum_{i=1}^{N} \lvert \hat{x}_i \rvert^2}},
$$

$$
L_{\infty} = \max_{i} \lvert x_i - \hat{x}_i \rvert,
\qquad
L_{\infty,\mathrm{rel}} = \frac{L_{\infty}}{\max_{i} \lvert \hat{x}_i \rvert} .
$$

### Defined in IEC TS 63587, not yet computed

These are on the roadmap in `TODO.md`. They are paraphrased from the draft, so check them against the published text.

- Beam maximum (IEC 61828): the maximum of $\mathrm{ppsi}$ on the beam axis, $\mathrm{ppsi}_{bm}$. For a continuous wave, $\mathrm{ppsi}$ is replaced by the mean-square pressure $\overline{p^2} = \tfrac{1}{2} \sum_n \lvert P_n \rvert^2$. $L_{bm}$ is then the −6 dB extent of $\mathrm{ppsi}$ along the axis, rather than of $\psi$.
- Beam maximum volume (3.18): the volume where $\mathrm{ppsi}$ is above the level, accumulated plane by plane,

$$
V_{bm}(L) = \sum_{z} \Delta z\, \Delta A\, \#\left\{ (i, j) : \mathrm{ppsi}(x_i, y_j, z) \ge 10^{L/10}\, \mathrm{ppsi}_{bm} \right\},
$$

  for $L = -6$ and $-3$ dB, with its centroid as the focus position (benchmark C.3).
- Mean peak pressure at the spatial-peak temporal-peak point (3.52): $p_m = (p_c + p_r)/2$.
- Local area factor (3.48): $F_a = \sqrt{A_{SAeff} / A_b}$, with $A_{SAeff}$ the effective area of the source aperture and $A_b$ the area where $\mathrm{ppsi}$ is above $1/e^2$ of its maximum.
- Local distortion parameter, with $\beta$ the coefficient of nonlinearity and $z$ the distance from the source to the point of $p_m$; the field is quasi-linear when $\sigma_q < 0.5$:

$$
\sigma_q = \frac{2 \pi f_{awf}\, \beta\, z\, p_m}{\rho_0 c_0^3 \sqrt{F_a}} .
$$

## Parameters

Status means: **recorded**, `acoustic_outputs` computes it in every run and writes it; **not possible**, a periodic field on planes cannot represent it (see the last column for why and what is given instead).

Where things are written:

- `metrics.json`, with the IEC parameters under its `iec` key (computed by `acoustic_outputs.iec`, through `Results.iec_metrics()`);
- `axial.txt`, the beam axis;
- `planes.txt`, one line per plane;
- `focal_plane.txt`;
- `results.npz`, which holds every recorded array;
- the point files from `Solver.Output.xyz`.

The options that control these outputs are under `Solver.Metrics`: `focal_quantity`, `levels_dB`, `harmonics`, `hydrophone_diameter`, `derating` and `face_distance`.

### Pressure

| Parameter | Definition (paraphrased) | Computation from the planes | Status and where |
| --- | --- | --- | --- |
| Peak-compressional acoustic pressure $p_c$ ($p_{+}$) | maximum positive instantaneous pressure at a point (62127-1) | $\max_\tau p(\tau)$ of the oversampled waveform; per point on the axis, along the $x$ and $y$ lines, over each plane and as the running maximum over $z$ | recorded: `iec.p_c`, `focus.p_positive`, `axial.txt` p+, `planes.txt` max p+, `max_p_plus` |
| Peak-rarefactional acoustic pressure $p_r$ ($p_{-}$) | magnitude of the most negative instantaneous pressure (62127-1) | $\min_\tau p(\tau)$ of the waveform, as above | recorded: `iec.p_r`, `focus.p_negative`, `axial.txt` p−, `planes.txt` min p− |
| Spatial-peak $p_c$ and $p_r$ | maxima of $p_c$ and $p_r$ over the field | running maxima over all planes, with the position of each | recorded: `spatial_peak_p_positive` ($x$, $y$, $z$), `spatial_peak_p_negative` ($z$) |
| Harmonic amplitudes | amplitude of each frequency component, which shows the degree of nonlinear distortion (62556) | the marched variables themselves, $\lvert P_n \rvert$ | recorded: `focus.harmonics`, `axial.txt`, `focal_plane.txt` |
| Acoustic working frequency $f_{awf}$ | arithmetic-mean frequency (the mean of the −3 dB band edges of the spectrum) and zero-crossing frequency $f_{zc}$ (62127-1) | band edges of the line spectrum $\lvert P_n \rvert$ at the focus; zero crossings per period of the oversampled waveform at the focus | recorded: `iec.acoustic_working_frequency` |
| Pressure waveform at a point | the time series that the parameters above are computed from | inverse FFT of the harmonics at the nearest grid point, on the nearest plane | recorded: the point files ($t$, $p$, $p^2/Z_0$) |
| Hydrophone-averaged values | what a hydrophone with a sensitive element of finite diameter $d_h$ reports (62127-1 spatial averaging) | the harmonics of the focal plane averaged over a disc of that diameter (FFT convolution), $\bar{P}_n$, then $p_c$, $p_r$ and $\lvert \bar{P}_n \rvert$ at the focus | recorded when `Metrics.hydrophone_diameter` > 0: `iec.hydrophone` |

### Intensity and power

| Parameter | Definition (paraphrased) | Computation from the planes | Status and where |
| --- | --- | --- | --- |
| Instantaneous intensity $i$ | $p\,v$; in measurements taken as $p^2/Z_0$ (62127-1) | $p(\tau)^2/Z_0$ from the waveform | recorded: third column of the point files |
| Temporal-average intensity $I_{ta}$ | average of the instantaneous intensity over the repetition period | plane-wave form $\sum_n \lvert P_n \rvert^2/(2 Z_0)$, as hydrophone measurements report it. Exact active intensity along $z$, $I_z = \sum_n \tfrac{1}{2} \mathrm{Re}(P_n v_{z,n}^{\ast})$, with $v_{z,n}$ from the angular spectrum (`iec.active_intensity`); the two differ by the obliquity of the waves | recorded: `iec.I_ta_focus` and `iec.I_ta_focus_exact`, `axial.txt` I, `focal_plane.txt` I |
| Spatial-peak temporal-average intensity $I_{spta}$ | maximum of $I_{ta}$ over the field | maximum of $I_{ta}$ over each plane, then over $z$ | recorded: `iec.I_spta`, `spatial_peak_intensity` (value, $z$), `planes.txt` max I, `iec.I_spta_focal_plane_exact` |
| Pulse-pressure-squared integral ($\mathrm{ppsi}$) and pulse-intensity integral ($\mathrm{pii}$) | $\int p^2\, \mathrm{d}t$ and $\int p^2/Z_0\, \mathrm{d}t$ over the pulse | over one period by Parseval: $\tfrac{T}{2} \sum_n \lvert P_n \rvert^2$ and that over $Z_0$ | recorded per period: `iec.ppsi_focus`, `iec.pii_focus`, `iec.pii_spatial_peak`, `axial.txt` pii |
| Pulse-average intensity $I_{pa}$, and $I_{sppa}$ | $\mathrm{pii}$ divided by the pulse duration, and its spatial peak | for a continuous wave, $I_{pa} = I_{ta}$ | recorded: `iec.I_sppa` ($= I_{spta}$, as the field is continuous) |
| Pulse duration $t_d$, pulse repetition period | $1.25 \times$ the time between 10 % and 90 % of the $\mathrm{ppsi}$ | **not possible**: the planes hold one period of a periodic (CW) field, so the field has no pulse. A tone burst long enough to reach steady state is the CW field. A short pulse would need the period to cover the whole repetition period, with enough harmonics for its bandwidth | `iec.field` states that the field is CW |
| Acoustic output power $P$ | total power through a plane (61161, 62555) | plane-wave estimate $\Delta A \sum_{i,j} \sum_n \lvert P_n \rvert^2/(2 Z_0)$ on every plane; exact power $\Delta A \sum_{i,j} I_z$ (evanescent components carry none) on the source, focal and last planes | recorded: `planes.txt` power, `iec.power` (`source`, `focal_plane`, `end`, and `exact`) |
| Power through the focal area $P_L$ | the part of the power that reaches the focus (62556) | sum of $I_{ta}\, \Delta A$ over the focal area $\mathcal{R}_L$: all the points of the focal plane above the first of `levels_dB` | recorded: `iec.power_in_focal_area_6dB` and `..._exact` |

### Beam geometry and focus (IEC 61828, 62556)

| Parameter | Definition (paraphrased) | Computation from the planes | Status and where |
| --- | --- | --- | --- |
| Beam axis | the line through the beam centre on planes along $z$ (beam-alignment axis) | the intensity centroid $(x_c, y_c)$ of the −6 dB region of every plane, then a least-squares line through them. The profiles are recorded on the transducer axis (the plane centre) | recorded: `planes.txt` beam centre x, y; `iec.beam_alignment_axis` (intercepts $x_0$, $y_0$, slopes $a_x$, $a_y$, tilt $\theta$); `axial.txt` along the transducer axis |
| Beam maximum / focus $\mathbf{x}_f$ | point of the maximum of the chosen quantity on the beam axis (pressure focus or intensity focus) | the focal quantity $\psi$ on the axis is compared at every recorded plane with the best so far, and that plane is kept (`focal_quantity`: `p_positive`, `p_negative` or `first_harmonic`). The intensity maximum is reported too | recorded: `focus` ($x$, $y$, $z$, index, $p_{+}$, $p_{-}$, harmonics, intensity, heating rate), `axis_peak_intensity` |
| Focal plane | the plane through the focus, perpendicular to the axis | the copy of the harmonics on the plane $z = z_f$ | recorded: `focal_plane.txt`, `focal_plane` in `results.npz` |
| Plane of the spatial peak | where the field maximum lies if it is not on the axis | as the focal plane, for the plane maximum of $p_{+}$ | recorded: `peak_plane`, `spatial_peak_p_positive.off_axis` |
| Focal length, beam-maximum distance, focal shift $\Delta z_f$ | distance from the source to the focus, and its difference from the geometric focal length $R$ | $z_f - z_0$, and $\Delta z_f = z_f - z_0 - R$ | recorded: `focus.z`, `focal_shift` |
| Focal widths $w_x(L)$, $w_y(L)$ (−3, −6, −12, −20 dB) | beamwidth in the focal plane at the level, in two orthogonal directions: the **greatest** distance between two points on the line where the level is reached (62127-1, 61828), so side lobes above the level are inside it | the outermost level crossings along the $x$ and $y$ lines through the focus, interpolated linearly; NaN if the line is above the level at its ends. The main lobe's width $w^{\mathrm{ml}}$ (the crossings either side of the focus) is given too: the two differ only below the level of the highest side lobe, e.g. at −20 dB for a focused bowl whose first side lobes are near −17 dB | recorded at every level in `levels_dB`: `focal_width_x_6dB`, `focal_width_y_6dB`, …, and `focal_width_x_6dB_main_lobe`, …; the crossing positions themselves (main lobe and outermost, along the axis, $x$ and $y$) from `Results.level_extents(level_dB)` |
| Focal depth $d(L)$ | extent along the axis above the level: at −6 dB, the beam maximum depth $L_{bm}$, the **smallest** distance between two points on the axis where the level is reached (61828; TS 63587, 3.16) | level crossings $z^{\pm}$ of the axial profile either side of the focus; NaN if the region reaches the source plane or the end of the domain | recorded: `focal_length_6dB`, … |
| Focal area, beam area $A(L)$ | area in the focal plane of **all** the points above the level (62127-1; the beam maximum volume $V_{bm}$ of 61828 and TS 63587, 3.18, is its extension in $z$) | grid cells above the level, times $\Delta x\, \Delta y$. The main lobe's area $A^{\mathrm{ml}}(L)$ (the cells connected to the focus) is given too | recorded: `focal_area_6dB`, …, and `focal_area_6dB_main_lobe`, … |
| Focal gain $G$ | focal pressure over source pressure | $\lvert P_1 \rvert$ at the focus over the source amplitude $p_0$ | recorded: `focal_gain` |
| Side lobes, prefocal and post-focal maxima | secondary maxima, for example for safety near skin and bone | the largest maxima outside the main lobe. The lobe extends past the −6 dB level to its first minimum, so ripples on it do not count. Computed along the axis (prefocal, post-focal) and along $x$ and $y$ in the focal plane. Every maximum outside the main lobe, not only the largest each side, is given by `iec.side_lobe_peaks` on the lines of `Results.focal_lines()` | recorded: `iec.side_lobes` (`axial`, `x`, `y`: value, position, level in dB) |

### Exposure indices (IEC 62359)

| Parameter | Definition (paraphrased) | Computation from the planes | Status and where |
| --- | --- | --- | --- |
| Derated quantities ($p_{r,\alpha}$, $I_{ta,\alpha}$, $P_{\alpha}$) | water values reduced by 0.3 dB cm⁻¹ MHz⁻¹ along the axis | multiply by $\delta = 10^{-a f_{awf} (z - z_0)/20}$ for pressures and by $\delta^2$ for intensity and power (`iec.derating`; $a$ = `Metrics.derating`). This is the standard's linear derating of the water field. For derated *nonlinear* propagation, run in a medium with that attenuation instead | recorded: `axial.txt` p_r,α and I_ta,α; `iec.derated.I_spta_derated_axis`, `p_r_derated_max_axis` |
| Mechanical index $\mathrm{MI}$ | $p_{r,\alpha}$ (in MPa) at the depth $z_{MI}$ of the maximum of the derated $\mathrm{pii}$ (beyond the break-point depth $z_{bp}$), divided by $\sqrt{f_{awf}}$ ($f_{awf}$ in MHz) | from the axial $p_r$ and $\mathrm{pii}$, derated, with $f_{awf}$ from the focus | recorded: `iec.derated.MI`, `z_MI`, `p_r_derated`; the value without derating is `focus.mechanical_index_water` |
| Thermal indices $\mathrm{TIS}$, $\mathrm{TIB}$, $\mathrm{TIC}$ (unscanned mode) | $\mathrm{TIS} = P f_{awf} / C_{TIS}$ for an aperture $\le 1$ cm², else $\max_{z \ge z_{bp}} \min(P_{\alpha}, I_{ta,\alpha} \times 1\ \mathrm{cm}^2)\, f_{awf} / C_{TIS}$; $\mathrm{TIB} = \max_{z \ge z_{bp}} \min(\sqrt{P_{\alpha} I_{ta,\alpha}}/C_{TIB1}, P_{\alpha}/C_{TIB2})$; $\mathrm{TIC} = P/(D_{eq} C_{TIC})$, with $D_{eq} = \sqrt{4 A_{aprt}/\pi}$ and $z_{bp} = 1.5 D_{eq}$ | $P$ from the source plane; $I_{ta,\alpha}$ along the axis; $A_{aprt}$ is the −12 dB output beam area of the source plane; $C_{TIS} = 210$ mW MHz, $C_{TIB1} = 50$ mW/cm, $C_{TIB2} = 4.4$ mW, $C_{TIC} = 40$ mW/cm (`iec.thermal_indices`). The formulae and constants are for diagnostic equipment: check them against the edition in force | recorded: `iec.derated.TIS`, `TIB`, `TIC`, `A_aprt`, `D_eq`, `z_bp` |
| Scanned-mode indices | the same for scanning beams | **not possible** with one beam: they need the field of every scan line | — |
| Heating rate $q$ | (not an IEC index; used to plan HITU treatments) $2 \sum_n \alpha_n I_n$ | from the harmonics and the absorption $\alpha_n$ of the medium | recorded: `axial.txt` q, `focus.heating_rate` |

### Physiotherapy (IEC 61689)

| Parameter | Definition (paraphrased) | Computation from the planes | Status and where |
| --- | --- | --- | --- |
| Beam cross-sectional area $A_{BCS}$ | smallest area holding 75 % of the mean-square pressure, on a plane near the face | on the plane kept at `face_distance` (default 0.3 cm): sort the intensity and add up the smallest area holding 75 % (`iec.area_holding`) | recorded: `iec.physiotherapy.A_BCS`, `z` |
| Effective radiating area $A_{ER}$ | $A_{BCS}$ at 0.3 cm times a factor (1.333) | $F_{AC} \times A_{BCS}$ | recorded: `iec.physiotherapy.A_ER` |
| Effective intensity $I_e$ | output power over $A_{ER}$ | $P / A_{ER}$ | recorded: `iec.physiotherapy.effective_intensity` |
| Beam non-uniformity ratio $R_{BN}$ | spatial-peak intensity relative to the mean intensity over $A_{ER}$ | $I_{spta} / (P / A_{ER})$, in the plane-wave intensity form | recorded: `iec.physiotherapy.R_BN` |

### Nonlinear and shock parameters (IEC TS 62556 conditions)

| Parameter | Use | Computation from the planes | Status and where |
| --- | --- | --- | --- |
| Degree of nonlinear distortion | whether the field is nonlinear, and so whether low-drive extrapolation is valid (62556) | $p_c/p_r$ and $\lvert P_2 \rvert / \lvert P_1 \rvert$ at the focus; a second run at another amplitude (or linear) shows the departure from linear scaling | recorded: `iec.nonlinear.p_c_over_p_r`, `P2_over_P1` |
| Steepness $S$, shock formation $z_{sh}$, shock amplitude $A_s$ | where shocks form and how large they are (histotripsy, boiling histotripsy) | $\max_\tau (\mathrm{d}p/\mathrm{d}\tau)$ over $\omega \lvert P_1 \rvert$ (1 for a sinusoid); the rise $A_s^{\mathrm{wf}}$ across the steepest part over $T/K$ (it includes the Gibbs overshoot of the $K$ harmonics, up to ~18 %); from the spectrum, the local exponent $s$ of $\lvert P_n \rvert \sim n^{-s}$ over $n = K/8, \dots, K/2$ ($4/3$ where the Fubini solution forms a shock, 1 for a developed shock) and the jump $A_s = \pi n \lvert P_n \rvert$ (`iec.spectral_shock`); the first $z$ on the axis where $s$ falls to $4/3$ (`Results.shock_formation`); the $z$ where the Godunov scheme takes over | recorded: `iec.nonlinear.steepness`, `shock_amplitude`, `shocked`, `spectral_exponent`, `shock_amplitude_spectral`, `shocked_spectral`, `shock_formation.z`, `shock_formation.distance_to_focus`, `godunov_from_z` |

## Practical notes

- **Resolution.** The −6 dB widths need at least about 4 points across the focal width, and the axial maxima about 4 planes over the focal depth. The peak pressures of shocked waveforms depend on the number of harmonics $K$: $p_{+}$ converges slowly, so check it with `SamplingFrequencyMultiplier` doubled.
- **Spatial averaging.** A hydrophone averages over its sensitive element. Set `Metrics.hydrophone_diameter` to compare focal values with a measurement.
- **Low-drive extrapolation.** 62556 lets HITU focal values be extrapolated from low-drive measurements when the field is close to linear. A linear run (`"NLN": false`) gives that extrapolation, and the nonlinear run gives the actual values. Comparing the two quantifies the error of extrapolating.
