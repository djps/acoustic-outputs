% Small real outputs of FOCUS and mSOUND, for tests/test_real_outputs.py.
%
%   FOCUS_DIR=~/focus MSOUND_DIR="$HOME/msound/mSOUND v01" octave-cli make_fixtures.m
%
% Run from tests/data with GNU Octave 10 (FOCUS 0.981's Octave build needs Octave 10), with the
% Octave environment activated (e.g. `micromamba run -n octave octave-cli ...`).
addpath(genpath(fullfile(getenv('FOCUS_DIR'), 'Core')));
addpath(getenv('MSOUND_DIR'));
addpath(fullfile(pwd, 'octave_stubs'));
f0 = 1e6;

%% FOCUS: a 3 mm piston at 1 MHz in water (1000 kg/m^3, 1500 m/s), on the xz plane y = 0
medium = set_medium('lossless');
lambda = medium.soundspeed / f0;
xdcr = get_circ(3e-3, [0 0 0], [0 0 0]);
% CW (fnm_cw), planes lambda/6 apart: the phase convention can be read from the field
coordinate_grid = set_coordinate_grid([lambda/6 0 lambda/6], -3e-3, 3e-3, 0, 0, 2e-3, 30e-3);
pressure = fnm_cw(xdcr, coordinate_grid, medium, 200, f0);
save('-v7', 'focus_cw.mat', 'pressure', 'coordinate_grid', 'medium', 'f0');
% transient (fnm_tsd): a 25-cycle tone burst, steady on every plane from 21 to 24 us
coordinate_grid = set_coordinate_grid([1e-3 0 2.5e-3], -3e-3, 3e-3, 0, 0, 5e-3, 30e-3);
deltat = 1 / (12 * f0); tmin = 0;
time_samples = set_time_samples(deltat, tmin, 24e-6);
excitation = set_excitation_function(1, 1, f0, 25 / f0, 0);
pressure = single(fnm_tsd(xdcr, coordinate_grid, medium, time_samples, 200, excitation));
save('-v7', 'focus_tsd.mat', 'pressure', 'coordinate_grid', 'medium', 'f0', 'deltat', 'tmin');

%% mSOUND 2D: a focused 2D aperture (radius 6 mm, focus 20 mm), the beam along mgrid.y
fc = f0; omega_c = 2 * pi * fc; p0 = 0.5e6;
medium = struct('c0', 1500, 'rho0', 1000, 'c', 1500, 'rho', 1000, 'beta', 3.6, 'ca', 0, 'cb', 2.0);
dx = lambda / 4; dy = lambda / 4;
mgrid = set_grid(0, 0, dx, 24e-3, dy, 15e-3);
delay = sqrt(mgrid.x.^2 + 20e-3^2) / medium.c0; delay = delay - min(delay);
source_p = p0 * exp(1i * omega_c * delay); source_p(abs(mgrid.x) > 6e-3) = 0;
P_fundamental = Forward2D_fund(mgrid, medium, source_p, omega_c, 0);
P_second = Forward2D_sec(mgrid, medium, P_fundamental, omega_c);
P_fundamental = single(P_fundamental); P_second = single(P_second);
% struct(mgrid) holds properties that Octave cannot save: keep the fields that are needed
mgrid_s = struct('dx', mgrid.dx, 'dy', mgrid.dy, 'x', mgrid.x, 'num_x', mgrid.num_x, 'num_y', mgrid.num_y, 'dim', mgrid.dim);
save('-v7', 'msound_fsmdm_2d.mat', 'P_fundamental', 'P_second', 'mgrid_s', 'medium', 'fc');
% the same, linear, in the time domain: a CW filling a periodic window of 4 periods
medium.beta = 0;
dt = 1 / (16 * fc);
mgrid = set_grid(dt, 4 / fc - dt, dx, 24e-3, dy, 15e-3);
source_p = p0 * sin(2 * pi * fc * (mgrid.t.' + delay));
source_p(:, abs(mgrid.x) > 6e-3) = 0;
sensor_mask = zeros(mgrid.num_x, mgrid.num_y + 1);
sensor_mask(26:38, 22:end) = 1;
p_time = single(Forward2D(mgrid, medium, source_p, sensor_mask, 0));
sensor_mask = uint8(sensor_mask);
mgrid_s = struct('dx', mgrid.dx, 'dy', mgrid.dy, 'dt', mgrid.dt, 'x', mgrid.x, 'num_x', mgrid.num_x, 'num_y', mgrid.num_y, 'dim', mgrid.dim);
save('-v7', 'msound_tmdm_2d.mat', 'p_time', 'sensor_mask', 'mgrid_s', 'medium', 'fc');
