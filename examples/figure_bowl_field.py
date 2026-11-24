"""An illustrative figure of the field of benchmark C.1 (IEC TS 63587): a focused bowl in water.

Three panels:

* |p| in dB on the plane through the axis, by the Rayleigh integral over the bowl, with the bowl,
  the geometric focus, the beam maximum, and the −3, −6, −12 and −20 dB contours; an inset
  shows the focal region at a finer resolution, with the prefocal and postfocal maxima and the
  side lobes;
* |p| on the axis: the analytic solution of C.1 and the Rayleigh integral, with the beam
  maximum and the prefocal and postfocal maxima (the largest secondary maxima before and after
  the main lobe);
* |p| across the beam in the plane of the beam maximum, in dB, with the focal widths at −3,
  −6, −12 and −20 dB and the side lobes.

Every marked value comes from this package's outputs, not from this script: the bowl's field is
recorded into a ``Results`` (``acoustic_outputs.benchmarks.simulate``), and the beam maximum, the
focal lengths and widths (``Results.metrics``, ``Results.level_extents``), the prefocal and postfocal
maxima and the side lobes (``iec.secondary_maxima``, ``iec.side_lobe_peaks``) are read from it (see
``beam_features``). The analytic axis and a fine Rayleigh profile across the beam are references. The default
transducer is that of C.2 (aperture 100 mm, radius of curvature 90 mm, 1 MHz, water), at its
linear level.

It writes three files: ``bowl_field.png`` (white background), ``bowl_field_transparent.png``
(transparent background) and ``bowl_field.jpg`` (white background: JPEG has no transparency)::

    python examples/figure_bowl_field.py                       # into docs/figures/
    python examples/figure_bowl_field.py -o out --spacing 0.25e-3 --text-color white

Needs matplotlib (``pip install -e ".[plot]"``). The fields take about a minute and a half at the
default 0.4 mm spacing.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from acoustic_outputs import iec
from acoustic_outputs.benchmarks import FocusedBowl, simulate

log = logging.getLogger("figure_bowl_field")

LEVELS = (-3.0, -6.0, -12.0, -20.0)
LEVEL_COLORS = {-3.0: "#b2182b", -6.0: "#e66101", -12.0: "#7b3294", -20.0: "#1b7837"}
MINUS = "−"  # noqa: RUF001 - a typographic minus for the figure


def longitudinal_field(bowl: FocusedBowl, spacing: float, x_max: float, z_lo: float, z_hi: float, ppw: float):
    """|p| on the plane through the axis: x (nx,), z (nz,), |p| (nx, nz), NaN behind and at the bowl."""
    r = spacing * np.arange(int(np.ceil(x_max / spacing)) + 1)
    z = z_lo + spacing * np.arange(int(np.ceil((z_hi - z_lo) / spacing)) + 1)
    t = time.perf_counter()
    p = np.abs(bowl.rayleigh(r, z, points_per_wavelength=ppw))
    log.info("Rayleigh integral at %d x %d points: %.1f s", r.size, z.size, time.perf_counter() - t)
    # behind the bowl's surface, and within half a wavelength of it (where the quadrature of the
    # nearly singular integrand is not accurate), the field is not shown
    R, Z = np.meshgrid(r, z, indexing="ij")
    dist_centre = np.hypot(R, Z - bowl.D)  # distance from the centre of curvature
    on_cap = R <= bowl.a + bowl.wavelength / 2
    behind = on_cap & (dist_centre >= bowl.D - bowl.wavelength / 2) & (Z <= bowl.h + bowl.wavelength / 2)
    # beside the bowl, behind its aperture plane and outside its rim, a focused bowl sends essentially
    # nothing: that is not shown either
    beside = (R > bowl.a) & (Z < bowl.h)
    p[behind | beside] = np.nan
    x = np.concatenate([-r[:0:-1], r])
    return x, z, np.concatenate([p[:0:-1], p])


def region_extent(field, level_db: float) -> tuple[float, float, float]:
    """The z range and the largest |x| of all the points of the field at or above ``level_db``."""
    x, z, p = field
    above = np.nan_to_num(p) >= 10 ** (level_db / 20) * np.nanmax(p)
    ix, iz = np.nonzero(above)
    return float(z[iz.min()]), float(z[iz.max()]), float(np.abs(x[ix]).max())


def beam_features(bowl: FocusedBowl, z_max: float, ppw: float) -> dict:
    """Everything the axial and lateral panels mark, from this package's own outputs.

    The bowl's field is simulated and recorded as any solver's output would be
    (:func:`acoustic_outputs.benchmarks.simulate`, the Rayleigh integral on an (r, z) grid an eighth
    of a wavelength apart, revolved and recorded into a :class:`~acoustic_outputs.results.Results`).
    From it come the beam maximum (``Results.metrics()["focus"]``), the focal lengths and widths and
    their ends (``metrics()``, ``Results.level_extents``), the prefocal and postfocal maxima and the
    largest side lobes (``metrics()["iec"]["side_lobes"]``, :func:`acoustic_outputs.iec.secondary_maxima`)
    and every side lobe (:func:`acoustic_outputs.iec.side_lobe_peaks`). The analytic axis (C.1) and a
    finely sampled Rayleigh profile across the beam are drawn as references only.
    """
    lam = bowl.wavelength
    r_max = max(8 * 1.41 * lam * bowl.f_number, 10 * lam)  # about eight -6 dB focal widths
    res = simulate(bowl, spacing=lam / 8, r_max=r_max, z_range=(bowl.h + lam, z_max), method="rayleigh",
                   points_per_wavelength=ppw)  # fmt: skip
    m = res.metrics()
    prof, line_x, _ = res.focal_lines()
    i, ix = res.focal_index, res.axis_ij[0]
    x0 = res.grid.x[ix]
    x = res.grid.x - x0
    ext = {lv: res.level_extents(lv) for lv in LEVELS}

    def tag(lv: float) -> str:
        return f"{abs(lv):g}dB"

    z_bm = float(m["focus"]["z"])
    # the references: the analytic axis, and the Rayleigh integral across the beam at the beam maximum
    za = np.linspace(bowl.h + lam, z_max, 6001)
    dr = lam / 40
    rr = dr * np.arange(int(np.ceil(x.max() / dr)) + 1)
    pr = np.abs(bowl.rayleigh(rr, [z_bm], points_per_wavelength=ppw)[:, 0])
    return {
        "results": res,
        "z_bm": z_bm,
        "p_bm": float(prof[i]),
        "axial_ref": (za, np.abs(bowl.axial_pressure(za))),
        "axial_sim": (res.z, prof),
        "axial_secondary": m["iec"]["side_lobes"]["axial"],
        # focal lengths: the smallest distance between crossings (IEC 61828 beam maximum depth at -6 dB)
        "focal_lengths": {
            lv: (m[f"focal_length_{tag(lv)}"], *ext[lv]["axis"]["main_lobe"], ext[lv]["level"]) for lv in LEVELS
        },
        # and the outermost crossings, the secondary maxima above the level included
        "axial_extents": {lv: ext[lv]["axis"]["all_lobes"] for lv in (-12.0, -20.0)},
        "lateral_ref": (np.concatenate([-rr[:0:-1], rr]), np.concatenate([pr[:0:-1], pr])),
        "lateral_sim": (x, line_x),
        # focal widths: the greatest distance (IEC 62127-1, 61828), and the main lobe's
        "widths": {lv: m[f"focal_width_x_{tag(lv)}"] for lv in LEVELS},
        "width_ends": {lv: tuple(e - x0 for e in ext[lv]["x"]["all_lobes"]) for lv in LEVELS},
        "main_lobe_widths": {lv: m[f"focal_width_x_{tag(lv)}_main_lobe"] for lv in LEVELS},
        "main_lobe_ends": {lv: tuple(e - x0 for e in ext[lv]["x"]["main_lobe"]) for lv in LEVELS},
        "side_lobes": {k: (v - x0 if k.endswith("position") else v) for k, v in m["iec"]["side_lobes"]["x"].items()},
        "all_lobes": [
            (pk["position"] - x0, pk["level_dB"])
            for pk in iec.side_lobe_peaks(res.grid.x, line_x, ix)
            if pk["level_dB"] > -40
        ],
    }


def figure(bowl, field, inset, feat, text_color):
    import matplotlib.pyplot as plt
    from matplotlib import colors

    x, z, p = field
    pmax = feat["p_bm"]
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(alpha=0.0)  # behind the bowl: the figure's background shows through
    norm = colors.Normalize(-30, 0)

    def db(v):
        return 20 * np.log10(np.maximum(v / pmax, 1e-6))

    plt.rcParams.update({
        "font.size": 9, "text.color": text_color, "axes.labelcolor": text_color, "axes.edgecolor": text_color,
        "xtick.color": text_color, "ytick.color": text_color, "axes.titlecolor": text_color,
    })  # fmt: skip
    fig = plt.figure(figsize=(7.4, 11.3), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[2.75, 1.0, 1.15], width_ratios=[1.0, 0.035])
    top = fig.add_subplot(grid[0, 0])
    axial = fig.add_subplot(grid[1, 0], sharex=top)
    lateral = fig.add_subplot(grid[2, 0])
    cax = fig.add_subplot(grid[0, 1])
    z_bm = feat["z_bm"]
    sec = feat["axial_secondary"]
    side = feat["side_lobes"]

    # -- the longitudinal plane ----------------------------------------------------------------
    mesh = top.pcolormesh(z * 1e3, x * 1e3, db(p), cmap=cmap, norm=norm, shading="auto", rasterized=True)
    for lv in LEVELS:
        top.contour(z * 1e3, x * 1e3, db(p), levels=[lv], colors=LEVEL_COLORS[lv], linewidths=0.8)
    theta = np.linspace(-bowl.phi, bowl.phi, 200)
    top.plot(bowl.D * (1 - np.cos(theta)) * 1e3, bowl.D * np.sin(theta) * 1e3, color="#6b7280", lw=3,
             solid_capstyle="round", label="bowl")  # fmt: skip
    top.plot([0, z[-1] * 1e3], [0, 0], color="#6b7280", lw=0.5, ls=(0, (4, 4)))
    top.plot(bowl.D * 1e3, 0, marker="+", color="black", ms=9, mew=1.4, ls="none", label="geometric focus")
    top.plot(z_bm * 1e3, 0, marker="x", color="#b2182b", ms=6, mew=1.4, ls="none", label="beam maximum")
    plane_style = {"color": "#b2182b", "lw": 0.9, "ls": ":"}
    top.plot([], [], **plane_style, label="plane of the beam maximum")  # drawn once the layout is fixed
    for lv in LEVELS:
        top.plot([], [], color=LEVEL_COLORS[lv], lw=1.2, label=f"{MINUS}{abs(lv):g} dB contour")
    # x from -52 to 52 mm (the bowl's rim, 2 mm in), at equal scales: the panel's height follows from its width
    top.set_ylim(-52.0, 52.0)
    top.set_aspect("equal", adjustable="box")
    top.tick_params(labelbottom=False)
    top.set_ylabel("x [mm]")
    title = fig.suptitle(
        f"Focused bowl, IEC TS 63587 benchmark C.1: aperture {bowl.aperture * 1e3:g} mm, "
        f"radius of curvature {bowl.D * 1e3:g} mm, {bowl.frequency / 1e6:g} MHz",
        fontsize=9,
    )
    top_legend = top.legend(loc="upper right", fontsize=7.5, frameon=False, borderaxespad=1.2, labelcolor="#202020")
    cb = fig.colorbar(mesh, cax=cax)
    # left-aligned over the colorbar and kept out of the layout, so that it does not push the colorbar
    # away from the field map
    cb_title = cb.ax.set_title(r"$|p|\,/\,\max|p|$" + "\n[dB]", fontsize=8, loc="left", pad=6)
    cb_title.set_in_layout(False)

    # inset: the focal region at a finer resolution
    xi, zi, pi = inset
    ins = top.inset_axes([0.37, 0.045, 0.6, 0.27])  # below the beam, clear of the bowl, down to the edge
    ins.set_zorder(5)
    ins.pcolormesh(zi * 1e3, xi * 1e3, db(pi), cmap=cmap, norm=norm, shading="auto", rasterized=True)
    for lv in LEVELS:
        ins.contour(zi * 1e3, xi * 1e3, db(pi), levels=[lv], colors=LEVEL_COLORS[lv], linewidths=0.9)
    ins.plot(bowl.D * 1e3, 0, marker="+", color="black", ms=8, mew=1.2)
    ins.plot(z_bm * 1e3, 0, marker="x", color="#b2182b", ms=5, mew=1.2)
    ins.axvline(z_bm * 1e3, color="#b2182b", lw=0.8, ls=":")
    ins.plot([zi[0] * 1e3, zi[-1] * 1e3], [0, 0], color="#6b7280", lw=0.5, ls=(0, (4, 4)), zorder=1)  # the beam axis
    for key, label in (("before", "prefocal"), ("after", "postfocal")):
        zp = sec[f"{key}_position"]
        if np.isfinite(zp):
            ins.plot(zp * 1e3, 0, marker="o", mfc="none", color="#e66101", ms=6, mew=1.3, ls="none")
            ins.annotate(label, (zp * 1e3, 0), xytext=(0, 8), textcoords="offset points", ha="center",
                         fontsize=6.5, color="#202020")  # fmt: skip
    for key in ("before", "after"):
        xs = side[f"{key}_position"]
        if np.isfinite(xs):
            ins.plot(z_bm * 1e3, xs * 1e3, marker="o", mfc="none", color="#7b3294", ms=5, mew=1.0)
    ins.annotate("side lobes", (z_bm * 1e3, side["after_position"] * 1e3), xytext=(8, 2), textcoords="offset points",
                 fontsize=6.5, color="#202020")  # fmt: skip
    ins.set_aspect("equal")
    ins.tick_params(labelsize=6.5, colors="#202020")
    ins.set_facecolor("white")
    # the zoomed box, and lines from its upper corners to the inset's upper corners
    from matplotlib.patches import ConnectionPatch

    zb0, zb1, xb = zi[0] * 1e3, zi[-1] * 1e3, xi[-1] * 1e3
    top.indicate_inset([zb0, -xb, zb1 - zb0, 2 * xb], edgecolor="#6b7280")
    for corner_inset, corner_box in (((0, 1), (zb0, xb)), ((1, 1), (zb1, xb))):
        fig.add_artist(ConnectionPatch(xyA=corner_inset, coordsA=ins.transAxes, xyB=corner_box,
                                       coordsB=top.transData, color="#6b7280", lw=0.8))  # fmt: skip

    # -- the axis ------------------------------------------------------------------------------
    za, pa = feat["axial_ref"]
    axial.plot(za * 1e3, pa / 1e6, color="#08519c", lw=1.6, label="analytic (C.1)")
    zs, ps = feat["axial_sim"]
    step = max(1, zs.size // 110)
    axial.plot(zs[::step] * 1e3, ps[::step] / 1e6, "o", ms=2.3, color="#6baed6",
               label="Rayleigh field, recorded")  # fmt: skip
    axial.axvline(bowl.D * 1e3, color="black", lw=0.9, ls="--", label=f"geometric focus D = {bowl.D * 1e3:g} mm")
    axial.plot(z_bm * 1e3, pmax / 1e6, marker="x", color="#b2182b", ms=7, mew=1.5, ls="none",
               label=f"beam maximum {z_bm * 1e3:.2f} mm")  # fmt: skip
    for key, label, dx, ha in (("before", "prefocal max", -8, "right"), ("after", "postfocal max", 0, "center")):
        zp, vp, lp = sec[f"{key}_position"], sec[f"{key}_value"], sec[f"{key}_level_dB"]
        if np.isfinite(zp):
            axial.plot(zp * 1e3, vp / 1e6, marker="o", mfc="none", color="#e66101", ms=7, mew=1.5, ls="none")
            axial.annotate(f"{label}\n{zp * 1e3:.1f} mm\n{lp:.1f} dB".replace("-", MINUS), (zp * 1e3, vp / 1e6),
                           xytext=(dx, 10), textcoords="offset points", ha=ha, fontsize=7)  # fmt: skip
    sources = {-6.0: " (L_bm)"}  # the beam maximum depth of IEC 61828, TS 62556 and TS 63587
    for lv, (length, z_lo, z_hi, level) in feat["focal_lengths"].items():
        axial.plot([z_lo * 1e3, z_hi * 1e3], [level / 1e6] * 2, color=LEVEL_COLORS[lv], lw=0.5,
                   label=f"{MINUS}{abs(lv):g} dB focal length {length * 1e3:.2f} mm{sources.get(lv, '')}")  # fmt: skip
    for lv, (e_lo, e_hi) in feat["axial_extents"].items():
        level = 10 ** (lv / 20) * pmax
        label = f"{MINUS}{abs(lv):g} dB incl. secondary maxima {(e_hi - e_lo) * 1e3:.2f} mm"
        axial.plot([e_lo * 1e3, e_hi * 1e3], [level / 1e6] * 2, color=LEVEL_COLORS[lv], lw=0.5, ls=(0, (4, 2)),
                   zorder=1, label=label)  # fmt: skip
    # from 2 mm before the apex, so that the bowl stands clear of the left edge of the field map; the
    # ticks stay those of the field (the two panels share their z axis)
    axial.set_xlim(z[0] * 1e3 - 2.0, z[-1] * 1e3)
    axial.set_xticks(np.arange(0.0, z[-1] * 1e3 + 1e-9, 20.0))
    axial.set_ylim(0, 1.12 * pmax / 1e6)
    axial.set_xlabel("z from the bowl's apex [mm]")
    axial.set_ylabel(r"$|p|$ on axis [MPa]")
    leg = axial.legend(loc="upper left", fontsize=6.5, frameon=False, ncols=2, columnspacing=1.0, handlelength=1.6)
    leg.set_in_layout(False)  # it lies within the panel; it must not move the colorbar
    axial.grid(alpha=0.25)

    # -- across the beam at the beam maximum ------------------------------------------------------------
    xl, pl = feat["lateral_ref"]
    lateral.plot(xl * 1e3, db(pl), color="#08519c", lw=1.4, label="Rayleigh integral")
    xs_, ps_ = feat["lateral_sim"]
    lateral.plot(xs_ * 1e3, db(ps_), "o", ms=2.0, color="#6baed6", label="Rayleigh field, recorded")
    half = 0.5 * np.array([feat["main_lobe_widths"][lv] for lv in LEVELS])
    for lv, w in feat["widths"].items():
        c = LEVEL_COLORS[lv]
        lo, hi = feat["width_ends"][lv]
        main = feat["main_lobe_widths"][lv]
        if abs(main - w) > 1e-3 * w:
            # as on the axis: solid for the main lobe, dashed for the width with the side lobes (the
            # standards' focal width, the greatest distance between the level crossings)
            lateral.plot([lo * 1e3, hi * 1e3], [lv, lv], color=c, lw=0.5, ls=(0, (4, 2)), zorder=1,
                         label=f"{MINUS}{abs(lv):g} dB with side lobes {w * 1e3:.2f} mm")  # fmt: skip
            m_lo, m_hi = feat["main_lobe_ends"][lv]
            lateral.plot([m_lo * 1e3, m_hi * 1e3], [lv, lv], color=c, lw=0.5,
                         label=f"{MINUS}{abs(lv):g} dB main lobe {main * 1e3:.2f} mm")  # fmt: skip
        else:
            lateral.plot([lo * 1e3, hi * 1e3], [lv, lv], color=c, lw=0.5,
                         label=f"{MINUS}{abs(lv):g} dB focal width {w * 1e3:.2f} mm")  # fmt: skip
    for xs, lv in feat["all_lobes"]:
        lateral.plot(xs * 1e3, lv, marker="o", mfc="none", color="#7b3294", ms=4, mew=0.9)
    for key, dx, ha in (("before", -10, "right"), ("after", 10, "left")):
        xs, lv = side[f"{key}_position"], side[f"{key}_level_dB"]
        if np.isfinite(xs):
            lateral.annotate(f"first side lobe\n{lv:.1f} dB".replace("-", MINUS), (xs * 1e3, lv), xytext=(dx, -2),
                             textcoords="offset points", ha=ha, va="center", fontsize=7, color="#7b3294")  # fmt: skip
    lateral.plot([], [], marker="o", mfc="none", color="#7b3294", ms=4, mew=0.9, ls="none", label="side lobes")
    # two legends: the curves and the side lobes on the left, the widths on the right
    handles, labels = lateral.get_legend_handles_labels()
    curves = [k for k, lab in enumerate(labels) if lab.startswith("Rayleigh") or lab == "side lobes"]
    widths = [k for k in range(len(labels)) if k not in curves]
    left = lateral.legend([handles[k] for k in curves], [labels[k] for k in curves], loc="upper left", fontsize=7,
                          frameon=False)  # fmt: skip
    lateral.add_artist(left)
    lateral.legend([handles[k] for k in widths], [labels[k] for k in widths], loc="upper right", fontsize=7,
                   frameon=False)  # fmt: skip
    span = max(6 * half.max(), 8 * bowl.wavelength) * 1e3
    lateral.set_xlim(-span, span)
    lateral.set_ylim(-40, 3)
    lateral.set_xlabel(f"x [mm], in the plane of the beam maximum (z = {z_bm * 1e3:.2f} mm)")
    lateral.set_ylabel(r"$|p|\,/\,\max|p|$ [dB]")
    lateral.grid(alpha=0.25)
    for ax in (top, axial, lateral):
        ax.set_facecolor("none")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    # the field map at the full width of the axial plot below it (their z axes are the same), with its
    # height from the equal scales: its slot is a little taller than that, and the map is moved down
    # to sit just above the axial plot
    renderer = fig.canvas.get_renderer()
    ap, tp = axial.get_position(), top.get_position()
    z0, z1 = top.get_xlim()
    x0, x1 = top.get_ylim()
    fw, fh = fig.get_size_inches()
    height = ap.width * fw * (x1 - x0) / (z1 - z0) / fh
    gap = tp.y0 - top.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0  # the tick marks
    top.set_position([ap.x0, ap.y1 + gap + 0.012, ap.width, height])
    top.set_aspect("auto")  # the position now holds the equal scales
    fig.canvas.draw()
    # the colorbar just beside the field map, at the map's height
    tp, cp = top.get_position(), cax.get_position()
    cax.set_position([tp.x1 + 0.015, tp.y0, cp.width, tp.height])
    # the plane of the beam maximum, from the top of the inset (whose z scale differs; it reaches down to
    # the bottom of the map) up to the legend, which has no frame to hide the line behind. The inset
    # keeps its aspect, so its edges are known only once the layout is done.
    tb = top.get_position()
    above = (ins.get_position().y1 - tb.y0) / tb.height
    lb = top_legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    top.axvline(z_bm * 1e3, ymin=above, ymax=(lb.y0 - tb.y0) / tb.height, **plane_style)
    # the title just above the colorbar's label; the space above it is trimmed when saving
    fig.canvas.draw()
    cbt = cb_title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    title.set_verticalalignment("bottom")
    title.set_y(max(cbt.y1, top.get_position().y1) + 0.008)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--name", default="bowl_field", help="file name stem")
    parser.add_argument("--spacing", type=float, default=0.4e-3, help="field grid spacing [m]")
    parser.add_argument("--ppw", type=float, default=4.0, help="quadrature nodes per wavelength")
    parser.add_argument("--aperture", type=float, default=100e-3)
    parser.add_argument("--radius-of-curvature", type=float, default=90e-3)
    parser.add_argument("--frequency", type=float, default=1e6)
    parser.add_argument("--attenuation", type=float, default=0.0, help="alpha [Np/m]")
    parser.add_argument("--text-color", default="#303030", help="text and axes colour")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import matplotlib

    matplotlib.use("Agg")
    bowl = FocusedBowl(aperture=args.aperture, radius_of_curvature=args.radius_of_curvature,
                       frequency=args.frequency, attenuation=args.attenuation)  # fmt: skip
    z_max = 1.3 * bowl.D
    # wide enough in x to fill the field map, whose x range follows from the z range and its shape
    field = longitudinal_field(bowl, args.spacing, 1.3 * bowl.a, 0.0, z_max, args.ppw)
    feat = beam_features(bowl, z_max, args.ppw)
    # the inset: the whole of the -20 dB region of the field and a wavelength round it, at an
    # eighth of a wavelength
    z_lo, z_hi, x_in = region_extent(field, -20.0)
    z_lo, z_hi, x_in = max(z_lo - bowl.wavelength, 0.0), min(z_hi + bowl.wavelength, z_max), x_in + bowl.wavelength
    inset = longitudinal_field(bowl, bowl.wavelength / 8, x_in, z_lo, z_hi, args.ppw)
    fig = figure(bowl, field, inset, feat, args.text_color)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / args.name
    files = [stem.with_suffix(".png"), stem.parent / f"{args.name}_transparent.png", stem.with_suffix(".jpg")]
    trim = {"bbox_inches": "tight", "pad_inches": 0.1}  # the spare space above the title
    fig.savefig(files[0], dpi=args.dpi, facecolor="white", **trim)
    fig.savefig(files[1], dpi=args.dpi, transparent=True, **trim)
    fig.savefig(files[2], dpi=args.dpi, facecolor="white", pil_kwargs={"quality": 92}, **trim)
    w = feat["widths"]
    log.info("focal widths: %s", ", ".join(f"{lv:g} dB {w[lv] * 1e3:.3f} mm" for lv in LEVELS))
    sec = feat["axial_secondary"]
    log.info("prefocal / postfocal maxima: %.2f mm (%.1f dB), %.2f mm (%.1f dB)",
             sec["before_position"] * 1e3, sec["before_level_dB"], sec["after_position"] * 1e3,
             sec["after_level_dB"])  # fmt: skip
    s = feat["side_lobes"]
    log.info("side lobes: x = %.2f / %.2f mm, %.1f / %.1f dB", s["before_position"] * 1e3, s["after_position"] * 1e3,
             s["before_level_dB"], s["after_level_dB"])  # fmt: skip
    for f in files:
        log.info("wrote %s (%.0f kB)", f, f.stat().st_size / 1e3)


if __name__ == "__main__":
    main()
