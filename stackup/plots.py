"""
Schematics and charts.  Every function draws onto a matplotlib ``Figure``
(cleared first) so the same code serves the GUI canvases and the PDF report.

  draw_cross_section  – true-scale radial section with tolerance bands
  draw_buildup        – layer-by-layer OD build-up (floating waterfall)
  draw_cutaway        – longitudinal stepped cut-away (side view)
  draw_pareto         – contribution Pareto for any metric
  draw_mc             – Monte-Carlo histogram + capability
  draw_heatshrink     – heat-shrink process window diagram
"""
from __future__ import annotations

import math
import textwrap
from typing import Dict, Optional

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

from .engine import Analysis, METHOD_LABELS
from .model import LAYER_TYPES
from .units import MM_PER_IN, fmt_dia_both, french, unit_name

THEMES = {
    "light": dict(bg="#ffffff", fg="#1b2a4a", muted="#6b7785",
                  grid="#e3e7ec", lumen="#ffffff", edge="#1b2a4a",
                  ok="#1f8a4c", bad="#c0392b", warn="#d68910",
                  wc="#e8a13d", rss="#0f8a8a", mrss="#7a5cc4", mc="#2c6fbb",
                  usl="#c0392b", box="#ffffff", steel="#59636e"),
    "dark": dict(bg="#1e2228", fg="#e6e9ee", muted="#9aa4b1",
                 grid="#39404b", lumen="#14171c", edge="#d5dbe3",
                 ok="#4cc27a", bad="#ff6b5b", warn="#f0b44c",
                 wc="#f0b44c", rss="#3cc4c4", mrss="#a58cf0", mc="#6aa8f0",
                 usl="#ff6b5b", box="#262b33", steel="#9aa6b3"),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mix(c, other, t):
    a, b = np.array(to_rgb(c)), np.array(to_rgb(other))
    return tuple(a * (1 - t) + b * t)


def layer_colors(design) -> Dict[str, tuple]:
    """Distinct colour per layer; repeated types get lighter/darker shades."""
    seen: Dict[str, int] = {}
    out = {}
    for L in design.layers:
        n = seen.get(L.TYPE, 0)
        seen[L.TYPE] = n + 1
        base = L.COLOR
        out[L.uid] = to_rgb(base) if n == 0 else \
            _mix(base, "#ffffff" if n % 2 else "#000000", min(0.32 + 0.1 * n, 0.6))
    return out


def _disp(v, kind, units):
    return v * MM_PER_IN if (kind == "len" and units == "mm") else v


def _ulab(kind, units):
    return {"len": unit_name(units), "angle": "deg", "ratio": "—"}.get(kind, "")


def _style_axes(ax, th, grid=True):
    ax.set_facecolor(th["bg"])
    for s in ax.spines.values():
        s.set_color(th["muted"])
    ax.tick_params(colors=th["fg"], labelsize=8)
    ax.xaxis.label.set_color(th["fg"])
    ax.yaxis.label.set_color(th["fg"])
    ax.title.set_color(th["fg"])
    if grid:
        ax.grid(True, color=th["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)


def _prep(fig, th):
    fig.clear()
    fig.set_facecolor(th["bg"])


def _nice(x):
    """Largest 1-2-5 number ≤ x."""
    if x <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    for m in (5, 2, 1):
        if m * e <= x:
            return m * e
    return e


def _method_color(th, m):
    return th.get(m, th["rss"])


def _message(fig, th, msg):
    _prep(fig, th)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=th["muted"],
            fontsize=10, wrap=True, transform=ax.transAxes)


# ---------------------------------------------------------------------------
# 1. cross-section
# ---------------------------------------------------------------------------
def draw_cross_section(fig, an: Analysis, th: dict, units="in",
                       method: Optional[str] = None, show_bands=True,
                       show_callouts=True):
    _prep(fig, th)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_aspect("equal")
    method = method or an.method
    mcol = _method_color(th, method)
    design = an.design
    cols = layer_colors(design)
    ctx = an.nom_ctx
    rings = [r for r in ctx.rings if r["type"] != "mandrel"]
    e = design.acceptance.ecc.nom
    last_uid = rings[-1]["uid"] if rings else None

    def sx(uid):                       # eccentric core shift
        return 0.0 if uid == last_uid else e

    z = 2
    for r in reversed(rings):
        L = r["layer"]
        fc = cols[r["uid"]]
        if r["type"] == "braid":
            fc = _mix(fc, th["bg"], 0.55)
        ax.add_patch(mpatches.Circle((sx(r["uid"]), 0), float(r["od"]) / 2,
                                     facecolor=fc, edgecolor=th["edge"],
                                     lw=0.6, zorder=z))
        z += 1
    # wires
    for r in rings:
        L = r["layer"]
        if r["type"] not in ("braid", "coil"):
            continue
        r0 = float(r["id"]) / 2
        flat = L.options["shape"] == "flat"
        h = L.dims["thick" if flat else "dia"].nom
        w = L.dims["width"].nom if flat else h
        steel = _mix(cols[r["uid"]], "#000000", 0.25)
        if r["type"] == "coil":
            ax.add_patch(mpatches.Annulus((sx(r["uid"]), 0), float(r["od"]) / 2,
                                          float(r["od"]) / 2 - r0,
                                          facecolor=steel, edgecolor=th["edge"],
                                          lw=0.4, zorder=z))
            continue
        nc = int(L.options["carriers"])
        stack = int(L.options["stack"])
        for i in range(nc):
            a = 2 * math.pi * i / nc
            lev = i % 2 if stack >= 2 else 0
            rr = r0 + h * (lev + 0.5)
            cx, cy = sx(r["uid"]) + rr * math.cos(a), rr * math.sin(a)
            if flat:
                patch = mpatches.Ellipse((cx, cy), w, h,
                                         angle=math.degrees(a) + 90)
            else:
                patch = mpatches.Circle((cx, cy), h / 2)
            patch.set(facecolor=steel, edgecolor=th["edge"], lw=0.3,
                      zorder=z + 1)
            ax.add_patch(patch)
    z += 3
    # lumen
    lum = an.nom("lumen")
    ax.add_patch(mpatches.Circle((e if rings else 0, 0), lum / 2,
                                 facecolor=th["lumen"], edgecolor=th["edge"],
                                 lw=0.8, zorder=z))
    # channels
    ch_col = to_rgb(LAYER_TYPES["channels"].COLOR)
    for cs in ctx.channel_sets:
        for it in cs.items:
            r, a = float(it["r"]), float(it["ang"])
            cx, cy = e + r * math.cos(a), r * math.sin(a)
            if it["kind"] == "lined":
                ax.add_patch(mpatches.Circle((cx, cy), float(it["od"]) / 2,
                                             facecolor=cols[cs.layer.uid],
                                             edgecolor=th["edge"], lw=0.6,
                                             zorder=z + 1))
            ax.add_patch(mpatches.Circle((cx, cy), float(it["id"]) / 2,
                                         facecolor=th["lumen"],
                                         edgecolor=th["edge"], lw=0.6,
                                         zorder=z + 2))
            ax.text(cx, cy, it["label"], ha="center", va="center",
                    fontsize=5.5, color=th["fg"], zorder=z + 3)
    z += 4
    # tolerance bands
    if show_bands:
        for key, cx in [("lumen", e if rings else 0.0)] + \
                [(f"od|{r['uid']}", sx(r["uid"])) for r in rings]:
            lo, hi = an.band(key, method)
            if hi - lo > 0:
                ax.add_patch(mpatches.Annulus((cx, 0), hi / 2, (hi - lo) / 2,
                                              facecolor=mcol, alpha=0.45,
                                              edgecolor="none", zorder=z))
    z += 1
    fin_lo, fin_hi = an.band("final_od", method)
    usl = design.acceptance.od_usl
    ax.add_patch(mpatches.Circle((0, 0), fin_hi / 2, fill=False,
                                 edgecolor=mcol, lw=1.2, ls="--", zorder=z))
    if usl > 0:
        ax.add_patch(mpatches.Circle((0, 0), usl / 2, fill=False,
                                     edgecolor=th["usl"], lw=1.4, ls=":",
                                     zorder=z))

    R = max(fin_hi, usl) / 2
    # callouts
    if show_callouts:
        items = [("Main lumen", lum, th["edge"], lum / 2, e)]
        items += [(r["name"], float(r["od"]), cols[r["uid"]],
                   float(r["od"]) / 2, sx(r["uid"])) for r in rings]
        for cs in ctx.channel_sets:
            seen = set()
            for it in cs.items:
                sig = (round(float(it["id"]), 6), round(float(it["od"]), 6),
                       it["kind"])
                if sig in seen:
                    continue
                seen.add(sig)
                n_same = sum(1 for j in cs.items
                             if (round(float(j["id"]), 6),
                                 round(float(j["od"]), 6), j["kind"]) == sig)
                lab = f"{n_same}× {'lined ch.' if it['kind'] == 'lined' else 'lumen'}" \
                      f" ID"
                items.append((lab, float(it["id"]), ch_col,
                              float(it["r"]) + float(it["od"]) / 2, e))
        items.append((f"{METHOD_LABELS[method]} max OD", fin_hi, mcol,
                      fin_hi / 2, 0))
        if usl > 0:
            items.append(("OD ceiling", usl, th["usl"], usl / 2, 0))
        x_txt = 1.4 * R
        n_c = len(items)
        for i, (name, dia, col, r_pt, cx) in enumerate(items):
            y_txt = R * (1.05 - 2.1 * i / max(n_c - 1, 1))
            ang = math.atan2(y_txt, x_txt)
            ax.annotate(f"{name}  {fmt_dia_both(dia, units)}",
                        xy=(cx + r_pt * math.cos(ang), r_pt * math.sin(ang)),
                        xytext=(x_txt, y_txt), fontsize=7, color=th["fg"],
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=col,
                                        shrinkA=0, shrinkB=0),
                        bbox=dict(boxstyle="round,pad=0.25", fc=th["box"],
                                  ec=col, lw=0.9), zorder=z + 5)
        ax.set_xlim(-1.3 * R, 3.45 * R)
    else:
        ax.set_xlim(-1.3 * R, 1.3 * R)
    ax.set_ylim(-1.3 * R, 1.3 * R)

    # scale bar (display units)
    k = MM_PER_IN if units == "mm" else 1.0
    bar_d = _nice(0.5 * R * k)
    bar = bar_d / k
    x0, y0 = -1.2 * R, -1.2 * R
    ax.plot([x0, x0 + bar], [y0, y0], color=th["fg"], lw=2.2,
            solid_capstyle="butt")
    for xx in (x0, x0 + bar):
        ax.plot([xx, xx], [y0 - 0.025 * R, y0 + 0.025 * R], color=th["fg"],
                lw=1)
    ax.text(x0 + bar / 2, y0 + 0.05 * R,
            f"{bar_d:g} {unit_name(units)}", ha="center", va="bottom",
            fontsize=7, color=th["fg"])
    fr = french(an.nom("final_od"))
    ax.set_title(f"{design.name}\nNominal cross-section (true scale) — "
                 f"{fr:.2f} Fr nominal; shaded = {METHOD_LABELS[method]} "
                 f"OD bands", fontsize=9, color=th["fg"])


# ---------------------------------------------------------------------------
# 2. radial build-up
# ---------------------------------------------------------------------------
def draw_buildup(fig, an: Analysis, th: dict, units="in",
                 method: Optional[str] = None):
    _prep(fig, th)
    ax = fig.add_subplot(111)
    _style_axes(ax, th)
    method = method or an.method
    cols = layer_colors(an.design)
    rows = an.layer_rows()
    k = MM_PER_IN if units == "mm" else 1.0
    prev = 0.0
    xs, names = [], []
    have_mc = "mc" in an.bands
    for i, row in enumerate(rows):
        nom = an.nom(row["key"])
        col = cols.get(row["uid"], (0.7, 0.7, 0.7))
        hatch = "///" if row["type"] == "mandrel" else None
        ax.bar(i, (nom - prev) * k, bottom=prev * k, width=0.62, color=col,
               edgecolor=th["edge"], lw=0.6, hatch=hatch, zorder=2)
        d = (nom - prev) * k
        ax.text(i, nom * k, f"\n+{d:.4f}" if i else f"\n{d:.4f}",
                ha="center", va="top", fontsize=6.5, color=th["fg"], zorder=4)
        for off, m, lw in ((-0.2, "wc", 6), (0.0, method if method != "wc"
                                              else "rss", 3.5),
                           (0.2, "mc", 2)):
            if m not in an.bands or (off == 0.0 and m == "wc"):
                continue
            lo, hi = an.band(row["key"], m)
            ax.plot([i + off, i + off], [lo * k, hi * k],
                    color=_method_color(th, m), lw=lw,
                    solid_capstyle="butt", alpha=0.85, zorder=3)
        prev = nom
        xs.append(i)
        names.append("\n".join(textwrap.wrap(row["name"], 14)))
    acc = an.design.acceptance
    if acc.od_usl > 0:
        ax.axhline(acc.od_usl * k, color=th["usl"], ls="--", lw=1.2,
                   label=f"OD ceiling {acc.od_usl * k:.4f}")
    if acc.od_lsl > 0:
        ax.axhline(acc.od_lsl * k, color=th["usl"], ls=":", lw=1.0,
                   label="OD LSL")
    # legend proxies
    from matplotlib.lines import Line2D
    prox = [Line2D([], [], color=th["wc"], lw=6, label="Worst-Case")]
    sm = method if method != "wc" else "rss"
    prox.append(Line2D([], [], color=_method_color(th, sm), lw=3.5,
                       label=METHOD_LABELS[sm]))
    if have_mc:
        prox.append(Line2D([], [], color=th["mc"], lw=2,
                           label="Monte Carlo (P0.135–P99.865)"))
    h, l = ax.get_legend_handles_labels()
    leg = ax.legend(handles=prox + h, fontsize=7, loc="upper left",
                    facecolor=th["box"], edgecolor=th["muted"])
    for t in leg.get_texts():
        t.set_color(th["fg"])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7, color=th["fg"])
    ax.set_ylabel(f"Diameter ({unit_name(units)})")
    fin = an.nom("final_od")
    ax.set_title(f"Radial build-up — diametral increment per layer  "
                 f"(final Ø{fin * k:.4f} {unit_name(units)}, "
                 f"{french(fin):.2f} Fr)", fontsize=9)
    ymax = max([an.band(r["key"], "wc")[1] for r in rows] + [acc.od_usl]) * k
    ax.set_ylim(0, ymax * 1.08)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# 3. longitudinal cut-away
# ---------------------------------------------------------------------------
def draw_cutaway(fig, an: Analysis, th: dict, units="in"):
    _prep(fig, th)
    ax = fig.add_subplot(111)
    _style_axes(ax, th, grid=False)
    design = an.design
    cols = layer_colors(design)
    ctx = an.nom_ctx
    k = MM_PER_IN if units == "mm" else 1.0
    ring_by_uid = {r["uid"]: r for r in ctx.rings}
    cs_by_uid = {cs.layer.uid: cs for cs in ctx.channel_sets}
    steps = []
    for L in design.layers:
        if L.TYPE == "mandrel":
            continue
        if L.uid in ring_by_uid:
            steps.append(("ring", ring_by_uid[L.uid]))
        elif L.uid in cs_by_uid:
            steps.append(("ch", cs_by_uid[L.uid]))
    n = max(len(steps), 1)
    Lx = 10.0
    s = Lx * 0.85 / n
    ends = [Lx - j * s for j in range(n)]
    lum = an.nom("lumen") / 2 * k
    Rmax = an.nom("final_od") / 2 * k
    # lumen + centre line
    ax.add_patch(mpatches.Rectangle((0, -lum), Lx + 0.4, 2 * lum,
                                    facecolor=th["lumen"], edgecolor="none",
                                    zorder=1))
    ax.plot([-0.2, Lx + 0.6], [0, 0], color=th["muted"], lw=0.7,
            ls=(0, (8, 3, 2, 3)), zorder=2)
    label_pts = []
    for j, (kind, obj) in enumerate(steps):
        xe = ends[j]
        if kind == "ring":
            r = obj
            ri, ro = float(r["id"]) / 2 * k, float(r["od"]) / 2 * k
            col = cols[r["uid"]]
            hatch = {"braid": "xxxx", "coil": "////"}.get(r["type"])
            for sgn in (1, -1):
                y0 = ri if sgn > 0 else -ro
                ax.add_patch(mpatches.Rectangle(
                    (0, y0), xe, ro - ri, facecolor=col,
                    edgecolor=_mix(col, "#000000", 0.45) if hatch
                    else th["edge"], lw=0.5, hatch=hatch, zorder=3))
            label_pts.append((xe - s * 0.45, ro, r["name"],
                              fmt_dia_both(float(r["od"]), units), col))
        else:
            cs = obj
            host = j + 1 if j + 1 < len(steps) else None
            he = ends[host] if host is not None else 0.0
            for it in cs.items:
                # developed view: channel rotated into the section plane
                y = float(it["r"]) * k * (1 if math.sin(float(it["ang"]))
                                          >= -1e-9 else -1)
                od, idv = float(it["od"]) * k, float(it["id"]) * k
                col = cols[cs.layer.uid]
                if it["kind"] == "lined":
                    ax.add_patch(mpatches.Rectangle(
                        (he, y - od / 2), xe - he, od, facecolor=col,
                        edgecolor=th["edge"], lw=0.5, zorder=6))
                ax.add_patch(mpatches.Rectangle(
                    (he, y - idv / 2), xe - he, idv, facecolor=th["lumen"],
                    edgecolor=th["edge"], lw=0.4, zorder=7))
                # hidden (embedded) portion
                ax.add_patch(mpatches.Rectangle(
                    (0, y - od / 2), he, od, fill=False,
                    edgecolor=th["fg"], lw=0.5, ls=(0, (3, 2)), zorder=8))
            top = max(float(it["r"]) * k + float(it["od"]) * k / 2
                      for it in cs.items)
            label_pts.append((xe - s * 0.45, top, cs.layer.name,
                              f"{len(cs.items)} channels", cols[cs.layer.uid]))
    # labels, staggered over 2-3 rows depending on layer count
    nrow = 2 if len(label_pts) <= 7 else 3
    fs = 6.5 if nrow == 2 else 5.8
    for i, (x, y, name, val, col) in enumerate(label_pts):
        yt = Rmax * (1.35 + 0.42 * (i % nrow))
        ax.annotate(f"{name}\n{val}", xy=(x, y), xytext=(x, yt), fontsize=fs,
                    ha="center", va="bottom", color=th["fg"],
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=col),
                    bbox=dict(boxstyle="round,pad=0.2", fc=th["box"], ec=col,
                              lw=0.7))
    ax.text(Lx + 0.5, 0, f"lumen {fmt_dia_both(an.nom('lumen'), units)}",
            fontsize=6.5, va="center", ha="left", color=th["fg"])
    ax.set_xlim(-0.3, Lx + 3.2)
    ax.set_ylim(-Rmax * 1.25, Rmax * (2.25 + 0.42 * (nrow - 2)))
    ax.set_xticks([])
    ax.set_ylabel(f"Radius ({unit_name(units)}) — true scale")
    ax.set_title("Longitudinal cut-away (radial true scale, axial schematic; "
                 "channels rotated into section, dashed = embedded)", fontsize=9)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# 4. contribution Pareto
# ---------------------------------------------------------------------------
def draw_pareto(fig, an: Analysis, th: dict, key="final_od", mode="rss",
                units="in", top=15):
    rows = an.contributions(key)
    if not rows:
        _message(fig, th, "No toleranced contributors for this metric.")
        return
    _prep(fig, th)
    ax = fig.add_subplot(111)
    _style_axes(ax, th)
    pk = "var_pct" if mode == "rss" else "wc_pct"
    rows = sorted(rows, key=lambda r: r[pk], reverse=True)[:top]
    vals = [r[pk] for r in rows]
    x = np.arange(len(rows))
    col = th["rss"] if mode == "rss" else th["wc"]
    ax.bar(x, vals, color=col, edgecolor=th["edge"], lw=0.5, zorder=2)
    for i, r in enumerate(rows):
        ax.text(i, vals[i], f"{vals[i]:.1f}%\ns={r['sens']:+.2f}",
                ha="center", va="bottom", fontsize=6, color=th["fg"])
    ax.set_xticks(x)
    ax.set_xticklabels(["\n".join(textwrap.wrap(r["label"], 18))
                        for r in rows], rotation=50, ha="right", fontsize=6.5,
                       color=th["fg"])
    ax.set_ylabel("Contribution (%)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax2 = ax.twinx()
    cum = np.cumsum(vals)
    ax2.plot(x, cum, color=th["fg"], marker="o", ms=3, lw=1)
    ax2.axhline(80, color=th["muted"], ls=":", lw=0.8)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative (%)", color=th["fg"])
    ax2.tick_params(colors=th["fg"], labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color(th["muted"])
    info = an.info[key]
    what = "variance share (RSS)" if mode == "rss" else \
        "share of worst-case band (|s|·tol)"
    ax.set_title(f"Contribution Pareto — {info.label}: {what}\n"
                 f"s = sensitivity ∂metric/∂dimension", fontsize=9)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# 5. Monte Carlo
# ---------------------------------------------------------------------------
def draw_mc(fig, an: Analysis, th: dict, key="final_od", units="in"):
    x = an.samples(key)
    if x is None:
        _message(fig, th, "Monte Carlo is disabled — enable it on the "
                          "Analysis tab.")
        return
    _prep(fig, th)
    ax = fig.add_subplot(111)
    _style_axes(ax, th)
    info = an.info[key]
    xd = _disp(x, info.kind, units)
    ax.hist(xd, bins=70, density=True, color=th["mc"], alpha=0.75,
            edgecolor=th["bg"], lw=0.3, zorder=2)
    mu, sd = float(np.mean(xd)), float(np.std(xd, ddof=1))
    if sd > 0:
        g = np.linspace(xd.min(), xd.max(), 300)
        ax.plot(g, np.exp(-0.5 * ((g - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)),
                color=th["fg"], lw=1.1, label="Normal fit", zorder=3)
    p_lo, p_hi = np.percentile(xd, [0.135, 99.865])
    for v, lab in ((p_lo, "P0.135"), (p_hi, "P99.865")):
        ax.axvline(v, color=th["mc"], ls="--", lw=1, zorder=4)
        ax.text(v, ax.get_ylim()[1] * 0.97, lab, rotation=90, va="top",
                ha="right", fontsize=6.5, color=th["fg"])
    chk = next((c for c in an.checks if c.key == key), None)
    lsl = usl = None
    if chk:
        lsl, usl = chk.lsl, chk.usl
        for v, lab in ((lsl, "LSL"), (usl, "USL")):
            if v is not None:
                vd = _disp(v, info.kind, units)
                ax.axvline(vd, color=th["usl"], lw=1.6, zorder=5)
                ax.text(vd, ax.get_ylim()[1] * 0.6, f" {lab}", color=th["usl"],
                        fontsize=8, fontweight="bold")
    cap = an.capability(key, lsl, usl)
    lines = [f"N = {x.size:,}" + (f"  (invalid {an.mc_invalid})"
                                 if an.mc_invalid else ""),
             f"mean = {mu:.5f}", f"σ = {sd:.6f}",
             f"P0.135 = {p_lo:.5f}", f"P99.865 = {p_hi:.5f}"]
    if cap.get("cp") is not None:
        lines.append(f"Cp = {cap['cp']:.2f}")
    if cap.get("cpk") is not None:
        lines.append(f"Cpk = {cap['cpk']:.2f}  ({an.settings.cap_method})")
    if cap.get("yield") is not None and (lsl is not None or usl is not None):
        y = cap["yield"]
        lines.append(f"yield = {100 * y:.4f} %")
        lines.append(f"defects ≈ {1e6 * (1 - y):,.0f} ppm")
    ax.text(0.99, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right",
            va="top", fontsize=7.5, family="monospace", color=th["fg"],
            zorder=10, bbox=dict(boxstyle="round", fc=th["box"],
                                 ec=th["muted"]))
    s = an.settings
    ax.set_title(f"Monte Carlo — {info.label}  [{s.mc_sampling.upper()}, "
                 f"default dist {s.default_dist}, tol = ±{s.k_sigma:g}σ"
                 f"{', truncated' if s.truncate else ''}]", fontsize=9)
    ax.set_xlabel(f"{info.label} ({_ulab(info.kind, units)})")
    ax.set_ylabel("Density")
    fig.tight_layout()


# ---------------------------------------------------------------------------
# 6. heat-shrink process diagram
# ---------------------------------------------------------------------------
def draw_heatshrink(fig, an: Analysis, th: dict, units="in",
                    method: Optional[str] = None):
    hs_layers = an.nom_ctx.hs_layers
    if not hs_layers:
        _message(fig, th, "No heat-shrink layer in the stack.\nAdd a "
                          "'Heat shrink (process aid)' layer to see its "
                          "process window.")
        return
    _prep(fig, th)
    method = method or an.method
    mcol = _method_color(th, method)
    k = MM_PER_IN if units == "mm" else 1.0
    n = len(hs_layers)
    sm = method if method != "wc" else "rss"
    smcol = _method_color(th, sm)
    gs = fig.add_gridspec(2 * n, 4, width_ratios=[1, 1, 0.95, 3.0], wspace=0.15,
                          hspace=0.35, height_ratios=[1.3, 1] * n)
    for row, L in enumerate(hs_layers):
        u = L.uid
        has_reinf = f"hs_reinf|{u}" in an.idx
        bars = [("Expanded ID (as supplied)", f"hs_exp|{u}"),
                ("Loaded stack OD (pre-reflow)", f"hs_loaded|{u}"),
                ("OD under shrink (reflowed)", f"hs_od|{u}"),
                ("Recovered ID (free)", f"hs_rec|{u}")]
        if has_reinf:
            bars.append(("Braid/coil crown OD", f"hs_reinf|{u}"))
        ax = fig.add_subplot(gs[2 * row:2 * row + 2, 3])
        _style_axes(ax, th)
        for i, (lab, key) in enumerate(bars):
            y = len(bars) - 1 - i
            wlo, whi = an.band(key, "wc")
            slo, shi = an.band(key, sm)
            ax.plot([wlo * k, whi * k], [y, y], color=th["wc"], lw=16,
                    alpha=0.45, solid_capstyle="butt")
            ax.plot([slo * k, shi * k], [y, y], color=smcol, lw=7,
                    solid_capstyle="butt")
            ax.plot([an.nom(key) * k] * 2, [y - 0.3, y + 0.3],
                    color=th["fg"], lw=1.5)
            ax.text(whi * k, y + 0.28, f"  {an.nom(key) * k:.4f}",
                    fontsize=6.5, color=th["fg"], va="bottom")
        clr = L.dims["clr"].nom
        lo_hi = an.band(f"hs_loaded|{u}", "wc")[1]
        ax.plot([lo_hi * k, (lo_hi + clr) * k], [len(bars) - 2] * 2,
                color=th["muted"], lw=16, alpha=0.35, solid_capstyle="butt")
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([b[0] for b in reversed(bars)], fontsize=7.5,
                           color=th["fg"])
        ax.set_xlabel(f"Diameter ({unit_name(units)})")
        ax.set_ylim(-0.8, len(bars) - 0.2)
        ax.set_title(f"{L.name} — process window\nwide = Worst-Case, "
                     f"narrow = {METHOD_LABELS[sm]}, tick = nominal,\n"
                     f"grey = required loading clearance", fontsize=8.5)

        # verdicts (judged with the acceptance method)
        tx = fig.add_subplot(gs[2 * row + 1, 0:3])
        tx.set_axis_off()
        lines = [f"Checks judged on {METHOD_LABELS[method]}:"]
        oks = [None]
        for c in an.checks:
            if c.key.endswith(f"|{u}") and c.key.startswith("hs_"):
                v = "PASS" if c.passed else ("WARN" if c.severity == "warn"
                                             else "FAIL")
                mg = c.margin * (k if c.kind == "len" else 1.0)
                lines.append(f"[{v}] {c.label.split(': ', 1)[-1]}\n"
                             f"       margin {mg:+.4f}"
                             + (f"   MC yield {100 * c.mc_yield:.3f}%"
                                if c.mc_yield is not None else ""))
                oks.append(c.passed)
        y = 0.98
        for t, ok in zip(lines, oks):
            tx.text(0.0, y, t, transform=tx.transAxes, fontsize=7,
                    va="top", ha="left", family="monospace",
                    color=th["fg"] if ok is None else
                    (th["ok"] if ok else th["bad"]))
            y -= 0.13 if ok is None else 0.24

        # before / after insets
        exp_id = L.dims["exp_id"].nom
        rec_id = L.dims["rec_id"].nom
        wall = L.dims["rec_wall"].nom
        loaded = an.nom(f"hs_loaded|{u}")
        od = an.nom(f"hs_od|{u}")
        hs_col = _mix(L.COLOR, th["bg"], 0.2)
        jac = (0.16, 0.62, 0.56)
        for c, (title, inner_fill, ring_id, free_id) in enumerate((
                ("Before recovery", loaded, exp_id, None),
                ("After recovery", od, od, rec_id))):
            a2 = fig.add_subplot(gs[2 * row, c])
            a2.set_aspect("equal")
            a2.set_axis_off()
            # shrink wall drawn with area conserved from the recovered state
            outer = math.sqrt(ring_id ** 2 + (rec_id + 2 * wall) ** 2
                              - rec_id ** 2)
            a2.add_patch(mpatches.Annulus((0, 0), outer / 2,
                                          (outer - ring_id) / 2,
                                          facecolor=hs_col,
                                          edgecolor=th["edge"], lw=0.6))
            a2.add_patch(mpatches.Circle((0, 0), inner_fill / 2,
                                         facecolor=jac, edgecolor=th["edge"],
                                         lw=0.6))
            if free_id:
                a2.add_patch(mpatches.Circle((0, 0), free_id / 2, fill=False,
                                             edgecolor=th["bad"], ls="--",
                                             lw=1))
                note = "red dashed = free\nrecovered ID"
            else:
                note = (f"radial gap\n{((exp_id - loaded) / 2 * k):.4f} "
                        f"{unit_name(units)}")
            a2.text(0, -outer / 2 * 1.12, note, ha="center", va="top",
                    fontsize=6.5, color=th["fg"])
            lim = max(exp_id, outer) / 2 * 1.15
            a2.set_xlim(-lim, lim)
            a2.set_ylim(-lim * 1.6, lim)
            a2.set_title(title, fontsize=8, color=th["fg"])
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.1)


PLOTS = {
    "Cross-section": "cross",
    "Build-up": "buildup",
    "Cut-away": "cutaway",
    "Pareto": "pareto",
    "Monte Carlo": "mc",
    "Heat shrink": "hs",
}
