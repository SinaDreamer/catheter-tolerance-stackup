"""CSV and PDF report export."""
from __future__ import annotations

import csv
import datetime as _dt
import textwrap
from typing import List

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

from . import __version__
from . import plots as P
from .engine import Analysis, METHOD_LABELS
from .units import MM_PER_IN, french, unit_name


def _d(v, kind, units):
    return v * MM_PER_IN if (kind == "len" and units == "mm") else v


def _f(v, kind, units):
    if v is None:
        return ""
    x = _d(v, kind, units)
    if kind == "ratio":
        return f"{x:.3f}"
    if kind == "angle":
        return f"{x:.2f}"
    return f"{x:.4f}" if units == "mm" else f"{x:.5f}"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def metric_rows(an: Analysis, units: str) -> List[List[str]]:
    head = ["Metric", "Unit", "Nominal"]
    methods = [m for m in ("wc", "rss", "mrss", "mc") if m in an.bands]
    for m in methods:
        head += [f"{METHOD_LABELS[m]} min", f"{METHOD_LABELS[m]} max"]
    rows = [head]
    for k in an.keys:
        info = an.info[k]
        r = [info.label, P._ulab(info.kind, units),
             _f(an.nom(k), info.kind, units)]
        for m in methods:
            lo, hi = an.band(k, m)
            r += [_f(lo, info.kind, units), _f(hi, info.kind, units)]
        rows.append(r)
    return rows


def check_rows(an: Analysis, units: str) -> List[List[str]]:
    rows = [["Status", "Check", "LSL", "USL", "Nominal",
             f"Bound min ({METHOD_LABELS[an.method]})",
             f"Bound max ({METHOD_LABELS[an.method]})", "Margin",
             "MC yield %", "Cpk", "Cpk source"]]
    for c in an.checks:
        st = "PASS" if c.passed else ("WARN" if c.severity == "warn" else "FAIL")
        rows.append([st, c.label, _f(c.lsl, c.kind, units),
                     _f(c.usl, c.kind, units), _f(c.nominal, c.kind, units),
                     _f(c.lo, c.kind, units), _f(c.hi, c.kind, units),
                     _f(c.margin, c.kind, units),
                     f"{100 * c.mc_yield:.4f}" if c.mc_yield is not None else "",
                     f"{c.cpk:.2f}" if c.cpk is not None else "", c.cpk_src])
    return rows


def input_rows(an: Analysis, units: str) -> List[List[str]]:
    rows = [["Layer", "Dimension", "Nominal", "+Tol", "−Tol", "Distribution"]]
    for L in an.design.layers:
        for p in L.visible_params():
            d = L.dims[p.key]
            rows.append([L.name, p.label, _f(d.nom, p.kind, units),
                         _f(d.tp, p.kind, units) if p.toleranced else "—",
                         _f(d.tm, p.kind, units) if p.toleranced else "—",
                         d.dist if p.toleranced else ""])
        for k, v in L.options.items():
            rows.append([L.name, f"[option] {k}", str(v), "", "", ""])
        for c in getattr(L, "channels", []):
            for nm, d, kind in (("ID", c.id, "len"), ("wall", c.wall, "len"),
                                ("angle", c.angle, "angle"),
                                ("radius", c.radius, "len")):
                if nm == "wall" and c.kind == "bare":
                    continue
                if nm == "radius" and L.options.get("radius_mode") != "manual":
                    continue
                rows.append([L.name, f"{c.label} {nm} ({c.kind})",
                             _f(d.nom, kind, units), _f(d.tp, kind, units),
                             _f(d.tm, kind, units), d.dist])
    return rows


def contribution_rows(an: Analysis, key="final_od") -> List[List[str]]:
    rows = [["Contributor", "Sensitivity", "Half-tol", "WC share %",
             "Variance share %"]]
    for r in an.contributions(key):
        rows.append([r["label"], f"{r['sens']:+.4f}", f"{r['half']:.6g}",
                     f"{r['wc_pct']:.2f}", f"{r['var_pct']:.2f}"])
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def export_csv(path: str, an: Analysis, units: str) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([f"Design: {an.design.name}"])
        w.writerow([f"Units: {unit_name(units)}",
                    f"Acceptance method: {METHOD_LABELS[an.method]}",
                    f"Status: {an.status()}"])
        w.writerow([])
        w.writerow(["RESULTS"])
        w.writerows(metric_rows(an, units))
        w.writerow([])
        w.writerow(["CHECKS"])
        w.writerows(check_rows(an, units))
        w.writerow([])
        w.writerow(["CONTRIBUTIONS TO FINAL OD"])
        w.writerows(contribution_rows(an))
        w.writerow([])
        w.writerow(["INPUTS"])
        w.writerows(input_rows(an, units))


def export_mc_csv(path: str, an: Analysis, units: str) -> None:
    if an.mc is None:
        raise ValueError("Monte Carlo is disabled.")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([an.info[k].label for k in an.keys])
        kinds = [an.info[k].kind for k in an.keys]
        data = an.mc.T
        for row in data:
            w.writerow([f"{_d(v, kd, units):.7g}" for v, kd in zip(row, kinds)])


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
_PAGE = (11.0, 8.5)          # US-letter landscape


def _text_pages(pdf, title, blocks, th, lines_per_page=56):
    """blocks: list of (heading, list-of-rows) rendered as monospace tables."""
    lines: List[str] = []
    for heading, rows in blocks:
        if heading:
            lines += ["", heading, "─" * min(len(heading) + 4, 120)]
        if rows and isinstance(rows[0], list):
            widths = [min(max(len(str(r[i])) for r in rows), 46)
                      for i in range(len(rows[0]))]
            for r in rows:
                cells = [str(c)[:46].ljust(widths[i]) for i, c in enumerate(r)]
                lines.append("  ".join(cells).rstrip())
        else:
            for t in rows:
                lines += textwrap.wrap(t, 150) or [""]
    for p in range(0, max(len(lines), 1), lines_per_page):
        fig = Figure(figsize=_PAGE)
        fig.set_facecolor("white")
        fig.text(0.04, 0.95, title, fontsize=13, fontweight="bold",
                 color=th["fg"])
        n_cols = max(len(l) for l in lines[p:p + lines_per_page] or [""])
        fs = 7.2 if n_cols < 150 else max(4.6, 7.2 * 150 / n_cols)
        fig.text(0.04, 0.91, "\n".join(lines[p:p + lines_per_page]),
                 family="monospace", fontsize=fs, va="top", color=th["fg"])
        pdf.savefig(fig)


def export_pdf(path: str, an: Analysis, units: str) -> None:
    th = P.THEMES["light"]
    s, acc, d = an.settings, an.design.acceptance, an.design
    k = MM_PER_IN if units == "mm" else 1.0
    fin = an.nom("final_od")
    flo, fhi = an.band("final_od")
    summary = [
        f"Design            : {d.name}",
        f"Generated         : {_dt.datetime.now():%Y-%m-%d %H:%M}   "
        f"(tool v{__version__})",
        f"Units             : {unit_name(units)}",
        f"Overall status    : {an.status()}",
        "",
        f"Final OD nominal  : {fin * k:.4f} {unit_name(units)}  "
        f"({fin * MM_PER_IN:.3f} mm, {french(fin):.2f} Fr)",
        f"Final OD band     : {flo * k:.4f} – {fhi * k:.4f}  "
        f"[{METHOD_LABELS[an.method]}]",
        f"OD ceiling (USL)  : {acc.od_usl * k:.4f}" if acc.od_usl > 0
        else "OD ceiling (USL)  : not set",
        "",
        "ANALYSIS SETTINGS",
        f"  acceptance method        : {METHOD_LABELS[an.method]}",
        f"  worst-case evaluation    : {s.wc_mode}",
        f"  tolerance interpretation : ±{s.k_sigma:g}σ",
        f"  modified-RSS factor      : {s.mrss_factor:g}",
        f"  reflow model (global)    : {s.reflow_model}; subtract channel "
        f"area: {s.subtract_channels}",
        f"  Monte Carlo              : {'on' if an.mc is not None else 'off'}"
        + (f", N={s.mc_n}, {s.mc_sampling}, seed={s.mc_seed}, default dist="
           f"{s.default_dist}, truncated={s.truncate}" if an.mc is not None
           else ""),
        f"  capability method        : {s.cap_method}",
    ]
    if an.warnings:
        summary += ["", "WARNINGS"] + [f"  ! {w}" for w in an.warnings]
    if d.notes:
        summary += ["", "NOTES"] + [f"  {t}" for t in d.notes.splitlines()]
    with PdfPages(path) as pdf:
        _text_pages(pdf, "Catheter Radial Tolerance Stack-Up Report",
                    [("SUMMARY", summary),
                     ("CHECKS", check_rows(an, units))], th)
        fig = Figure(figsize=_PAGE)
        P.draw_cross_section(fig, an, th, units)
        pdf.savefig(fig)
        for fn in (lambda f: P.draw_buildup(f, an, th, units),
                   lambda f: P.draw_cutaway(f, an, th, units),
                   lambda f: P.draw_pareto(f, an, th, "final_od", "rss", units),
                   lambda f: P.draw_pareto(f, an, th, "final_od", "wc", units),
                   lambda f: P.draw_mc(f, an, th, "final_od", units),
                   lambda f: P.draw_heatshrink(f, an, th, units)):
            fig = Figure(figsize=_PAGE)
            fn(fig)
            pdf.savefig(fig)
        _text_pages(pdf, "Results — all metrics",
                    [("", metric_rows(an, units))], th)
        _text_pages(pdf, "Contributions to final OD",
                    [("", contribution_rows(an))], th)
        _text_pages(pdf, "Inputs", [("", input_rows(an, units))], th)
        info = pdf.infodict()
        info["Title"] = f"Stack-up report — {d.name}"
