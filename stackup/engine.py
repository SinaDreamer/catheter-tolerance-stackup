"""
Stack-up engine: geometry evaluation + Worst-Case / RSS / Modified-RSS /
Monte-Carlo analysis, sensitivities, contributions and process capability.

Statistical conventions
-----------------------
* Each tolerance zone  [nom−tm, nom+tp]  is treated as ±k·σ about its centre
  (k = Settings.k_sigma, default 3).
* RSS band   = centre ± 3·σ_stack,   σ_stack = √Σ (sᵢ·halfᵢ/k)²
* MRSS band  = centre ± min(f·RSS, WC_linear)    (Bender, f default 1.5)
* WC band    = exact corner evaluation (default) or linearised Σ|sᵢ|·halfᵢ
* MC band    = [P0.135 %, P99.865 %] of the simulated population (±3σ equiv.)
where sᵢ are numerical sensitivities (central differences of the full
non-linear model) and centre = nominal + Σ sᵢ·(tpᵢ − tmᵢ)/2.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import Design, Dim, Layer

METHODS = ("wc", "rss", "mrss", "mc")
METHOD_LABELS = {"wc": "Worst-Case", "rss": "RSS (3σ)",
                 "mrss": "Modified RSS", "mc": "Monte Carlo"}
Q_LO, Q_HI = 0.135, 99.865         # ±3σ-equivalent percentiles

try:                                # fast inverse normal CDF
    from scipy.special import ndtri as _ndtri, ndtr as _ndtr
except Exception:                   # pragma: no cover - fallback
    from statistics import NormalDist as _ND
    _nd = _ND()
    _ndtri = np.vectorize(lambda u: _nd.inv_cdf(min(max(u, 1e-15), 1 - 1e-15)))
    _ndtr = np.vectorize(_nd.cdf)


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------
@dataclass
class MetricInfo:
    key: str
    label: str
    kind: str = "len"
    group: str = "layer"     # layer | lumen | channel | process | hs | final
    uid: Optional[str] = None


@dataclass
class CheckDef:
    key: str
    label: str
    lsl: Optional[float] = None
    usl: Optional[float] = None
    severity: str = "fail"   # fail | warn


class Ctx:
    def __init__(self, design: Design, values: Dict[str, object],
                 nominal: bool):
        self.design = design
        self.settings = design.settings
        self.acc = design.acceptance
        self.values = values
        self.nominal = nominal
        self.D = None
        self.loaded = None
        self.lumen = None
        self.reinf = None
        self.pending = None
        self.channel_sets = []
        self.hs_layers = []
        self.metrics: Dict[str, object] = {}
        self.info: Dict[str, MetricInfo] = {}
        self.checks: List[CheckDef] = []
        self.rings: List[dict] = []
        self.warnings: List[str] = []

    def value(self, gid: str, nom: float):
        return self.values.get(gid, nom)

    def metric(self, key, label, val, kind="len", group="layer", uid=None):
        self.metrics[key] = val
        if key not in self.info:
            self.info[key] = MetricInfo(key, label, kind, group, uid)

    def check(self, key, label, lsl=None, usl=None, severity="fail"):
        self.checks.append(CheckDef(key, label, lsl, usl, severity))

    def ring(self, layer: Layer, id_, od):
        self.rings.append(dict(uid=layer.uid, name=layer.name,
                               type=layer.TYPE, id=id_, od=od, layer=layer))

    def layer_done(self, layer: Layer, ring_id):
        self.ring(layer, ring_id, self.D)
        self.metric(f"od|{layer.uid}", f"OD over {layer.name}", self.D,
                    group="layer", uid=layer.uid)


def evaluate(design: Design, values: Optional[Dict[str, object]] = None) -> Ctx:
    """Run the layer stack once.  ``values`` maps dimension ids to scalars or
    equal-length numpy arrays; missing ids take their nominal value."""
    nominal = not values
    ctx = Ctx(design, values or {}, nominal)
    layers = design.layers
    if not layers or layers[0].TYPE != "mandrel":
        raise ValueError("The stack must start with a 'Mandrel / core' layer "
                         "(it defines the main lumen ID).")
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, L in enumerate(layers):
            if i > 0 and L.TYPE == "mandrel":
                raise ValueError("Only one Mandrel / core layer is allowed, "
                                 "and it must be first.")
            L.apply(ctx)
        _finalize(ctx)
    return ctx


def _finalize(ctx: Ctx) -> None:
    acc = ctx.acc
    if ctx.pending is not None:
        if ctx.nominal:
            ctx.warnings.append(
                f"Channels of '{ctx.pending.layer.name}' are not covered by "
                f"any tube — they sit proud on the surface and set the OD.")
        ctx.D = np.maximum(ctx.D, ctx.pending.env)
        ctx.pending = None
    ctx.metric("final_od", "Final OD", ctx.D, group="final")
    ctx.check("final_od", "Final OD within spec",
              lsl=acc.od_lsl if acc.od_lsl > 0 else None,
              usl=acc.od_usl if acc.od_usl > 0 else None)
    if acc.min_lumen > 0:
        ctx.check("lumen", "Main lumen ID ≥ minimum", lsl=acc.min_lumen)

    # channel → outer-surface wall, reduced by core eccentricity (≥ 0)
    e = ctx.value("design.ecc", acc.ecc.nom)
    for cs in ctx.channel_sets:
        for i, it in enumerate(cs.items):
            key = f"cover|{cs.layer.uid}|{i}"
            ctx.metric(key, f"{it['label']}: wall to final OD",
                       ctx.D / 2.0 - (it["r"] + it["od"] / 2.0) - e,
                       group="channel", uid=cs.layer.uid)
            ctx.check(key, f"{it['label']} wall to OD",
                      lsl=acc.min_cover if acc.min_cover > 0 else 0.0)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@dataclass
class DimRef:
    id: str
    dim: Dim
    label: str
    kind: str
    dist: str


def collect_dims(design: Design) -> List[DimRef]:
    out = []
    dd = design.settings.default_dist
    for L in design.layers:
        for gid, d, label, kind in L.dim_items():
            out.append(DimRef(gid, d, label, kind,
                              dd if d.dist == "default" else d.dist))
    e = design.acceptance.ecc
    if e.half > 0 or e.nom > 0:
        out.append(DimRef("design.ecc", e, "Core eccentricity", "len",
                          dd if e.dist == "default" else e.dist))
    return out


@dataclass
class CheckResult:
    key: str
    label: str
    lsl: Optional[float]
    usl: Optional[float]
    severity: str
    kind: str
    nominal: float
    lo: float
    hi: float
    margin: float
    passed: bool
    mc_yield: Optional[float] = None
    cpk: Optional[float] = None
    cpk_src: str = ""


def _stack(metrics: Dict[str, object], keys: List[str], n: int) -> np.ndarray:
    return np.vstack([np.broadcast_to(np.asarray(metrics[k], float), (n,))
                      for k in keys]) if keys else np.zeros((0, n))


class Analysis:
    """Runs every analysis method for a design.  Raises ValueError on invalid
    input (message is user-presentable)."""

    def __init__(self, design: Design):
        t0 = time.perf_counter()
        self.design = design
        s = self.settings = design.settings
        self.nom_ctx = evaluate(design)
        self.keys = list(self.nom_ctx.metrics)
        self.idx = {k: i for i, k in enumerate(self.keys)}
        self.info = self.nom_ctx.info
        self.warnings = list(self.nom_ctx.warnings)
        self.nominal = np.array([float(self.nom_ctx.metrics[k])
                                 for k in self.keys])
        self.dims = collect_dims(design)
        m, n = len(self.keys), len(self.dims)
        k = max(float(s.k_sigma), 1e-6)

        half = np.array([d.dim.half for d in self.dims])
        shift = np.array([(d.dim.tp - d.dim.tm) / 2.0 for d in self.dims])
        self.half = half

        # ---- sensitivities (central differences, one vectorised pass) -----
        if n:
            h = np.maximum(half, 1e-6) * 1e-3
            vals = {}
            for j, d in enumerate(self.dims):
                arr = np.full(2 * n, d.dim.nom)
                arr[2 * j] += h[j]
                arr[2 * j + 1] -= h[j]
                vals[d.id] = arr
            M = _stack(evaluate(design, vals).metrics, self.keys, 2 * n)
            self.S = (M[:, 0::2] - M[:, 1::2]) / (2.0 * h)
            self.S[~np.isfinite(self.S)] = 0.0
        else:
            self.S = np.zeros((m, 0))

        S = self.S
        self.center = self.nominal + (S @ shift if n else 0.0)
        self.wc_lin = np.abs(S) @ half if n else np.zeros(m)
        self.sigma = np.sqrt(((S * half / k) ** 2).sum(axis=1)) if n \
            else np.zeros(m)
        self.rss_half = 3.0 * self.sigma
        self.mrss_half = np.minimum(float(s.mrss_factor) * self.rss_half,
                                    self.wc_lin)

        # ---- worst case ----------------------------------------------------
        if s.wc_mode == "linear" or not n or not m:
            wc_lo, wc_hi = self.center - self.wc_lin, self.center + self.wc_lin
        else:
            vals = {}
            for j, d in enumerate(self.dims):
                col = S[:, j]
                up = np.where(col > 0, d.dim.hi,
                              np.where(col < 0, d.dim.lo, d.dim.nom))
                dn = np.where(col > 0, d.dim.lo,
                              np.where(col < 0, d.dim.hi, d.dim.nom))
                arr = np.empty(2 * m)
                arr[0::2], arr[1::2] = up, dn
                vals[d.id] = arr
            E = _stack(evaluate(design, vals).metrics, self.keys, 2 * m)
            ii = np.arange(m)
            a, b = E[ii, 2 * ii], E[ii, 2 * ii + 1]
            wc_lo = np.minimum(np.minimum(a, b), self.nominal)
            wc_hi = np.maximum(np.maximum(a, b), self.nominal)
        self.bands = {
            "wc": (wc_lo, wc_hi),
            "rss": (self.center - self.rss_half, self.center + self.rss_half),
            "mrss": (self.center - self.mrss_half,
                     self.center + self.mrss_half),
        }

        # ---- Monte Carlo -----------------------------------------------------
        self.mc = None
        self.mc_invalid = 0
        if s.mc_enabled or s.method == "mc":
            self.mc = self._monte_carlo()
            with np.errstate(invalid="ignore"):
                lo = np.nanpercentile(self.mc, Q_LO, axis=1)
                hi = np.nanpercentile(self.mc, Q_HI, axis=1)
            self.bands["mc"] = (lo, hi)
            self.mc_mean = np.nanmean(self.mc, axis=1)
            self.mc_std = np.nanstd(self.mc, axis=1, ddof=1)
        method = s.method if s.method in self.bands else "wc"
        self.method = method

        # ---- checks ------------------------------------------------------------
        self.checks = [self._judge(c) for c in self.nom_ctx.checks]
        self.elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _monte_carlo(self) -> np.ndarray:
        s = self.settings
        N = max(int(s.mc_n), 100)
        rng = np.random.default_rng(int(s.mc_seed))
        k = max(float(s.k_sigma), 1e-6)
        vals = {}
        for d in self.dims:
            D = d.dim
            if D.half <= 0:
                continue
            if s.mc_sampling == "lhs":
                u = (rng.permutation(N) + rng.random(N)) / N
            else:
                u = rng.random(N)
            u = np.clip(u, 1e-12, 1 - 1e-12)
            if d.dist == "uniform":
                x = D.lo + u * (D.hi - D.lo)
            elif d.dist == "triangular":
                lo, hi, c = D.lo, D.hi, D.nom
                f = (c - lo) / (hi - lo)
                x = np.where(u < f, lo + np.sqrt(u * (hi - lo) * (c - lo)),
                             hi - np.sqrt((1 - u) * (hi - lo) * (hi - c)))
            else:                                   # normal
                if s.truncate:
                    p = float(_ndtr(-k))
                    u = p + u * (1.0 - 2.0 * p)
                x = D.mid + (D.half / k) * _ndtri(u)
            vals[d.id] = x
        if not vals:
            return np.repeat(self.nominal[:, None], N, axis=1)
        ctx = evaluate(self.design, vals)
        M = _stack(ctx.metrics, self.keys, N)
        bad = ~np.isfinite(M)
        self.mc_invalid = int(bad.any(axis=0).sum())
        M[bad] = np.nan
        return M

    # ------------------------------------------------------------------
    def band(self, key: str, method: Optional[str] = None) -> Tuple[float, float]:
        method = method or self.method
        if method not in self.bands:
            method = "wc"
        lo, hi = self.bands[method]
        i = self.idx[key]
        return float(lo[i]), float(hi[i])

    def nom(self, key: str) -> float:
        return float(self.nominal[self.idx[key]])

    def samples(self, key: str) -> Optional[np.ndarray]:
        if self.mc is None:
            return None
        x = self.mc[self.idx[key]]
        return x[np.isfinite(x)]

    def capability(self, key, lsl=None, usl=None) -> dict:
        """Cp/Cpk etc. from MC samples (preferred) or the RSS model."""
        out = {}
        x = self.samples(key)
        i = self.idx[key]
        if x is not None and x.size > 10:
            mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
            out.update(mean=mu, std=sd, src="MC")
            ok = np.ones_like(x, bool)
            if lsl is not None:
                ok &= x >= lsl
            if usl is not None:
                ok &= x <= usl
            out["yield"] = float(ok.mean())
            if self.settings.cap_method == "percentile":
                plo, p50, phi = np.percentile(x, [Q_LO, 50, Q_HI])
                cpu = (usl - p50) / (phi - p50) if usl is not None and \
                    phi > p50 else None
                cpl = (p50 - lsl) / (p50 - plo) if lsl is not None and \
                    p50 > plo else None
                spread = phi - plo
                out["src"] = "MC percentile"
            else:
                cpu = (usl - mu) / (3 * sd) if usl is not None and sd > 0 \
                    else None
                cpl = (mu - lsl) / (3 * sd) if lsl is not None and sd > 0 \
                    else None
                spread = 6 * sd
        else:
            mu, sd = float(self.center[i]), float(self.sigma[i])
            out.update(mean=mu, std=sd, src="RSS model")
            cpu = (usl - mu) / (3 * sd) if usl is not None and sd > 0 else None
            cpl = (mu - lsl) / (3 * sd) if lsl is not None and sd > 0 else None
            spread = 6 * sd
            if sd > 0:
                p = 1.0
                if usl is not None:
                    p -= float(_ndtr((mu - usl) / sd))
                if lsl is not None:
                    p -= float(_ndtr((lsl - mu) / sd))
                out["yield"] = max(p, 0.0)
        if lsl is not None and usl is not None and spread > 0:
            out["cp"] = (usl - lsl) / spread
        cands = [c for c in (cpu, cpl) if c is not None]
        out["cpk"] = min(cands) if cands else None
        out["cpu"], out["cpl"] = cpu, cpl
        return out

    def _judge(self, c: CheckDef) -> CheckResult:
        lo, hi = self.band(c.key)
        nom = self.nom(c.key)
        margins = []
        if c.lsl is not None:
            margins.append(lo - c.lsl)
        if c.usl is not None:
            margins.append(c.usl - hi)
        margin = min(margins) if margins else 0.0
        cap = self.capability(c.key, c.lsl, c.usl)
        return CheckResult(c.key, c.label, c.lsl, c.usl, c.severity,
                           self.info[c.key].kind, nom, lo, hi, margin,
                           margin >= -1e-12, cap.get("yield"), cap.get("cpk"),
                           cap.get("src", ""))

    # ------------------------------------------------------------------
    def status(self) -> str:
        if any(not c.passed and c.severity == "fail" for c in self.checks):
            return "FAIL"
        if any(not c.passed for c in self.checks):
            return "WARN"
        return "PASS"

    def contributions(self, key: str) -> List[dict]:
        """Per-dimension contributions to a metric, sorted (largest first)."""
        if not self.dims:
            return []
        row = self.S[self.idx[key]]
        k = max(float(self.settings.k_sigma), 1e-6)
        wc = np.abs(row) * self.half
        var = (row * self.half / k) ** 2
        tw, tv = wc.sum(), var.sum()
        out = []
        for j, d in enumerate(self.dims):
            if wc[j] <= 0:
                continue
            out.append(dict(label=d.label, id=d.id, kind=d.kind,
                            sens=float(row[j]), half=float(self.half[j]),
                            wc=float(wc[j]),
                            wc_pct=100 * wc[j] / tw if tw else 0.0,
                            var_pct=100 * var[j] / tv if tv else 0.0))
        out.sort(key=lambda r: r["var_pct"], reverse=True)
        return out

    def layer_rows(self) -> List[dict]:
        """Build-up rows: main lumen + each structural layer."""
        rows = [dict(key="lumen", name="Main lumen (mandrel)", type="mandrel",
                     uid=self.design.layers[0].uid)]
        for r in self.nom_ctx.rings:
            if r["type"] == "mandrel":
                continue
            rows.append(dict(key=f"od|{r['uid']}", name=r["name"],
                             type=r["type"], uid=r["uid"]))
        return rows
