"""
Data model — toleranced dimensions, the layer library and the Design container.

Every layer converts the current substrate (the stack built so far) into a new
substrate through its ``apply(ctx)`` method.  ``apply`` is written with numpy
operations only, so the very same code is evaluated

  * once with scalar nominal values               (nominal geometry),
  * once with 2·n perturbed vectors              (numerical sensitivities),
  * once with 2·m extreme-corner vectors         (exact worst case),
  * once with N random vectors                   (Monte Carlo),

which guarantees that every analysis method uses identical physics.

Conventions
-----------
* All lengths in inches.  Angles in degrees.
* Diameters are diametral, walls are radial (per side) unless noted.
* A ``Dim`` carries an asymmetric tolerance  nom  +tp / -tm  and an optional
  per-dimension distribution used by Monte Carlo.
"""
from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

DIST_CHOICES = ("default", "normal", "uniform", "triangular")


# ---------------------------------------------------------------------------
# Toleranced dimension
# ---------------------------------------------------------------------------
@dataclass
class Dim:
    nom: float
    tp: float = 0.0          # + tolerance (magnitude)
    tm: float = 0.0          # - tolerance (magnitude)
    dist: str = "default"    # MC distribution; "default" -> design setting

    @classmethod
    def sym(cls, nom: float, tol: float = 0.0, dist: str = "default") -> "Dim":
        return cls(float(nom), float(tol), float(tol), dist)

    @property
    def lo(self) -> float:
        return self.nom - self.tm

    @property
    def hi(self) -> float:
        return self.nom + self.tp

    @property
    def mid(self) -> float:          # centre of the tolerance zone
        return self.nom + (self.tp - self.tm) / 2.0

    @property
    def half(self) -> float:         # half-width of the tolerance zone
        return (self.tp + self.tm) / 2.0

    def to_json(self):
        return {"nom": self.nom, "tp": self.tp, "tm": self.tm,
                "dist": self.dist}

    @classmethod
    def from_json(cls, d) -> "Dim":
        if isinstance(d, (int, float)):
            return cls(float(d))
        if isinstance(d, dict):
            tp = float(d.get("tp", d.get("tol", 0.0)))
            return cls(float(d["nom"]), tp, float(d.get("tm", tp)),
                       d.get("dist", "default"))
        nom, tp, tm, *rest = d
        return cls(float(nom), float(tp), float(tm),
                   rest[0] if rest else "default")


@dataclass(frozen=True)
class ParamDef:
    key: str
    label: str
    default: float
    tol: float = 0.0
    kind: str = "len"                    # len | angle | ratio
    toleranced: bool = True
    tip: str = ""
    show_if: Optional[Tuple[str, Tuple[Any, ...]]] = None


@dataclass(frozen=True)
class OptionDef:
    key: str
    label: str
    default: Any
    choices: Optional[Tuple[str, ...]] = None
    tip: str = ""
    minimum: float = 0.0
    maximum: float = 1e6


# ---------------------------------------------------------------------------
# Layer library
# ---------------------------------------------------------------------------
LAYER_TYPES: Dict[str, type] = {}


def register(cls):
    LAYER_TYPES[cls.TYPE] = cls
    return cls


def _new_uid() -> str:
    return uuid.uuid4().hex[:6]


class Layer:
    TYPE = ""
    LABEL = ""
    DESCRIPTION = ""
    COLOR = "#cccccc"
    STRUCTURAL = True          # forms a ring of the shaft (vs process aid)
    PARAMS: Tuple[ParamDef, ...] = ()
    OPTIONS: Tuple[OptionDef, ...] = ()

    def __init__(self, name: Optional[str] = None, uid: Optional[str] = None,
                 **kw):
        self.uid = uid or _new_uid()
        self.name = name or self.LABEL
        self.dims: Dict[str, Dim] = {p.key: Dim.sym(p.default, p.tol)
                                     for p in self.PARAMS}
        self.options: Dict[str, Any] = {o.key: o.default for o in self.OPTIONS}
        for k, v in kw.items():
            if k in self.dims:
                self.dims[k] = v if isinstance(v, Dim) else Dim.sym(v)
            elif k in self.options:
                self.options[k] = v
            else:
                raise KeyError(f"{self.TYPE}: unknown field '{k}'")

    # ---- parameter helpers -------------------------------------------------
    def param_visible(self, p: ParamDef) -> bool:
        if p.show_if is None:
            return True
        k, allowed = p.show_if
        return self.options.get(k) in allowed

    def visible_params(self) -> List[ParamDef]:
        return [p for p in self.PARAMS if self.param_visible(p)]

    def dim_items(self) -> Iterator[Tuple[str, Dim, str, str]]:
        """(global id, Dim, label, kind) for every toleranced dimension."""
        for p in self.visible_params():
            if p.toleranced:
                yield (f"{self.uid}.{p.key}", self.dims[p.key],
                       f"{self.name}: {p.label}", p.kind)

    def v(self, ctx, key: str):
        return ctx.value(f"{self.uid}.{key}", self.dims[key].nom)

    def apply(self, ctx) -> None:          # pragma: no cover - abstract
        raise NotImplementedError

    # ---- serialisation ------------------------------------------------------
    def to_json(self) -> dict:
        return {"type": self.TYPE, "uid": self.uid, "name": self.name,
                "dims": {k: d.to_json() for k, d in self.dims.items()},
                "options": dict(self.options)}

    def _extra_from_json(self, d: dict) -> None:
        pass

    @staticmethod
    def from_json(d: dict) -> "Layer":
        cls = LAYER_TYPES[d["type"]]
        L = cls(name=d.get("name"), uid=d.get("uid"))
        for k, v in d.get("dims", {}).items():
            if k in L.dims:
                L.dims[k] = Dim.from_json(v)
        for k, v in d.get("options", {}).items():
            if k in L.options:
                L.options[k] = v
        L._extra_from_json(d)
        return L

    def clone(self) -> "Layer":
        c = copy.deepcopy(self)
        c.uid = _new_uid()
        c.name = self.name + " (copy)"
        return c


# ---- 1. mandrel ------------------------------------------------------------
@register
class MandrelLayer(Layer):
    TYPE = "mandrel"
    LABEL = "Mandrel / core"
    COLOR = "#aab4c0"
    DESCRIPTION = ("Precision-ground mandrel. The liner is formed "
                   "line-to-line on it, so main-lumen ID = mandrel OD "
                   "(diametral sensitivity 1). Must be the first layer.")
    PARAMS = (ParamDef("od", "Mandrel OD", 0.0200, 0.0003,
                       tip="Defines the main (central) lumen ID"),)

    def apply(self, ctx):
        od = self.v(ctx, "od")
        ctx.D = od
        ctx.loaded = od
        ctx.lumen = od
        ctx.metric("lumen", "Main lumen ID", od, group="lumen", uid=self.uid)
        ctx.ring(self, 0.0, od)


# ---- 2. thin wall layers ---------------------------------------------------
class _WallLayer(Layer):
    PARAMS = (ParamDef("wall", "Wall (radial)", 0.0015, 0.0003),)

    def apply(self, ctx):
        w = self.v(ctx, "wall")
        sub = ctx.D
        ctx.D = sub + 2.0 * w
        ctx.loaded = ctx.loaded + 2.0 * w
        ctx.layer_done(self, sub)


@register
class LinerLayer(_WallLayer):
    TYPE = "liner"
    LABEL = "Liner (PTFE)"
    COLOR = "#d3dcea"
    DESCRIPTION = "Lubricious liner (PTFE / etched PTFE / HDPE), radial wall."


@register
class TieLayer(_WallLayer):
    TYPE = "tie"
    LABEL = "Tie layer / adhesive"
    COLOR = "#ecdfb8"
    DESCRIPTION = "Thin tie/adhesive layer (e.g. Pebax/Tecoflex tie), radial wall."
    PARAMS = (ParamDef("wall", "Wall (radial)", 0.0006, 0.0001),)


@register
class CoatingLayer(_WallLayer):
    TYPE = "coating"
    LABEL = "Coating (hydrophilic)"
    COLOR = "#9fd18b"
    DESCRIPTION = "Surface coating (hydrophilic, parylene, …) — adds to final OD."
    PARAMS = (ParamDef("wall", "Coating thickness", 0.0004, 0.0001),)


@register
class GenericWallLayer(_WallLayer):
    TYPE = "wall"
    LABEL = "Generic wall layer"
    COLOR = "#c8c8c8"
    DESCRIPTION = "Any conformal layer described by its radial wall."


# ---- 3. reinforcement ------------------------------------------------------
_WIRE_PARAMS = (
    ParamDef("dia", "Wire Ø", 0.0015, 0.0001, show_if=("shape", ("round",))),
    ParamDef("thick", "Wire thickness (radial)", 0.0012, 0.0001,
             show_if=("shape", ("flat",)), tip="Governs radial build"),
    ParamDef("width", "Wire width (OD-neutral)", 0.0040, 0.0003,
             show_if=("shape", ("flat",)),
             tip="Does not enter the radial build — recorded / drawn only"),
)


class _WireLayer(Layer):
    PARAMS = _WIRE_PARAMS

    def wire_h(self, ctx):
        return self.v(ctx, "dia") if self.options["shape"] == "round" \
            else self.v(ctx, "thick")


@register
class BraidLayer(_WireLayer):
    TYPE = "braid"
    LABEL = "Braid"
    COLOR = "#7d8a99"
    DESCRIPTION = ("Braid crown: at a crossover N wires stack per side → "
                   "diametral build = 2·N·h (h = wire Ø or thickness).")
    OPTIONS = (
        OptionDef("shape", "Wire shape", "flat", ("flat", "round")),
        OptionDef("stack", "Wires stacked per side at crossover", 2,
                  minimum=1, maximum=4,
                  tip="2 for a standard 1-over-1 / 2-over-2 crossover"),
        OptionDef("carriers", "Carriers (drawing)", 16, minimum=2, maximum=64),
        OptionDef("ppi", "PPI (drawing)", 80.0, minimum=5, maximum=400),
    )

    def apply(self, ctx):
        h = self.wire_h(ctx)
        sub = ctx.D
        build = 2.0 * float(self.options["stack"]) * h
        ctx.D = sub + build
        ctx.loaded = ctx.loaded + build
        ctx.reinf = ctx.D
        ctx.layer_done(self, sub)


@register
class CoilLayer(_WireLayer):
    TYPE = "coil"
    LABEL = "Coil"
    COLOR = "#5d6977"
    DESCRIPTION = ("Wound coil: diametral build = 2·(layers)·h "
                   "(h = wire Ø or thickness).")
    PARAMS = _WIRE_PARAMS + (
        ParamDef("pitch", "Pitch (drawing)", 0.005, toleranced=False),)
    OPTIONS = (
        OptionDef("shape", "Wire shape", "flat", ("flat", "round")),
        OptionDef("layers", "Coil layers", 1, minimum=1, maximum=4),
    )

    def apply(self, ctx):
        h = self.wire_h(ctx)
        sub = ctx.D
        build = 2.0 * float(self.options["layers"]) * h
        ctx.D = sub + build
        ctx.loaded = ctx.loaded + build
        ctx.reinf = ctx.D
        ctx.layer_done(self, sub)


# ---- 4. tubes ----------------------------------------------------------------
_FIT_OPTION = OptionDef("check_fit", "Check tube loads over substrate", True,
                        tip="Loading clearance = tube ID − loaded stack OD")


@register
class ReflowTubeLayer(Layer):
    TYPE = "reflow"
    LABEL = "Reflowed polymer tube"
    COLOR = "#2a9d8f"
    DESCRIPTION = (
        "Pre-reflow extruded tube melted onto the substrate. Its annulus area "
        "(plus embedded channel area, if enabled) is conserved:\n"
        "  exact:     Dout = √(Dsub² + ODp² − IDp² [+ ΣODch²]) − 2p\n"
        "  thin-wall: Dout = Dsub + 2·[(ODp² − IDp² [+ ΣODch²])/(4·Dsub) − p]\n"
        "If peripheral channels precede this layer they are embedded in it; "
        "if the melt cannot bury them the channel envelope governs the OD.")
    PARAMS = (
        ParamDef("id", "Pre-reflow tube ID", 0.0300, 0.0006),
        ParamDef("od", "Pre-reflow tube OD", 0.0440, 0.0006),
        ParamDef("pen", "Interstice penetration (radial)", 0.0004, 0.0,
                 tip="Melt sinking into braid / channel voids — reduces "
                     "net build"),
        ParamDef("clr", "Min loading clearance (diametral)", 0.0,
                 toleranced=False),
    )
    OPTIONS = (
        _FIT_OPTION,
        OptionDef("model", "Reflow model", "global",
                  ("global", "exact", "thin-wall"),
                  tip="'global' follows the Analysis-tab setting"),
    )

    def apply(self, ctx):
        idp, odp, pen = self.v(ctx, "id"), self.v(ctx, "od"), self.v(ctx, "pen")
        sub = ctx.D
        a4 = odp ** 2 - idp ** 2                # 4·A/π of the tube annulus
        ch = ctx.pending
        if ch is not None and ctx.settings.subtract_channels:
            a4 = a4 + ch.area4                  # channels occupy the annulus
        model = self.options["model"]
        if model == "global":
            model = ctx.settings.reflow_model
        if model == "exact":
            dout = np.sqrt(sub ** 2 + a4) - 2.0 * pen
        else:
            dout = sub + 2.0 * (a4 / (4.0 * sub) - pen)
        poly_only = (np.sqrt(sub ** 2 + odp ** 2 - idp ** 2) - 2.0 * pen
                     if model == "exact" else
                     sub + 2.0 * ((odp ** 2 - idp ** 2) / (4.0 * sub) - pen))
        if ctx.nominal and (odp <= idp or poly_only <= sub):
            raise ValueError(
                f"'{self.name}': reflow model yields a non-positive wall — "
                f"tube annulus too small or penetration too large for "
                f"substrate Ø{sub:.4f}\".")
        if self.options["check_fit"]:
            key = f"fit|{self.uid}"
            ctx.metric(key, f"{self.name}: loading clearance",
                       idp - ctx.loaded, group="process", uid=self.uid)
            ctx.check(key, f"{self.name} loads over substrate",
                      lsl=self.dims["clr"].nom)
        loaded = np.sqrt(np.maximum(ctx.loaded, idp) ** 2 + odp ** 2 - idp ** 2)
        if ch is not None:
            key = f"embed|{self.uid}"
            ctx.metric(key, f"{self.name}: channel burial margin (Ø)",
                       dout - ch.env, group="channel", uid=self.uid)
            ctx.check(key, f"{self.name} buries channels", lsl=0.0,
                      severity="warn")
            if ctx.nominal and dout < ch.env:
                ctx.warnings.append(
                    f"'{self.name}': reflowed OD Ø{dout:.4f}\" < channel "
                    f"envelope Ø{ch.env:.4f}\" — channels are not buried; "
                    f"the channel envelope governs the OD (local lobing). "
                    f"Increase the pre-reflow annulus area or accept it.")
            dout = np.maximum(dout, ch.env)
            loaded = np.maximum(loaded, ch.env)
            ch.host = self.name
            ctx.pending = None
        ctx.D = dout
        ctx.loaded = loaded
        ctx.metric(f"wall|{self.uid}", f"{self.name}: effective wall",
                   (dout - sub) / 2.0, group="layer", uid=self.uid)
        ctx.layer_done(self, sub)


@register
class ExtrudedTubeLayer(Layer):
    TYPE = "extruded"
    LABEL = "Extruded tube (slip fit)"
    COLOR = "#8e7cc3"
    DESCRIPTION = ("Tube slid over the stack without reflow. If its ID is "
                   "smaller than the stack it is stretched with conserved "
                   "area: Dout = √(max(Dsub, ID)² + OD² − ID²). Captures any "
                   "preceding peripheral channels in the gap.")
    PARAMS = (
        ParamDef("id", "Tube ID", 0.0375, 0.0006),
        ParamDef("od", "Tube OD", 0.0490, 0.0006),
        ParamDef("clr", "Min loading clearance (diametral)", 0.0,
                 toleranced=False),
    )
    OPTIONS = (_FIT_OPTION,)

    def apply(self, ctx):
        idt, odt = self.v(ctx, "id"), self.v(ctx, "od")
        sub, sub_l = ctx.D, ctx.loaded
        ch = ctx.pending
        if ch is not None:
            sub = np.maximum(sub, ch.env)
            sub_l = np.maximum(sub_l, ch.env)
            ch.host = self.name
            ctx.pending = None
        if self.options["check_fit"]:
            key = f"fit|{self.uid}"
            ctx.metric(key, f"{self.name}: loading clearance", idt - sub_l,
                       group="process", uid=self.uid)
            ctx.check(key, f"{self.name} loads over substrate",
                      lsl=self.dims["clr"].nom)
        ring_id = np.maximum(sub, idt)
        ctx.D = np.sqrt(ring_id ** 2 + odt ** 2 - idt ** 2)
        ctx.loaded = np.sqrt(np.maximum(sub_l, idt) ** 2 + odt ** 2 - idt ** 2)
        ctx.layer_done(self, ring_id)


# ---- 5. peripheral channels -----------------------------------------------
@dataclass
class Channel:
    label: str
    kind: str = "lined"                    # lined (tube) | bare (lumen)
    id: Dim = field(default_factory=lambda: Dim.sym(0.005, 0.0004))
    wall: Dim = field(default_factory=lambda: Dim.sym(0.0012, 0.0002))
    angle: Dim = field(default_factory=lambda: Dim.sym(0.0, 0.0))
    radius: Dim = field(default_factory=lambda: Dim.sym(0.018, 0.0))

    def to_json(self):
        return {"label": self.label, "kind": self.kind,
                "id": self.id.to_json(), "wall": self.wall.to_json(),
                "angle": self.angle.to_json(), "radius": self.radius.to_json()}

    @classmethod
    def from_json(cls, d):
        return cls(d.get("label", "C"), d.get("kind", "lined"),
                   Dim.from_json(d["id"]), Dim.from_json(d["wall"]),
                   Dim.from_json(d["angle"]), Dim.from_json(d["radius"]))


class ChannelSet:
    """Evaluated channel geometry handed to the embedding layer."""

    def __init__(self, layer):
        self.layer = layer
        self.items: List[dict] = []    # label, kind, r, ang(rad), id, od
        self.env = 0.0
        self.area4 = 0.0
        self.sub = 0.0
        self.host: Optional[str] = None


@register
class ChannelLayer(Layer):
    TYPE = "channels"
    LABEL = "Peripheral channels / lumens"
    COLOR = "#e8a13d"
    STRUCTURAL = False
    DESCRIPTION = (
        "Any number of peripheral channels (lined tubes, e.g. PI/PTFE, or bare "
        "lumens) placed on the current substrate at any angle and size — "
        "symmetric or not. The NEXT reflowed/extruded tube embeds them. "
        "Auto radius: channel rests on the substrate (r = Dsub/2 + ODch/2 + "
        "gap). Manual radius: centre radius entered per channel, which also "
        "allows off-centre working lumens.")
    PARAMS = (ParamDef("gap", "Radial gap to substrate (auto radius)", 0.0,
                       0.0, show_if=("radius_mode", ("auto",))),)
    OPTIONS = (OptionDef("radius_mode", "Channel centre radius", "auto",
                         ("auto", "manual")),)

    def __init__(self, name=None, uid=None, channels=None, **kw):
        super().__init__(name, uid, **kw)
        self.channels: List[Channel] = channels if channels is not None \
            else self.equal_spacing(4, 45.0, 0.005, 0.0004, 0.0012, 0.0002)

    @staticmethod
    def equal_spacing(n, start_deg, id_, id_tol, wall, wall_tol,
                      kind="lined", radius=0.016, prefix="C") -> List[Channel]:
        out = []
        for i in range(int(n)):
            out.append(Channel(f"{prefix}{i + 1}", kind,
                               Dim.sym(id_, id_tol), Dim.sym(wall, wall_tol),
                               Dim.sym((start_deg + i * 360.0 / n) % 360.0),
                               Dim.sym(radius)))
        return out

    def dim_items(self):
        yield from super().dim_items()
        manual = self.options["radius_mode"] == "manual"
        for i, c in enumerate(self.channels):
            p = f"{self.uid}.c{i}"
            yield (f"{p}.id", c.id, f"{self.name}: {c.label} ID", "len")
            if c.kind == "lined":
                yield (f"{p}.wall", c.wall, f"{self.name}: {c.label} wall",
                       "len")
            yield (f"{p}.angle", c.angle, f"{self.name}: {c.label} angle",
                   "angle")
            if manual:
                yield (f"{p}.radius", c.radius,
                       f"{self.name}: {c.label} centre radius", "len")

    def apply(self, ctx):
        sub = ctx.D
        manual = self.options["radius_mode"] == "manual"
        gap = self.v(ctx, "gap")
        cs = ChannelSet(self)
        cs.sub = sub
        env = None
        area4 = 0.0
        for i, c in enumerate(self.channels):
            p = f"{self.uid}.c{i}"
            idv = ctx.value(f"{p}.id", c.id.nom)
            wall = ctx.value(f"{p}.wall", c.wall.nom) if c.kind == "lined" \
                else 0.0
            od = idv + 2.0 * wall
            ang = np.radians(ctx.value(f"{p}.angle", c.angle.nom))
            if manual:
                r = ctx.value(f"{p}.radius", c.radius.nom)
                key = f"chcore|{self.uid}|{i}"
                ctx.metric(key, f"{c.label}: clearance to substrate",
                           r - od / 2.0 - sub / 2.0, group="channel",
                           uid=self.uid)
                ctx.check(key, f"{c.label} clears substrate", lsl=0.0)
            else:
                r = sub / 2.0 + od / 2.0 + gap
            cs.items.append(dict(label=c.label, kind=c.kind, r=r, ang=ang,
                                 id=idv, od=od, nom_ang=c.angle.nom))
            ctx.metric(f"chid|{self.uid}|{i}", f"{c.label}: ID", idv,
                       group="channel", uid=self.uid)
            outer = 2.0 * r + od
            env = outer if env is None else np.maximum(env, outer)
            area4 = area4 + od ** 2
        cs.env = env if env is not None else sub
        cs.area4 = area4

        # webs between angularly adjacent channels
        n = len(cs.items)
        order = sorted(range(n), key=lambda k: cs.items[k]["nom_ang"] % 360.0)
        pairs = []
        if n == 2:
            pairs = [(order[0], order[1])]
        elif n > 2:
            pairs = [(order[k], order[(k + 1) % n]) for k in range(n)]
        min_web = ctx.acc.min_web
        for a, b in pairs:
            A, B = cs.items[a], cs.items[b]
            dist = np.sqrt(A["r"] ** 2 + B["r"] ** 2
                           - 2.0 * A["r"] * B["r"] * np.cos(B["ang"] - A["ang"]))
            key = f"web|{self.uid}|{a}-{b}"
            ctx.metric(key, f"Web {A['label']}–{B['label']}",
                       dist - (A["od"] + B["od"]) / 2.0, group="channel",
                       uid=self.uid)
            ctx.check(key, f"Web {A['label']}–{B['label']}",
                      lsl=min_web if min_web > 0 else 0.0)

        if ctx.pending is not None and ctx.nominal:
            ctx.warnings.append(
                f"'{self.name}' follows another channel layer that was never "
                f"embedded — the earlier channels are ignored.")
        ctx.pending = cs
        ctx.channel_sets.append(cs)

    def to_json(self):
        d = super().to_json()
        d["channels"] = [c.to_json() for c in self.channels]
        return d

    def _extra_from_json(self, d):
        if "channels" in d:
            self.channels = [Channel.from_json(c) for c in d["channels"]]


# ---- 6. heat shrink (process aid) -----------------------------------------
@register
class HeatShrinkLayer(Layer):
    TYPE = "heatshrink"
    LABEL = "Heat shrink (process aid)"
    COLOR = "#cfcfcf"
    STRUCTURAL = False
    DESCRIPTION = (
        "Peelable FEP/PET heat shrink used to drive reflow — NOT part of the "
        "final OD. Checks:\n"
        "  • loading: expanded ID ≥ loaded stack OD + clearance\n"
        "  • drive: recovered ID < OD under the shrink (compression)\n"
        "  • no crush: recovered ID > braid/coil crown OD\n"
        "  • shrink ratio = expanded ID / recovered ID ≥ minimum")
    PARAMS = (
        ParamDef("exp_id", "Expanded ID (as supplied)", 0.0780, 0.0012),
        ParamDef("rec_id", "Recovered ID (free)", 0.0470, 0.0012),
        ParamDef("rec_wall", "Recovered wall (info)", 0.0070,
                 toleranced=False),
        ParamDef("clr", "Required loading clearance (Ø)", 0.0035,
                 toleranced=False),
    )
    OPTIONS = (OptionDef("min_ratio", "Minimum shrink ratio", 1.2,
                         minimum=1.0, maximum=4.0),)

    def apply(self, ctx):
        e, r = self.v(ctx, "exp_id"), self.v(ctx, "rec_id")
        clr = self.dims["clr"].nom
        D = ctx.D
        loaded = ctx.loaded
        if ctx.pending is not None:
            D = np.maximum(D, ctx.pending.env)
            loaded = np.maximum(loaded, ctx.pending.env)
        u, n = self.uid, self.name
        ctx.metric(f"hs_loaded|{u}", f"{n}: loaded stack OD", loaded,
                   group="hs", uid=u)
        ctx.metric(f"hs_od|{u}", f"{n}: OD under shrink", D, group="hs", uid=u)
        ctx.metric(f"hs_exp|{u}", f"{n}: expanded ID", e, group="hs", uid=u)
        ctx.metric(f"hs_rec|{u}", f"{n}: recovered ID", r, group="hs", uid=u)
        ctx.metric(f"hs_load|{u}", f"{n}: loading margin",
                   e - loaded - clr, group="process", uid=u)
        ctx.check(f"hs_load|{u}", f"{n}: expanded ID ≥ loaded OD + clr",
                  lsl=0.0)
        ctx.metric(f"hs_drive|{u}", f"{n}: recovery interference (Ø)",
                   D - r, group="process", uid=u)
        ctx.check(f"hs_drive|{u}", f"{n}: recovered ID < OD (drive)",
                  lsl=0.0)
        if ctx.reinf is not None:
            ctx.metric(f"hs_reinf|{u}", f"{n}: reinforcement crown OD",
                       ctx.reinf, group="hs", uid=u)
            ctx.metric(f"hs_crush|{u}", f"{n}: crush margin (Ø)",
                       r - ctx.reinf, group="process", uid=u)
            ctx.check(f"hs_crush|{u}", f"{n}: recovered ID > crown OD "
                      f"(no crush)", lsl=0.0)
        ctx.metric(f"hs_ratio|{u}", f"{n}: shrink ratio", e / r,
                   kind="ratio", group="process", uid=u)
        ctx.check(f"hs_ratio|{u}", f"{n}: shrink ratio ≥ "
                  f"{float(self.options['min_ratio']):.2f}",
                  lsl=float(self.options["min_ratio"]))
        ctx.hs_layers.append(self)


# ---------------------------------------------------------------------------
# Design container
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    units: str = "in"
    method: str = "wc"                # wc | rss | mrss | mc   (acceptance)
    wc_mode: str = "extreme"          # extreme | linear
    k_sigma: float = 3.0              # each tolerance = ±k·σ
    mrss_factor: float = 1.5          # Bender modified RSS
    mc_enabled: bool = True
    mc_n: int = 20000
    mc_sampling: str = "lhs"          # lhs | random
    mc_seed: int = 12345
    default_dist: str = "normal"      # normal | uniform | triangular
    truncate: bool = True             # truncate normals at tolerance limits
    cap_method: str = "normal"        # normal | percentile (ISO 22514-2)
    reflow_model: str = "exact"       # exact | thin-wall
    subtract_channels: bool = True


@dataclass
class Acceptance:
    od_usl: float = 0.0620            # OD ceiling
    od_lsl: float = 0.0               # 0 = not used
    min_lumen: float = 0.0            # 0 = not used
    min_web: float = 0.0010
    min_cover: float = 0.0020
    ecc: Dim = field(default_factory=lambda: Dim(0.0, 0.0, 0.0))


class Design:
    def __init__(self, name="Untitled design", layers=None, settings=None,
                 acceptance=None, notes=""):
        self.name = name
        self.layers: List[Layer] = layers or []
        self.settings = settings or Settings()
        self.acceptance = acceptance or Acceptance()
        self.notes = notes

    def find(self, uid) -> Optional[Layer]:
        return next((L for L in self.layers if L.uid == uid), None)

    def to_json(self) -> dict:
        acc = asdict(self.acceptance)
        acc["ecc"] = self.acceptance.ecc.to_json()
        return {"format": "catheter-stackup", "version": 3,
                "name": self.name, "notes": self.notes,
                "settings": asdict(self.settings), "acceptance": acc,
                "layers": [L.to_json() for L in self.layers]}

    @classmethod
    def from_json(cls, d: dict) -> "Design":
        s = Settings()
        for k, v in d.get("settings", {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        a = Acceptance()
        for k, v in d.get("acceptance", {}).items():
            if k == "ecc":
                a.ecc = Dim.from_json(v)
            elif hasattr(a, k):
                setattr(a, k, float(v))
        return cls(d.get("name", "Untitled design"),
                   [Layer.from_json(x) for x in d.get("layers", [])],
                   s, a, d.get("notes", ""))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Design":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def copy(self) -> "Design":
        return Design.from_json(self.to_json())
