"""Built-in example constructions.

All dimensions are fictitious, illustrative values chosen to demonstrate
the tool; they do not describe any specific product.
"""
from .model import (Acceptance, BraidLayer, Channel, ChannelLayer, CoilLayer,
                    Design, Dim, ExtrudedTubeLayer, HeatShrinkLayer,
                    LinerLayer, MandrelLayer, ReflowTubeLayer, Settings,
                    TieLayer)

# example channel: lined polyimide tube
CH_ID, CH_ID_TOL, CH_WALL, CH_WALL_TOL = 0.0050, 0.0004, 0.0012, 0.0002


def _core():
    return [
        MandrelLayer(od=Dim.sym(0.0200, 0.0003)),
        LinerLayer("PTFE liner", wall=Dim.sym(0.0015, 0.0003)),
        BraidLayer("SS braid", shape="flat", thick=Dim.sym(0.0012, 0.0001),
                   width=Dim.sym(0.0040, 0.0003)),
    ]


def _jacket_and_hs():
    return [
        ReflowTubeLayer("Outer Pebax jacket", id=Dim.sym(0.0560, 0.0006),
                        od=Dim.sym(0.0660, 0.0006), pen=Dim.sym(0.0004)),
        HeatShrinkLayer("FEP heat shrink", exp_id=Dim.sym(0.0780, 0.0012),
                        rec_id=Dim.sym(0.0470, 0.0012)),
    ]


def example_4ch() -> Design:
    """Example: central lumen + 4 lined pull-wire channels."""
    layers = _core() + [
        ChannelLayer("PI pull-wire channels",
                     channels=ChannelLayer.equal_spacing(
                         4, 45.0, CH_ID, CH_ID_TOL, CH_WALL, CH_WALL_TOL,
                         prefix="PI")),
        ReflowTubeLayer("Mid Pebax (tie)", id=Dim.sym(0.0300, 0.0006),
                        od=Dim.sym(0.0440, 0.0006), pen=Dim.sym(0.0004)),
    ] + _jacket_and_hs()
    return Design("Example shaft — central lumen + 4× PI channels", layers,
                  Settings(), Acceptance(od_usl=0.0620))


def three_channel() -> Design:
    d = example_4ch()
    d.name = "Example — 3 channels @ 120°"
    d.layers[3].channels = ChannelLayer.equal_spacing(
        3, 90.0, CH_ID, CH_ID_TOL, CH_WALL, CH_WALL_TOL, prefix="PI")
    return d


def two_channel() -> Design:
    d = example_4ch()
    d.name = "Example — 2 channels @ 180° (bidirectional)"
    d.layers[3].channels = ChannelLayer.equal_spacing(
        2, 90.0, CH_ID, CH_ID_TOL, CH_WALL, CH_WALL_TOL, prefix="PI")
    return d


def asymmetric() -> Design:
    """2 pull-wire channels + 1 larger bare working lumen, non-symmetric."""
    d = example_4ch()
    d.name = "Example — asymmetric: 2× PI + working lumen"
    d.layers[3].channels = [
        Channel("PI1", "lined", Dim.sym(CH_ID, CH_ID_TOL),
                Dim.sym(CH_WALL, CH_WALL_TOL), Dim.sym(60.0, 1.0)),
        Channel("PI2", "lined", Dim.sym(CH_ID, CH_ID_TOL),
                Dim.sym(CH_WALL, CH_WALL_TOL), Dim.sym(120.0, 1.0)),
        Channel("WL", "bare", Dim.sym(0.0110, 0.0006), Dim.sym(0.0),
                Dim.sym(270.0, 1.0)),
    ]
    d.layers[4].dims["od"] = Dim.sym(0.0470, 0.0006)
    return d


def coil_extruded() -> Design:
    """Simple coil-reinforced shaft with a slip-fit outer tube."""
    layers = [
        MandrelLayer(od=Dim.sym(0.0250, 0.0003)),
        LinerLayer("PTFE liner", wall=Dim.sym(0.0009, 0.0002)),
        TieLayer("Tie layer", wall=Dim.sym(0.0006, 0.0001)),
        CoilLayer("Flat-wire coil", shape="flat",
                  thick=Dim.sym(0.0020, 0.0001), width=Dim.sym(0.0060, 0.0003)),
        ExtrudedTubeLayer("Pebax jacket", id=Dim.sym(0.0375, 0.0006),
                          od=Dim.sym(0.0490, 0.0006)),
    ]
    return Design("Example — coil-reinforced shaft, slip-fit jacket", layers,
                  Settings(), Acceptance(od_usl=0.0520, min_web=0,
                                         min_cover=0))


PRESETS = {
    "Example — 4× PI channels": example_4ch,
    "Example — 3 channels @ 120°": three_channel,
    "Example — 2 channels @ 180°": two_channel,
    "Example — 2× PI + working lumen": asymmetric,
    "Example — coil + slip-fit jacket": coil_extruded,
}


def blank() -> Design:
    return Design("New design", [MandrelLayer(), LinerLayer()])
