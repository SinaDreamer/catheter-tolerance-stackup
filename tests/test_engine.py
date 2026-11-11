"""Engine verification suite (hand calculations, statistics, I/O, and an
optional cross-check against the previous v2 implementation).

Run:  python main.py --selftest      or      python -m unittest discover tests

Expected values are derived from the example design's own inputs, so the
tests stay valid if the example dimensions are changed.
"""
import importlib.util
import math
import os
import tempfile
import unittest

import numpy as np

from stackup.engine import Analysis, evaluate
from stackup.model import (BraidLayer, Channel, Design, Dim,
                           ExtrudedTubeLayer, LinerLayer, MandrelLayer,
                           Settings)
from stackup import presets

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY = os.path.join(HERE, "..", "legacy", "main_v2.py")


def _legacy():
    spec = importlib.util.spec_from_file_location("legacy_v2", LEGACY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ex():
    """Example design + handy nominal values."""
    d = presets.example_4ch()
    m, ln, br, ch, mid, jac, hs = d.layers
    v = dict(
        mandrel=m.dims["od"].nom, liner=ln.dims["wall"].nom,
        h=br.dims["thick"].nom, ch_id=ch.channels[0].id.nom,
        ch_wall=ch.channels[0].wall.nom, mid_id=mid.dims["id"].nom,
        mid_od=mid.dims["od"].nom, mid_pen=mid.dims["pen"].nom,
        jac_id=jac.dims["id"].nom, jac_od=jac.dims["od"].nom,
        jac_pen=jac.dims["pen"].nom, exp=hs.dims["exp_id"].nom,
        rec=hs.dims["rec_id"].nom)
    v["braid_od"] = v["mandrel"] + 2 * v["liner"] + 4 * v["h"]
    v["ch_od"] = v["ch_id"] + 2 * v["ch_wall"]
    v["env"] = v["braid_od"] + 2 * v["ch_od"]
    return d, v


def _simple(settings=None):
    """Linear stack: mandrel + liner + round-wire braid."""
    return Design("simple", [
        MandrelLayer(od=Dim.sym(0.020, 0.0003)),
        LinerLayer(wall=Dim.sym(0.001, 0.0002)),
        BraidLayer(shape="round", dia=Dim.sym(0.001, 0.0001)),
    ], settings or Settings(mc_n=40000))


@unittest.skipUnless(os.path.exists(LEGACY), "legacy v2 file not present")
class TestLegacyCompatibility(unittest.TestCase):
    """Thin-wall model + no channel-area subtraction must reproduce v2."""

    def _pair(self):
        leg = _legacy()
        d, _ = _ex()
        m, ln, br, ch, mid, jac, hs = d.layers
        c = ch.channels[0]
        specs = dict(
            mandrel=leg.MandrelSpec(m.dims["od"].nom, m.dims["od"].tp),
            liner=leg.LinerSpec(ln.dims["wall"].nom, ln.dims["wall"].tp),
            braid=leg.BraidSpec(shape="rect", width=br.dims["width"].nom,
                                width_tol=br.dims["width"].tp,
                                thick=br.dims["thick"].nom,
                                thick_tol=br.dims["thick"].tp),
            mid=leg.ReflowPebaxSpec(mid.dims["id"].nom, mid.dims["id"].tp,
                                    mid.dims["od"].nom, mid.dims["od"].tp,
                                    mid.dims["pen"].nom),
            pi=leg.PIChannelSpec(len(ch.channels), c.id.nom, c.id.tp,
                                 c.wall.nom, c.wall.tp),
            jacket=leg.ReflowPebaxSpec(jac.dims["id"].nom, jac.dims["id"].tp,
                                       jac.dims["od"].nom, jac.dims["od"].tp,
                                       jac.dims["pen"].nom))
        r = leg.compute_stackup(**specs)
        d.settings.reflow_model = "thin-wall"
        d.settings.subtract_channels = False
        return r, Analysis(d)

    def test_nominals_match_v2(self):
        r, an = self._pair()
        rows = an.layer_rows()[1:]
        self.assertEqual(len(rows), len(r.layers))
        for row, L in zip(rows, r.layers):
            self.assertAlmostEqual(an.nom(row["key"]), L.od_nom, places=12)
        self.assertAlmostEqual(an.nom("lumen"), r.lumen_id_nom, places=12)

    def test_inner_bands_match_v2(self):
        r, an = self._pair()
        for row, L in zip(an.layer_rows()[1:3], r.layers[:2]):
            lo, hi = an.band(row["key"], "wc")
            self.assertAlmostEqual(lo, L.od_min_wc, places=10)
            self.assertAlmostEqual(hi, L.od_max_wc, places=10)
            rlo, rhi = an.band(row["key"], "rss")
            self.assertAlmostEqual(rlo, L.od_min_rss, places=10)
            self.assertAlmostEqual(rhi, L.od_max_rss, places=10)


class TestPhysics(unittest.TestCase):
    def test_exact_reflow_hand_calc(self):
        d, v = _ex()
        an = Analysis(d)
        a4 = v["mid_od"] ** 2 - v["mid_id"] ** 2 + 4 * v["ch_od"] ** 2
        mid = max(math.sqrt(v["braid_od"] ** 2 + a4) - 2 * v["mid_pen"],
                  v["env"])
        self.assertAlmostEqual(an.nom(an.layer_rows()[3]["key"]), mid,
                               places=12)
        jac = math.sqrt(mid ** 2 + v["jac_od"] ** 2 - v["jac_id"] ** 2) \
            - 2 * v["jac_pen"]
        self.assertAlmostEqual(an.nom("final_od"), jac, places=12)

    def test_channel_area_subtraction(self):
        d, v = _ex()
        mid = d.layers[4]
        mid.dims["od"] = Dim.sym(v["mid_od"] + 0.006, 0.0006)  # buries
        a4 = (v["mid_od"] + 0.006) ** 2 - v["mid_id"] ** 2
        ch4 = 4 * v["ch_od"] ** 2
        sub = v["braid_od"]
        for subtract, expect in (
                (True, math.sqrt(sub ** 2 + a4 + ch4) - 2 * v["mid_pen"]),
                (False, math.sqrt(sub ** 2 + a4) - 2 * v["mid_pen"])):
            d.settings.subtract_channels = subtract
            ctx = evaluate(d)
            self.assertAlmostEqual(float(ctx.metrics[f"od|{mid.uid}"]),
                                   max(expect, v["env"]), places=12)

    def test_thin_wall_formula(self):
        d, v = _ex()
        d.settings.reflow_model = "thin-wall"
        d.settings.subtract_channels = False
        ctx = evaluate(d)
        sub = v["braid_od"]
        t = (v["mid_od"] ** 2 - v["mid_id"] ** 2) / (4 * sub) - v["mid_pen"]
        od_mid = sub + 2 * max(t, v["ch_od"])
        t = (v["jac_od"] ** 2 - v["jac_id"] ** 2) / (4 * od_mid) - v["jac_pen"]
        self.assertAlmostEqual(float(ctx.metrics["final_od"]), od_mid + 2 * t,
                               places=12)

    def test_braid_width_neutral_and_round_equivalence(self):
        d, v = _ex()
        base = float(evaluate(d).metrics["final_od"])
        d.layers[2].dims["width"] = Dim.sym(0.009, 0.0002)
        self.assertAlmostEqual(float(evaluate(d).metrics["final_od"]), base,
                               places=12)
        d, v = _ex()
        d.layers[2].options["shape"] = "round"
        d.layers[2].dims["dia"] = Dim.sym(v["h"], 0.0001)
        self.assertAlmostEqual(float(evaluate(d).metrics["final_od"]), base,
                               places=12)

    def test_three_channel_web(self):
        _, v = _ex()
        an = Analysis(presets.three_channel())
        r = v["braid_od"] / 2 + v["ch_od"] / 2
        webs = [k for k in an.keys if k.startswith("web|")]
        self.assertEqual(len(webs), 3)
        for k in webs:
            self.assertAlmostEqual(an.nom(k), r * math.sqrt(3) - v["ch_od"],
                                   places=12)

    def test_two_channels_single_web(self):
        _, v = _ex()
        an = Analysis(presets.two_channel())
        webs = [k for k in an.keys if k.startswith("web|")]
        self.assertEqual(len(webs), 1)
        r = v["braid_od"] / 2 + v["ch_od"] / 2
        self.assertAlmostEqual(an.nom(webs[0]), 2 * r - v["ch_od"], places=12)

    def test_asymmetric_channels_and_bare_lumen(self):
        _, v = _ex()
        d = presets.asymmetric()
        an = Analysis(d)
        cs = an.nom_ctx.channel_sets[0]
        wl = [it for it in cs.items if it["label"] == "WL"][0]
        wl_id = [c for c in d.layers[3].channels if c.label == "WL"][0].id.nom
        self.assertEqual(wl["od"], wl_id)                # bare: OD = ID
        self.assertAlmostEqual(cs.env, v["braid_od"] + 2 * wl_id, places=12)

    def test_manual_radius_offcentre_lumen(self):
        d, v = _ex()
        ch = d.layers[3]
        ch.options["radius_mode"] = "manual"
        r, wl = 0.0200, 0.0075
        ch.channels = [Channel("WL", "bare", Dim.sym(wl, 0.0003), Dim.sym(0),
                               Dim.sym(0, 0), Dim.sym(r, 0.0002))]
        an = Analysis(d)
        key = [k for k in an.keys if k.startswith("chcore|")][0]
        self.assertAlmostEqual(an.nom(key), r - wl / 2 - v["braid_od"] / 2,
                               places=12)

    def test_extruded_tube_stretch(self):
        d = Design("x", [MandrelLayer(od=Dim.sym(0.030)),
                         ExtrudedTubeLayer(id=Dim.sym(0.028),
                                           od=Dim.sym(0.036))])
        an = Analysis(d)
        self.assertAlmostEqual(an.nom("final_od"),
                               math.sqrt(0.030 ** 2 + 0.036 ** 2 - 0.028 ** 2),
                               places=12)
        fit = [k for k in an.keys if k.startswith("fit|")][0]
        self.assertAlmostEqual(an.nom(fit), -0.002, places=12)
        d.layers[1].dims["id"] = Dim.sym(0.032)
        self.assertAlmostEqual(Analysis(d).nom("final_od"), 0.036, places=12)

    def test_heat_shrink_checks(self):
        d, v = _ex()
        an = Analysis(d)
        hs = [c for c in an.checks if c.key.startswith("hs_")]
        self.assertEqual(len(hs), 4)
        ratio = [k for k in an.keys if k.startswith("hs_ratio|")][0]
        self.assertAlmostEqual(an.nom(ratio), v["exp"] / v["rec"], places=12)
        self.assertTrue(all(c.passed for c in hs))


class TestStatistics(unittest.TestCase):
    def test_linear_stack_wc_and_rss(self):
        an = Analysis(_simple())
        lo, hi = an.band("final_od", "wc")
        wc = 0.0003 + 2 * 0.0002 + 4 * 0.0001
        self.assertAlmostEqual(hi - an.nom("final_od"), wc, places=12)
        self.assertAlmostEqual(an.nom("final_od") - lo, wc, places=12)
        rss = math.sqrt(0.0003 ** 2 + 0.0004 ** 2 + 0.0004 ** 2)
        rlo, rhi = an.band("final_od", "rss")
        self.assertAlmostEqual((rhi - rlo) / 2, rss, places=12)
        mlo, mhi = an.band("final_od", "mrss")
        self.assertAlmostEqual((mhi - mlo) / 2, min(1.5 * rss, wc), places=12)

    def test_linear_wc_equals_extreme_for_linear_stack(self):
        a = Analysis(_simple(Settings(wc_mode="extreme", mc_enabled=False)))
        b = Analysis(_simple(Settings(wc_mode="linear", mc_enabled=False)))
        for ka, kb in zip(a.keys, b.keys):
            for x, y in zip(a.band(ka, "wc"), b.band(kb, "wc")):
                self.assertAlmostEqual(x, y, places=12)

    def test_mc_matches_rss_sigma(self):
        for samp in ("lhs", "random"):
            an = Analysis(_simple(Settings(mc_n=60000, truncate=False,
                                           mc_sampling=samp)))
            sd = float(np.std(an.samples("final_od"), ddof=1))
            self.assertAlmostEqual(sd / an.sigma[an.idx["final_od"]], 1.0,
                                   delta=0.03)

    def test_uniform_distribution_sigma(self):
        an = Analysis(_simple(Settings(mc_n=60000, default_dist="uniform")))
        sd = float(np.std(an.samples("final_od"), ddof=1))
        expect = math.sqrt(0.0003 ** 2 + 0.0004 ** 2 + 0.0004 ** 2) / \
            math.sqrt(3)
        self.assertAlmostEqual(sd / expect, 1.0, delta=0.03)

    def test_asymmetric_tolerance_shifts_centre(self):
        d = Design("a", [MandrelLayer(od=Dim(0.0200, 0.0002, 0.0))])
        an = Analysis(d)
        lo, hi = an.band("lumen", "wc")
        self.assertAlmostEqual(lo, 0.0200, places=12)
        self.assertAlmostEqual(hi, 0.0202, places=12)
        self.assertAlmostEqual(an.center[an.idx["lumen"]], 0.0201, places=12)
        self.assertAlmostEqual(float(np.mean(an.samples("lumen"))), 0.0201,
                               places=6)

    def test_capability_and_yield(self):
        an = Analysis(_simple(Settings(mc_n=60000, truncate=False)))
        usl = an.nom("final_od") + an.rss_half[an.idx["final_od"]]  # +3σ
        cap = an.capability("final_od", None, usl)
        self.assertAlmostEqual(cap["cpk"], 1.0, delta=0.04)
        self.assertAlmostEqual(cap["yield"], 0.99865, delta=0.0006)

    def test_reflow_attenuates_upstream_variation(self):
        d, _ = _ex()
        an = Analysis(d)
        row = an.S[an.idx["final_od"]]
        j = [i for i, dd in enumerate(an.dims) if dd.id.endswith(".od")
             and "Mandrel" in dd.label][0]
        self.assertLess(row[j], 1.0)            # D_sub/D_out < 1
        self.assertGreater(row[j], 0.0)


class TestIO(unittest.TestCase):
    def test_json_roundtrip_all_presets(self):
        for name, fn in presets.PRESETS.items():
            d = fn()
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, "d.json")
                d.save(p)
                d2 = Design.load(p)
            self.assertEqual(len(d.layers), len(d2.layers), name)
            a = float(evaluate(d).metrics["final_od"])
            b = float(evaluate(d2).metrics["final_od"])
            self.assertAlmostEqual(a, b, places=14, msg=name)

    def test_errors(self):
        with self.assertRaises(ValueError):
            evaluate(Design("x", [LinerLayer()]))           # no mandrel
        d, v = _ex()
        d.layers[4].dims["od"] = Dim.sym(v["mid_id"] + 0.0002)  # tiny annulus
        d.layers[4].dims["pen"] = Dim.sym(0.002)
        with self.assertRaises(ValueError):
            evaluate(d)
        d, _ = _ex()
        d.layers.append(MandrelLayer())
        with self.assertRaises(ValueError):
            evaluate(d)

    def test_all_presets_analyse(self):
        for name, fn in presets.PRESETS.items():
            an = Analysis(fn())
            self.assertIn(an.status(), ("PASS", "WARN", "FAIL"), name)
            self.assertTrue(np.all(np.isfinite(an.nominal)), name)


if __name__ == "__main__":
    unittest.main()
