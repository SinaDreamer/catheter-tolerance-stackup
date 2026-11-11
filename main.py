#!/usr/bin/env python3
"""
Catheter Radial Tolerance Stack-Up Tool — v3
============================================

Layer-configurable radial (diametral) tolerance stack-up for layered,
multi-lumen catheter shafts: mandrel, liners, tie layers, braids, coils,
reflowed and slip-fit tubes, any number of peripheral channels/lumens
(symmetric or not), coatings and heat-shrink process checks.

Methods: Worst-Case (exact corner or linear), RSS, Modified RSS and
Monte Carlo (LHS / random; normal, uniform, triangular), contribution
Pareto, Cp/Cpk (normal or ISO 22514-2 percentile) and PDF/CSV reporting.

Usage
-----
  python main.py                 launch the GUI (PyQt6 + matplotlib)
  python main.py --selftest      run the engine verification suite (no GUI)
  python main.py --report design.json out.pdf [--mm]
                                 headless PDF report for a saved design

The previous v2 single-file tool is kept locally in ``legacy/`` (not published).
"""
import os
import sys


def _selftest() -> int:
    import unittest
    here = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.defaultTestLoader.discover(os.path.join(here, "tests"),
                                                top_level_dir=here)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if res.wasSuccessful():
        from stackup.engine import Analysis
        from stackup.presets import example_4ch
        an = Analysis(example_4ch())
        print("\nSELF-TEST PASSED — example design summary (inches):")
        for m in ("wc", "rss", "mrss", "mc"):
            lo, hi = an.band("final_od", m)
            print(f"  {m:5s} final OD {lo:.4f} – {hi:.4f}")
        print(f"  nominal {an.nom('final_od'):.4f}   status {an.status()}")
        return 0
    return 1


def _headless_report(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from stackup.engine import Analysis
    from stackup.model import Design
    from stackup.report import export_pdf
    src, dst = args[0], args[1]
    units = "mm" if "--mm" in args else "in"
    export_pdf(dst, Analysis(Design.load(src)), units)
    print(f"Report written to {dst}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if "--report" in sys.argv:
        i = sys.argv.index("--report")
        sys.exit(_headless_report(sys.argv[i + 1:]))
    try:
        from stackup.gui import run
    except ImportError as e:
        sys.exit(f"GUI dependencies missing ({e}).\n"
                 f"Install with:  pip install -r requirements.txt")
    sys.exit(run(sys.argv))
