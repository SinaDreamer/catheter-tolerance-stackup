# Catheter Radial Tolerance Stack-Up Tool (v3)

Radial (diametral) tolerance stack-up for **layered, multi-lumen catheter
shafts**. Build the shaft from a configurable list of layers, enter
nominal dimensions with (asymmetric) tolerances, and evaluate the stack with
Worst-Case, RSS, Modified RSS and Monte Carlo methods, with process
capability, contribution Pareto, schematics and PDF/CSV reporting.

## Quick start

```bash
pip install -r requirements.txt
python main.py                      # GUI
python main.py --selftest           # engine verification suite (no GUI)
python main.py --report examples/example_4ch.json report.pdf [--mm]
```

Requires Python ≥ 3.9, PyQt6, matplotlib, numpy (scipy optional, speeds up
Monte Carlo).

## Layer library

| Layer | Model (diametral) |
|---|---|
| Mandrel / core (first, required) | main-lumen ID = mandrel OD (line-to-line) |
| Liner, tie layer, coating, generic wall | `D += 2·wall` |
| Braid (round / flat wire) | `D += 2·N·h`, N wires stacked per side at the crossover (default 2); wire width is OD-neutral |
| Coil (round / flat wire) | `D += 2·layers·h` |
| Reflowed polymer tube | area conservation, exact `Dout = √(D² + ODp² − IDp² [+ ΣODch²]) − 2p` or thin-wall (legacy v2) |
| Extruded tube (slip fit) | `Dout = √(max(D, ID)² + OD² − ID²)` (stretches if undersized) + loading-clearance check |
| Peripheral channels / lumens | any count, size, angle, lined (ID + wall) or bare; auto radius (rests on substrate) or manual radius (off-centre lumens); embedded by the next tube |
| Heat shrink (process aid) | not in OD; loading, recovery-drive, no-crush and shrink-ratio checks |

Layers can be added, removed, duplicated and reordered freely (the mandrel
must be first). Reflow model and channel-area displacement can be switched
globally (Analysis tab) or per reflow layer.

## Analysis methods

* **Worst-Case** – exact corner evaluation of the full non-linear model
  (default) or linearised `Σ|sᵢ|·halfᵢ`.
* **RSS** – `centre ± 3σ`, `σ² = Σ(sᵢ·halfᵢ/k)²`, each tolerance = ±kσ
  (k selectable, default 3).
* **Modified RSS** – Bender factor × RSS (default 1.5), capped at worst case.
* **Monte Carlo** – Latin-hypercube or pseudo-random sampling; normal
  (optionally truncated), uniform or triangular per dimension; band =
  P0.135–P99.865 %.
* **Capability** – Cp/Cpk from mean/σ or ISO 22514-2 percentile method;
  predicted yield and ppm.

`sᵢ` are numerical sensitivities of the complete model, so substrate
cross-terms are included (e.g. reflowed layers attenuate upstream variation
by `D_sub/D_out`). Asymmetric tolerances shift the centre of the band.

Checks (OD ceiling/LSL, main-lumen minimum, channel webs, channel wall to
OD, tube loading clearances, channel burial, heat-shrink process window) are
judged with the selected acceptance method.

## Schematics

Cross-section (true scale, tolerance bands, braid wires, channels, scale
bar), radial build-up waterfall, longitudinal cut-away, contribution Pareto,
Monte Carlo histogram with capability, and heat-shrink process window. All
plots support zoom/pan and export (PNG/SVG/PDF), light and dark themes, and
inch/mm units.

## Project layout

```
main.py              entry point (GUI / --selftest / --report)
stackup/model.py     toleranced dimensions, layer library, Design + JSON I/O
stackup/engine.py    evaluation, WC/RSS/MRSS/MC, sensitivities, capability
stackup/plots.py     schematics and charts
stackup/report.py    PDF and CSV export
stackup/gui.py       PyQt6 application
stackup/presets.py   example constructions
tests/               verification suite (incl. cross-check against v2)
examples/            saved designs (JSON)
legacy/              previous v2 version (kept locally, not published)
```

## Verification

`python main.py --selftest` runs hand-calculation checks, statistical
checks (MC σ vs RSS σ, uniform σ, Cpk/yield), I/O round trips, and a
cross-check proving that the thin-wall model without channel-area
subtraction reproduces the v2 nominal geometry exactly. Keep its output as
objective evidence for tool validation (21 CFR 820 / ISO 13485 §4.1.6)
before using results in the DHF.
