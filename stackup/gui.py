"""PyQt6 desktop application."""
from __future__ import annotations

import os
import sys
import traceback
from typing import Callable, List, Optional

from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import (QAction, QActionGroup, QColor, QIcon, QKeySequence,
                         QPalette, QPixmap, QFont)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout,
    QWidget, QAbstractItemView)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as Canvas,
                                               NavigationToolbar2QT as Toolbar)
from matplotlib.figure import Figure
from matplotlib.colors import to_hex

from . import __version__, plots as P, presets, report
from .engine import Analysis, METHOD_LABELS
from .model import (Channel, ChannelLayer, Design, Dim, DIST_CHOICES,
                    LAYER_TYPES, Layer)
from .units import (MM_PER_IN, french, from_disp, od_from_french, to_disp,
                    unit_name)

APP_NAME = "Catheter Tolerance Stack-Up"
ALIGN_R = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _fmt(v: float, kind: str, units: str) -> str:
    if kind == "len":
        return f"{to_disp(v, units):.6g}"
    return f"{v:.6g}"


def _rfmt(v: float, kind: str, units: str) -> str:
    """Fixed-decimal formatting for results."""
    if kind == "len":
        return f"{to_disp(v, units):.4f}" if units == "mm" else f"{v:.5f}"
    if kind == "ratio":
        return f"{v:.3f}"
    return f"{v:.2f}"


def _parse(txt: str) -> float:
    return float(txt.strip().replace(",", "."))


def _color_icon(color) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(QColor(to_hex(color)))
    return QIcon(pm)


class LenEdit(QLineEdit):
    """Line edit holding a length in inches, shown in display units."""

    def __init__(self, value_in: float, units: str, on_change: Callable,
                 tip: str = ""):
        super().__init__()
        self.units = units
        self._v = value_in
        self.setText(_fmt(value_in, "len", units))
        self.setToolTip(tip)
        self._cb = on_change
        self.editingFinished.connect(self._done)

    def _done(self):
        try:
            v = from_disp(_parse(self.text()), self.units)
        except ValueError:
            self.setText(_fmt(self._v, "len", self.units))
            return
        if v != self._v:
            self._v = v
            self._cb(v)

    def value(self) -> float:
        return self._v


# ---------------------------------------------------------------------------
# dimension table for one layer
# ---------------------------------------------------------------------------
class DimTable(QTableWidget):
    COLS = ["Parameter", "Nominal", "+Tol", "−Tol", "MC distribution"]

    def __init__(self, layer: Layer, units: str, on_change: Callable):
        super().__init__(0, len(self.COLS))
        self.layer, self.units, self.on_change = layer, units, on_change
        u = unit_name(units)
        self.setHorizontalHeaderLabels(
            ["Parameter", f"Nominal ({u})", f"+Tol ({u})", f"−Tol ({u})",
             "MC distribution"])
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setToolTip("Editing +Tol of a symmetric tolerance mirrors it to "
                        "−Tol. Angles are in degrees.")
        self._loading = True
        self.params = layer.visible_params()
        self.setRowCount(len(self.params))
        for r, p in enumerate(self.params):
            d = layer.dims[p.key]
            it = QTableWidgetItem(p.label + ("  [°]" if p.kind == "angle"
                                             else ""))
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            it.setToolTip(p.tip)
            self.setItem(r, 0, it)
            self._set(r, 1, _fmt(d.nom, p.kind, units), True)
            self._set(r, 2, _fmt(d.tp, p.kind, units) if p.toleranced else "—",
                      p.toleranced)
            self._set(r, 3, _fmt(d.tm, p.kind, units) if p.toleranced else "—",
                      p.toleranced)
            if p.toleranced:
                cb = QComboBox()
                cb.addItems(DIST_CHOICES)
                cb.setCurrentText(d.dist)
                cb.currentTextChanged.connect(
                    lambda t, key=p.key: self._dist(key, t))
                self.setCellWidget(r, 4, cb)
            else:
                self._set(r, 4, "", False)
        self._loading = False
        self.cellChanged.connect(self._changed)
        self.setMinimumHeight(self.sizeHintForRow(0) * (len(self.params) + 1)
                              + 8 if self.params else 40)
        self.setMaximumHeight(self.minimumHeight() + 6)

    def _set(self, r, c, txt, editable):
        it = QTableWidgetItem(txt)
        it.setTextAlignment(ALIGN_R)
        if not editable:
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.setItem(r, c, it)

    def _dist(self, key, t):
        self.layer.dims[key].dist = t
        self.on_change()

    def _changed(self, r, c):
        if self._loading or c not in (1, 2, 3):
            return
        p = self.params[r]
        d = self.layer.dims[p.key]
        try:
            v = _parse(self.item(r, c).text())
            if p.kind == "len":
                v = from_disp(v, self.units)
            if c in (2, 3) and v < 0:
                raise ValueError
        except ValueError:
            self._loading = True
            old = (d.nom, d.tp, d.tm)[c - 1]
            self.item(r, c).setText(_fmt(old, p.kind, self.units))
            self._loading = False
            return
        if c == 1:
            d.nom = v
        elif c == 2:
            sym = abs(d.tp - d.tm) < 1e-15
            d.tp = v
            if sym:
                d.tm = v
                self._loading = True
                self.item(r, 3).setText(_fmt(v, p.kind, self.units))
                self._loading = False
        else:
            d.tm = v
        self.on_change()


# ---------------------------------------------------------------------------
# channel editor
# ---------------------------------------------------------------------------
class ChannelEditor(QWidget):
    COLS = ["Label", "Type", "ID", "± ID", "Wall", "± Wall", "Angle °",
            "± Angle", "Radius", "± Radius"]

    def __init__(self, layer: ChannelLayer, units: str, on_change: Callable):
        super().__init__()
        self.layer, self.units, self.on_change = layer, units, on_change
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        g = QGroupBox("Generate equally spaced channels (replaces the list)")
        gl = QGridLayout(g)
        first = layer.channels[0] if layer.channels else Channel("C1")
        self.g_n = QSpinBox()
        self.g_n.setRange(1, 24)
        self.g_n.setValue(max(len(layer.channels), 1))
        self.g_start = QDoubleSpinBox()
        self.g_start.setRange(-360, 360)
        self.g_start.setValue(first.angle.nom)
        self.g_kind = QComboBox()
        self.g_kind.addItems(["lined", "bare"])
        self.g_kind.setCurrentText(first.kind)
        noop = lambda _v: None
        self.g_id = LenEdit(first.id.nom, units, noop)
        self.g_idt = LenEdit(first.id.tp, units, noop)
        self.g_w = LenEdit(first.wall.nom, units, noop)
        self.g_wt = LenEdit(first.wall.tp, units, noop)
        self.g_prefix = QLineEdit("".join(ch for ch in first.label
                                          if not ch.isdigit()) or "C")
        u = unit_name(units)
        for i, (lab, w) in enumerate((
                ("Count", self.g_n), ("Start angle °", self.g_start),
                ("Type", self.g_kind), ("Label prefix", self.g_prefix),
                (f"ID ({u})", self.g_id), (f"± ID", self.g_idt),
                (f"Wall ({u})", self.g_w), (f"± Wall", self.g_wt))):
            gl.addWidget(QLabel(lab), i // 4 * 2, i % 4)
            gl.addWidget(w, i // 4 * 2 + 1, i % 4)
        b = QPushButton("Generate")
        b.clicked.connect(self._generate)
        gl.addWidget(b, 4, 3)
        v.addWidget(g)

        self.tbl = QTableWidget(0, len(self.COLS))
        self.tbl.setHorizontalHeaderLabels(
            [c if c in ("Label", "Type") or "Angle" in c else f"{c} ({u})"
             for c in self.COLS])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.setToolTip(
            "Per-channel geometry. Tolerances here are symmetric (±). "
            "'bare' = lumen without a liner tube (wall ignored). Radius = "
            "channel-centre radius, used only in 'manual' radius mode "
            "(e.g. for off-centre working lumens).")
        self.tbl.cellChanged.connect(self._changed)
        v.addWidget(self.tbl)
        h = QHBoxLayout()
        for txt, fn in (("Add channel", self._add),
                        ("Remove selected", self._remove)):
            bt = QPushButton(txt)
            bt.clicked.connect(fn)
            h.addWidget(bt)
        h.addStretch(1)
        v.addLayout(h)
        self._fill()

    def _fill(self):
        self._loading = True
        chs = self.layer.channels
        manual = self.layer.options["radius_mode"] == "manual"
        self.tbl.setRowCount(len(chs))
        for r, c in enumerate(chs):
            self.tbl.setItem(r, 0, QTableWidgetItem(c.label))
            cb = QComboBox()
            cb.addItems(["lined", "bare"])
            cb.setCurrentText(c.kind)
            cb.currentTextChanged.connect(lambda t, row=r: self._kind(row, t))
            self.tbl.setCellWidget(r, 1, cb)
            vals = [(c.id.nom, "len", True), (c.id.tp, "len", True),
                    (c.wall.nom, "len", c.kind == "lined"),
                    (c.wall.tp, "len", c.kind == "lined"),
                    (c.angle.nom, "angle", True), (c.angle.tp, "angle", True),
                    (c.radius.nom, "len", manual),
                    (c.radius.tp, "len", manual)]
            for j, (val, kind, ed) in enumerate(vals):
                it = QTableWidgetItem(_fmt(val, kind, self.units) if ed
                                      else "—")
                it.setTextAlignment(ALIGN_R)
                if not ed:
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.tbl.setItem(r, j + 2, it)
        self.tbl.setMinimumHeight(min(60 + 26 * len(chs), 320))
        self._loading = False

    def _kind(self, row, t):
        self.layer.channels[row].kind = t
        self._fill()
        self.on_change()

    def _changed(self, r, c):
        if self._loading:
            return
        ch = self.layer.channels[r]
        txt = self.tbl.item(r, c).text()
        if c == 0:
            ch.label = txt.strip() or ch.label
            self.on_change()
            return
        attr = ["id", "id", "wall", "wall", "angle", "angle", "radius",
                "radius"][c - 2]
        is_tol = (c - 2) % 2 == 1
        kind = "angle" if attr == "angle" else "len"
        d: Dim = getattr(ch, attr)
        try:
            v = _parse(txt)
            if kind == "len":
                v = from_disp(v, self.units)
            if is_tol and v < 0:
                raise ValueError
        except ValueError:
            self._fill()
            return
        if is_tol:
            d.tp = d.tm = v
        else:
            d.nom = v
        self.on_change()

    def _generate(self):
        n = self.g_n.value()
        self.layer.channels = ChannelLayer.equal_spacing(
            n, self.g_start.value(), self.g_id.value(), self.g_idt.value(),
            self.g_w.value(), self.g_wt.value(), self.g_kind.currentText(),
            prefix=self.g_prefix.text().strip() or "C")
        self._fill()
        self.on_change()

    def _add(self):
        chs = self.layer.channels
        base = chs[-1] if chs else Channel("C0")
        import copy
        c = copy.deepcopy(base)
        c.label = f"C{len(chs) + 1}"
        c.angle.nom = (base.angle.nom + 45.0) % 360.0
        chs.append(c)
        self._fill()
        self.on_change()

    def _remove(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()},
                      reverse=True)
        for r in rows:
            del self.layer.channels[r]
        self._fill()
        self.on_change()


# ---------------------------------------------------------------------------
# layer editor
# ---------------------------------------------------------------------------
class LayerEditor(QWidget):
    def __init__(self, layer: Layer, units: str, on_change: Callable,
                 on_rename: Callable, on_rebuild: Callable):
        super().__init__()
        self.layer = layer
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        hdr = QLabel(f"<b>{layer.LABEL}</b>")
        v.addWidget(hdr)
        form = QFormLayout()
        name = QLineEdit(layer.name)

        def _rename():
            layer.name = name.text().strip() or layer.LABEL
            on_rename()
        name.editingFinished.connect(_rename)
        form.addRow("Name:", name)
        gated = {p.show_if[0] for p in layer.PARAMS if p.show_if}
        for o in layer.OPTIONS:
            val = layer.options[o.key]
            if o.choices:
                w = QComboBox()
                w.addItems(o.choices)
                w.setCurrentText(str(val))
                sig, get = w.currentTextChanged, w.currentText
            elif isinstance(o.default, bool):
                w = QCheckBox()
                w.setChecked(bool(val))
                sig, get = w.toggled, w.isChecked
            elif isinstance(o.default, int):
                w = QSpinBox()
                w.setRange(int(o.minimum), int(o.maximum))
                w.setValue(int(val))
                sig, get = w.valueChanged, w.value
            else:
                w = QDoubleSpinBox()
                w.setDecimals(2)
                w.setRange(o.minimum, o.maximum)
                w.setValue(float(val))
                sig, get = w.valueChanged, w.value
            w.setToolTip(o.tip)

            def _opt(*_a, key=o.key, get=get):
                layer.options[key] = get()
                if key in gated or isinstance(layer, ChannelLayer):
                    QTimer.singleShot(0, on_rebuild)
                on_change()
            sig.connect(_opt)
            form.addRow(o.label + ":", w)
        v.addLayout(form)
        if layer.visible_params():
            v.addWidget(DimTable(layer, units, on_change))
        if isinstance(layer, ChannelLayer):
            v.addWidget(ChannelEditor(layer, units, on_change))
        d = QLabel(layer.DESCRIPTION)
        d.setWordWrap(True)
        d.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(d)
        v.addStretch(1)


# ---------------------------------------------------------------------------
# plot tab
# ---------------------------------------------------------------------------
class PlotTab(QWidget):
    def __init__(self, kind: str, owner: "MainWindow"):
        super().__init__()
        self.kind, self.owner = kind, owner
        self.dirty = True
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.controls = QHBoxLayout()
        self.metric = self.mode = self.band = self.callouts = None
        if kind in ("pareto", "mc"):
            self.controls.addWidget(QLabel("Metric:"))
            self.metric = QComboBox()
            self.metric.setMinimumWidth(260)
            self.metric.currentIndexChanged.connect(self.redraw)
            self.controls.addWidget(self.metric, 1)
        if kind == "pareto":
            self.mode = QComboBox()
            self.mode.addItems(["RSS variance share", "Worst-case share"])
            self.mode.currentIndexChanged.connect(self.redraw)
            self.controls.addWidget(self.mode)
        if kind in ("cross", "hs", "buildup"):
            self.controls.addWidget(QLabel("Band method:"))
            self.band = QComboBox()
            self.band.addItem("Acceptance method", None)
            for m in ("wc", "rss", "mrss", "mc"):
                self.band.addItem(METHOD_LABELS[m], m)
            self.band.currentIndexChanged.connect(self.redraw)
            self.controls.addWidget(self.band)
        if kind == "cross":
            self.callouts = QCheckBox("Callouts")
            self.callouts.setChecked(True)
            self.callouts.toggled.connect(self.redraw)
            self.controls.addWidget(self.callouts)
            self.bands = QCheckBox("Tolerance bands")
            self.bands.setChecked(True)
            self.bands.toggled.connect(self.redraw)
            self.controls.addWidget(self.bands)
        self.controls.addStretch(1)
        v.addLayout(self.controls)
        self.fig = Figure(figsize=(7, 6))
        self.canvas = Canvas(self.fig)
        v.addWidget(Toolbar(self.canvas, self))
        v.addWidget(self.canvas, 1)

    def set_metrics(self, an: Analysis):
        if self.metric is None:
            return
        cur = self.metric.currentData() or "final_od"
        self.metric.blockSignals(True)
        self.metric.clear()
        for k in an.keys:
            self.metric.addItem(an.info[k].label, k)
        i = self.metric.findData(cur)
        self.metric.setCurrentIndex(i if i >= 0 else
                                    self.metric.findData("final_od"))
        self.metric.blockSignals(False)

    def redraw(self, *_):
        an = self.owner.analysis
        if an is None:
            return
        th = P.THEMES[self.owner.theme]
        u = self.owner.units
        band = self.band.currentData() if self.band is not None else None
        if band and band not in an.bands:
            band = None
        try:
            if self.kind == "cross":
                P.draw_cross_section(self.fig, an, th, u, band,
                                     self.bands.isChecked(),
                                     self.callouts.isChecked())
            elif self.kind == "buildup":
                P.draw_buildup(self.fig, an, th, u, band)
            elif self.kind == "cutaway":
                P.draw_cutaway(self.fig, an, th, u)
            elif self.kind == "pareto":
                P.draw_pareto(self.fig, an, th,
                              self.metric.currentData() or "final_od",
                              "rss" if self.mode.currentIndex() == 0 else "wc",
                              u)
            elif self.kind == "mc":
                P.draw_mc(self.fig, an, th,
                          self.metric.currentData() or "final_od", u)
            elif self.kind == "hs":
                P.draw_heatshrink(self.fig, an, th, u, band)
        except Exception as e:                       # keep the UI alive
            P._message(self.fig, th, f"Plot error: {e}")
            traceback.print_exc()
        self.canvas.draw_idle()
        self.dirty = False


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    PLOT_TABS = [("Cross-section", "cross"), ("Build-up", "buildup"),
                 ("Cut-away", "cutaway"), ("Pareto", "pareto"),
                 ("Monte Carlo", "mc"), ("Heat shrink", "hs")]

    def __init__(self):
        super().__init__()
        self.qs = QSettings("CatheterStackUp", "StackUpTool")
        self.theme = self.qs.value("theme", "light")
        self.design: Design = presets.example_4ch()
        self.path: Optional[str] = None
        self.dirty = False
        self.analysis: Optional[Analysis] = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.recalculate)
        self.resize(1600, 940)
        self._build_menus()
        self._build_ui()
        self.apply_theme(self.theme)
        self.load_design(self.design, None)

    @property
    def units(self) -> str:
        return self.design.settings.units

    # ------------------------------------------------------------- menus
    def _build_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        self._act(fm, "&New", self.new_design, QKeySequence.StandardKey.New)
        pm = fm.addMenu("Open &preset")
        for name, fn in presets.PRESETS.items():
            self._act(pm, name, lambda _=False, f=fn: self._open_preset(f))
        self._act(fm, "&Open…", self.open_file, QKeySequence.StandardKey.Open)
        self._act(fm, "&Save", self.save_file, QKeySequence.StandardKey.Save)
        self._act(fm, "Save &as…", self.save_as,
                  QKeySequence.StandardKey.SaveAs)
        fm.addSeparator()
        self._act(fm, "Export PDF &report…", self.export_pdf, "Ctrl+P")
        self._act(fm, "Export results &CSV…", self.export_csv)
        self._act(fm, "Export Monte Carlo samples CSV…", self.export_mc)
        self._act(fm, "Export current plot (PNG/SVG/PDF)…", self.export_plot)
        fm.addSeparator()
        self._act(fm, "&Quit", self.close, QKeySequence.StandardKey.Quit)

        vm = mb.addMenu("&View")
        um = vm.addMenu("Units")
        self.unit_group = QActionGroup(self)
        self.unit_acts = {}
        for u, lab in (("in", "Inches"), ("mm", "Millimetres")):
            a = QAction(lab, self, checkable=True)
            a.triggered.connect(lambda _=False, uu=u: self.set_units(uu))
            self.unit_group.addAction(a)
            um.addAction(a)
            self.unit_acts[u] = a
        tm = vm.addMenu("Theme")
        g = QActionGroup(self)
        self.theme_acts = {}
        for t in ("light", "dark"):
            a = QAction(t.capitalize(), self, checkable=True)
            a.setChecked(t == self.theme)
            a.triggered.connect(lambda _=False, tt=t: self.apply_theme(tt))
            g.addAction(a)
            tm.addAction(a)
            self.theme_acts[t] = a

        hm = mb.addMenu("&Help")
        self._act(hm, "Model reference", self.show_model_help)
        self._act(hm, "About", self.show_about)

    def _act(self, menu, text, fn, shortcut=None):
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    # ------------------------------------------------------------- layout
    def _build_ui(self):
        split = QSplitter()
        self.setCentralWidget(split)

        # ---- left: inputs
        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(self._build_stack_tab(), "Layer stack")
        self.left_tabs.addTab(self._wrap(self._build_analysis_tab()),
                              "Analysis")
        self.left_tabs.addTab(self._wrap(self._build_accept_tab()),
                              "Acceptance && design")
        split.addWidget(self.left_tabs)

        # ---- middle: results
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(2, 2, 2, 2)
        self.banner = QLabel("—")
        self.banner.setWordWrap(True)
        self.banner.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        mv.addWidget(self.banner)
        self.res_tabs = QTabWidget()
        self.tbl_build = self._table()
        self.tbl_checks = self._table()
        self.tbl_all = self._table()
        self.txt_notes = QPlainTextEdit()
        self.txt_notes.setReadOnly(True)
        self.res_tabs.addTab(self.tbl_build, "Build-up")
        self.res_tabs.addTab(self.tbl_checks, "Checks")
        self.res_tabs.addTab(self.tbl_all, "All metrics")
        self.res_tabs.addTab(self.txt_notes, "Notes && warnings")
        mv.addWidget(self.res_tabs, 1)
        split.addWidget(mid)

        # ---- right: plots
        self.plot_tabs = QTabWidget()
        self.plots: List[PlotTab] = []
        for title, kind in self.PLOT_TABS:
            t = PlotTab(kind, self)
            self.plots.append(t)
            self.plot_tabs.addTab(t, title)
        self.plot_tabs.currentChanged.connect(self._plot_tab_changed)
        split.addWidget(self.plot_tabs)
        split.setSizes([470, 470, 660])

    @staticmethod
    def _wrap(w):
        s = QScrollArea()
        s.setWidget(w)
        s.setWidgetResizable(True)
        return s

    @staticmethod
    def _table():
        t = QTableWidget()
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        t.setAlternatingRowColors(True)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        return t

    def _build_stack_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(QLabel("Layers, inside → out (first must be the "
                           "mandrel):"))
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(210)
        self.layer_list.currentRowChanged.connect(self._show_editor)
        v.addWidget(self.layer_list)
        h = QHBoxLayout()
        add = QToolButton()
        add.setText("Add layer ▾")
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(add)
        for t, cls in LAYER_TYPES.items():
            a = menu.addAction(_color_icon(cls.COLOR), cls.LABEL)
            a.triggered.connect(lambda _=False, c=cls: self._add_layer(c))
        add.setMenu(menu)
        h.addWidget(add)
        for txt, fn, tip in (("Duplicate", self._dup_layer, ""),
                             ("Remove", self._del_layer, ""),
                             ("Up", lambda: self._move_layer(-1), "Move layer outward→inward (up)"),
                             ("Down", lambda: self._move_layer(1), "Move layer down (outward)")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
        v.addLayout(h)
        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        v.addWidget(self.editor_scroll, 1)
        return w

    def _build_analysis_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        self.a_method = QComboBox()
        for m in ("wc", "rss", "mrss", "mc"):
            self.a_method.addItem(METHOD_LABELS[m], m)
        self.a_method.setToolTip("Method used to judge every check "
                                 "(pass/fail) and shade the schematics.")
        self.a_wc = QComboBox()
        self.a_wc.addItem("Exact corner evaluation (non-linear)", "extreme")
        self.a_wc.addItem("Linearised Σ|s|·tol", "linear")
        self.a_k = QDoubleSpinBox()
        self.a_k.setRange(1.0, 6.0)
        self.a_k.setSingleStep(0.5)
        self.a_k.setToolTip("Each tolerance zone is treated as ±kσ of its "
                            "process (3 = classic RSS assumption, "
                            "~Cp 1.0).")
        self.a_mrss = QDoubleSpinBox()
        self.a_mrss.setRange(1.0, 3.0)
        self.a_mrss.setSingleStep(0.1)
        self.a_mrss.setToolTip("Bender modified RSS: band = f × RSS, capped "
                               "at the linear worst case (1.5 typical).")
        self.a_reflow = QComboBox()
        self.a_reflow.addItem("Exact area conservation", "exact")
        self.a_reflow.addItem("Thin-wall approximation (legacy v2)",
                              "thin-wall")
        self.a_sub = QCheckBox("Embedded channels displace polymer "
                               "(subtract channel area)")
        f.addRow(QLabel("<b>Stack-up methods</b>"))
        f.addRow("Acceptance method:", self.a_method)
        f.addRow("Worst-case evaluation:", self.a_wc)
        f.addRow("Tolerance = ± k·σ, k:", self.a_k)
        f.addRow("Modified-RSS factor:", self.a_mrss)
        f.addRow(QLabel("<b>Reflow physics</b>"))
        f.addRow("Reflow model:", self.a_reflow)
        f.addRow("", self.a_sub)
        f.addRow(QLabel("<b>Monte Carlo</b>"))
        self.a_mc = QCheckBox("Run Monte Carlo")
        self.a_n = QSpinBox()
        self.a_n.setRange(1000, 1_000_000)
        self.a_n.setSingleStep(5000)
        self.a_samp = QComboBox()
        self.a_samp.addItem("Latin hypercube", "lhs")
        self.a_samp.addItem("Pseudo-random", "random")
        self.a_seed = QSpinBox()
        self.a_seed.setRange(0, 2_000_000_000)
        self.a_dist = QComboBox()
        self.a_dist.addItems(["normal", "uniform", "triangular"])
        self.a_dist.setToolTip("Used by every dimension whose distribution is "
                               "'default'. Per-dimension overrides live in "
                               "the layer tables.")
        self.a_trunc = QCheckBox("Truncate normals at tolerance limits "
                                 "(100 % inspected parts)")
        self.a_cap = QComboBox()
        self.a_cap.addItem("Normal (Cp/Cpk from mean, σ)", "normal")
        self.a_cap.addItem("Percentile (ISO 22514-2, non-normal)",
                           "percentile")
        f.addRow("", self.a_mc)
        f.addRow("Samples N:", self.a_n)
        f.addRow("Sampling:", self.a_samp)
        f.addRow("Random seed:", self.a_seed)
        f.addRow("Default distribution:", self.a_dist)
        f.addRow("", self.a_trunc)
        f.addRow("Capability method:", self.a_cap)
        b = QPushButton("Recalculate now")
        b.clicked.connect(self.recalculate)
        f.addRow("", b)
        note = QLabel(
            "WC: exact min/max of the full non-linear model at the tolerance "
            "corners.\nRSS: centre ± 3σ, σ² = Σ(sᵢ·halfᵢ/k)².\nMRSS: "
            "f × RSS (≤ WC).\nMC: P0.135–P99.865 % of N simulated builds.\n"
            "sᵢ = numerical sensitivity of the metric to dimension i "
            "(includes substrate cross-terms).")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        f.addRow(note)
        for wdg in (self.a_method, self.a_wc, self.a_reflow, self.a_samp,
                    self.a_dist, self.a_cap):
            wdg.currentIndexChanged.connect(self._read_settings)
        for wdg in (self.a_k, self.a_mrss, self.a_n, self.a_seed):
            wdg.valueChanged.connect(self._read_settings)
        for wdg in (self.a_sub, self.a_mc, self.a_trunc):
            wdg.toggled.connect(self._read_settings)
        return w

    def _build_accept_tab(self):
        w = QWidget()
        self.acc_form = QFormLayout(w)
        return w

    def _fill_accept_tab(self):
        f = self.acc_form
        while f.rowCount():
            f.removeRow(0)
        d, a, u = self.design, self.design.acceptance, self.units
        name = QLineEdit(d.name)

        def _n():
            d.name = name.text().strip() or "Untitled design"
            self._touch()
        name.editingFinished.connect(_n)
        f.addRow(QLabel("<b>Design</b>"))
        f.addRow("Name:", name)
        notes = QPlainTextEdit(d.notes)
        notes.setMaximumHeight(90)
        notes.textChanged.connect(lambda: (setattr(d, "notes",
                                                    notes.toPlainText()),
                                           self._mark_dirty()))
        f.addRow("Notes:", notes)
        f.addRow(QLabel("<b>Acceptance limits</b>  (0 = not used)"))

        def len_row(label, attr, tip=""):
            e = LenEdit(getattr(a, attr), u,
                        lambda v, at=attr: (setattr(a, at, v), self._touch()),
                        tip)
            f.addRow(f"{label} ({unit_name(u)}):", e)
            return e
        self.e_usl = len_row("OD ceiling / USL", "od_usl",
                             "e.g. introducer-sheath ID")
        fr = QHBoxLayout()
        sp = QDoubleSpinBox()
        sp.setRange(1, 30)
        sp.setSingleStep(0.5)
        sp.setValue(round(french(a.od_usl), 1) if a.od_usl > 0 else 5.0)
        bt = QPushButton("Set USL from Fr")

        def _fr():
            a.od_usl = od_from_french(sp.value())
            self.e_usl.setText(_fmt(a.od_usl, "len", u))
            self.e_usl._v = a.od_usl
            self._touch()
        bt.clicked.connect(_fr)
        fr.addWidget(sp)
        fr.addWidget(bt)
        f.addRow("French size:", fr)
        len_row("OD minimum / LSL", "od_lsl")
        len_row("Min main-lumen ID", "min_lumen")
        len_row("Min web between channels", "min_web")
        len_row("Min wall channel → OD", "min_cover")
        f.addRow(QLabel("<b>Core eccentricity</b> (lumen/core offset from "
                        "the jacket centre)"))
        e = a.ecc

        def _ecc_nom(v):
            e.nom = v
            e.tm = min(e.tp, e.nom)
            self._touch()

        def _ecc_tol(v):
            e.tp = v
            e.tm = min(v, e.nom)
            self._touch()
        f.addRow(f"Nominal offset ({unit_name(u)}):",
                 LenEdit(e.nom, u, _ecc_nom,
                         "Drawn in the cross-section; reduces channel walls "
                         "to OD"))
        f.addRow(f"+ Offset tolerance ({unit_name(u)}):",
                 LenEdit(e.tp, u, _ecc_tol, "Concentricity variation"))
        hint = QLabel("Off-centre working lumens: add a 'Peripheral channels / "
                      "lumens' layer, set its radius mode to 'manual' and "
                      "use a 'bare' channel at the required radius/angle.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        f.addRow(hint)

    # ------------------------------------------------------------- design io
    def load_design(self, d: Design, path: Optional[str]):
        self.design = d
        self.path = path
        self.dirty = False
        self.unit_acts[d.settings.units].setChecked(True)
        self._fill_settings()
        self._fill_accept_tab()
        self._refresh_list(0)
        self.recalculate()

    def _fill_settings(self):
        s = self.design.settings
        ws = [self.a_method, self.a_wc, self.a_reflow, self.a_samp,
              self.a_dist, self.a_cap, self.a_k, self.a_mrss, self.a_n,
              self.a_seed, self.a_sub, self.a_mc, self.a_trunc]
        for wdg in ws:
            wdg.blockSignals(True)
        self.a_method.setCurrentIndex(self.a_method.findData(s.method))
        self.a_wc.setCurrentIndex(self.a_wc.findData(s.wc_mode))
        self.a_reflow.setCurrentIndex(self.a_reflow.findData(s.reflow_model))
        self.a_samp.setCurrentIndex(self.a_samp.findData(s.mc_sampling))
        self.a_dist.setCurrentText(s.default_dist)
        self.a_cap.setCurrentIndex(self.a_cap.findData(s.cap_method))
        self.a_k.setValue(s.k_sigma)
        self.a_mrss.setValue(s.mrss_factor)
        self.a_n.setValue(s.mc_n)
        self.a_seed.setValue(s.mc_seed)
        self.a_sub.setChecked(s.subtract_channels)
        self.a_mc.setChecked(s.mc_enabled)
        self.a_trunc.setChecked(s.truncate)
        for wdg in ws:
            wdg.blockSignals(False)

    def _read_settings(self, *_):
        s = self.design.settings
        s.method = self.a_method.currentData()
        s.wc_mode = self.a_wc.currentData()
        s.reflow_model = self.a_reflow.currentData()
        s.mc_sampling = self.a_samp.currentData()
        s.default_dist = self.a_dist.currentText()
        s.cap_method = self.a_cap.currentData()
        s.k_sigma = self.a_k.value()
        s.mrss_factor = self.a_mrss.value()
        s.mc_n = self.a_n.value()
        s.mc_seed = self.a_seed.value()
        s.subtract_channels = self.a_sub.isChecked()
        s.mc_enabled = self.a_mc.isChecked()
        s.truncate = self.a_trunc.isChecked()
        self._touch()

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(
            self, APP_NAME, "Discard unsaved changes to the current design?",
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel)
        return r == QMessageBox.StandardButton.Discard

    def new_design(self):
        if self._confirm_discard():
            d = presets.blank()
            d.settings.units = self.units
            self.load_design(d, None)

    def _open_preset(self, fn):
        if self._confirm_discard():
            d = fn()
            d.settings.units = self.units
            self.load_design(d, None)

    def open_file(self):
        if not self._confirm_discard():
            return
        p, _ = QFileDialog.getOpenFileName(self, "Open design",
                                           self.qs.value("dir", ""),
                                           "Stack-up design (*.json)")
        if not p:
            return
        try:
            d = Design.load(p)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Could not open file:\n{e}")
            return
        self.qs.setValue("dir", os.path.dirname(p))
        self.load_design(d, p)

    def save_file(self):
        if not self.path:
            return self.save_as()
        try:
            self.design.save(self.path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Could not save:\n{e}")
            return
        self.dirty = False
        self._title()
        self.statusBar().showMessage(f"Saved {self.path}", 4000)

    def save_as(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save design as", os.path.join(
                self.qs.value("dir", ""), self._safe_name() + ".json"),
            "Stack-up design (*.json)")
        if p:
            self.path = p
            self.qs.setValue("dir", os.path.dirname(p))
            self.save_file()

    def _safe_name(self):
        return "".join(c if c.isalnum() or c in "-_ " else "_"
                       for c in self.design.name).strip() or "design"

    def _ask_path(self, title, ext, filt):
        p, _ = QFileDialog.getSaveFileName(
            self, title, os.path.join(self.qs.value("dir", ""),
                                      self._safe_name() + ext), filt)
        if p:
            self.qs.setValue("dir", os.path.dirname(p))
        return p

    def export_pdf(self):
        if not self._need_analysis():
            return
        p = self._ask_path("Export PDF report", ".pdf", "PDF (*.pdf)")
        if p:
            self._run_export(lambda: report.export_pdf(p, self.analysis,
                                                       self.units), p)

    def export_csv(self):
        if not self._need_analysis():
            return
        p = self._ask_path("Export results CSV", "_results.csv", "CSV (*.csv)")
        if p:
            self._run_export(lambda: report.export_csv(p, self.analysis,
                                                       self.units), p)

    def export_mc(self):
        if not self._need_analysis():
            return
        if self.analysis.mc is None:
            QMessageBox.information(self, APP_NAME,
                                    "Monte Carlo is disabled.")
            return
        p = self._ask_path("Export Monte Carlo samples", "_mc.csv",
                           "CSV (*.csv)")
        if p:
            self._run_export(lambda: report.export_mc_csv(p, self.analysis,
                                                          self.units), p)

    def export_plot(self):
        tab: PlotTab = self.plot_tabs.currentWidget()
        p = self._ask_path("Export plot", f"_{tab.kind}.png",
                           "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if p:
            self._run_export(lambda: tab.fig.savefig(
                p, dpi=200, facecolor=tab.fig.get_facecolor()), p)

    def _run_export(self, fn, p):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fn()
            self.statusBar().showMessage(f"Exported {p}", 5000)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Export failed:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            for t in self.plots:       # report drawing may reuse styles
                t.dirty = True
            self._plot_tab_changed()

    def _need_analysis(self):
        if self.analysis is None:
            QMessageBox.warning(self, APP_NAME, "Fix the input errors first.")
            return False
        return True

    # ------------------------------------------------------------- stack ops
    def _refresh_list(self, select: Optional[int] = None):
        cols = P.layer_colors(self.design)
        row = self.layer_list.currentRow() if select is None else select
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for i, L in enumerate(self.design.layers):
            tag = "" if L.STRUCTURAL else "  (process / channels)"
            it = QListWidgetItem(_color_icon(cols[L.uid]),
                                 f"{i + 1}. {L.name}   — {L.LABEL}{tag}")
            self.layer_list.addItem(it)
        self.layer_list.blockSignals(False)
        row = max(0, min(row, len(self.design.layers) - 1))
        self.layer_list.setCurrentRow(row)
        self._show_editor(row)

    def _show_editor(self, row):
        if row < 0 or row >= len(self.design.layers):
            self.editor_scroll.setWidget(QWidget())
            return
        L = self.design.layers[row]
        ed = LayerEditor(L, self.units, self._touch,
                         lambda: (self._refresh_list(), self._touch()),
                         lambda: self._show_editor(
                             self.layer_list.currentRow()))
        self.editor_scroll.setWidget(ed)

    def _add_layer(self, cls):
        layers = self.design.layers
        if cls.TYPE == "mandrel" and any(L.TYPE == "mandrel" for L in layers):
            QMessageBox.information(self, APP_NAME,
                                    "The stack already has a mandrel.")
            return
        L = cls()
        row = self.layer_list.currentRow()
        if cls.TYPE == "mandrel":
            layers.insert(0, L)
            row = 0
        else:
            row = (row + 1) if row >= 0 else len(layers)
            layers.insert(row, L)
        self._refresh_list(row)
        self._touch()

    def _dup_layer(self):
        row = self.layer_list.currentRow()
        if row < 0:
            return
        L = self.design.layers[row]
        if L.TYPE == "mandrel":
            return
        self.design.layers.insert(row + 1, L.clone())
        self._refresh_list(row + 1)
        self._touch()

    def _del_layer(self):
        row = self.layer_list.currentRow()
        if row < 0:
            return
        del self.design.layers[row]
        self._refresh_list(row - 1 if row else 0)
        self._touch()

    def _move_layer(self, step):
        row = self.layer_list.currentRow()
        j = row + step
        layers = self.design.layers
        if row < 0 or j < 0 or j >= len(layers):
            return
        layers[row], layers[j] = layers[j], layers[row]
        self._refresh_list(j)
        self._touch()

    # ------------------------------------------------------------- settings
    def set_units(self, u):
        if u == self.units:
            return
        self.design.settings.units = u
        self._fill_accept_tab()
        self._show_editor(self.layer_list.currentRow())
        self._touch()

    def apply_theme(self, t):
        self.theme = t
        self.qs.setValue("theme", t)
        app = QApplication.instance()
        app.setStyle("Fusion")
        pal = QPalette()
        if t == "dark":
            c = {QPalette.ColorRole.Window: "#262b33",
                 QPalette.ColorRole.WindowText: "#e6e9ee",
                 QPalette.ColorRole.Base: "#1e2228",
                 QPalette.ColorRole.AlternateBase: "#2b313a",
                 QPalette.ColorRole.Text: "#e6e9ee",
                 QPalette.ColorRole.Button: "#313843",
                 QPalette.ColorRole.ButtonText: "#e6e9ee",
                 QPalette.ColorRole.Highlight: "#2f7f9e",
                 QPalette.ColorRole.HighlightedText: "#ffffff",
                 QPalette.ColorRole.ToolTipBase: "#313843",
                 QPalette.ColorRole.ToolTipText: "#e6e9ee",
                 QPalette.ColorRole.PlaceholderText: "#8a94a3"}
            for role, col in c.items():
                pal.setColor(role, QColor(col))
            pal.setColor(QPalette.ColorGroup.Disabled,
                         QPalette.ColorRole.Text, QColor("#6b7380"))
            pal.setColor(QPalette.ColorGroup.Disabled,
                         QPalette.ColorRole.ButtonText, QColor("#6b7380"))
        else:
            pal = app.style().standardPalette()
        app.setPalette(pal)
        if hasattr(self, "theme_acts"):
            self.theme_acts[t].setChecked(True)
        for tab in getattr(self, "plots", []):
            tab.dirty = True
        if hasattr(self, "plot_tabs"):
            self._plot_tab_changed()
            if self.analysis is not None:
                self._update_banner(self.analysis)

    # ------------------------------------------------------------- compute
    def _touch(self, *_):
        self._mark_dirty()
        self._timer.start()

    def _mark_dirty(self):
        self.dirty = True
        self._title()

    def _title(self):
        nm = os.path.basename(self.path) if self.path else "unsaved"
        self.setWindowTitle(f"{APP_NAME} v{__version__} — {self.design.name}"
                            f"  [{nm}{' *' if self.dirty else ''}]")

    def recalculate(self):
        self._timer.stop()
        self._title()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            an = Analysis(self.design)
        except ValueError as e:
            self._error(str(e))
            return
        except Exception as e:                       # unexpected → visible
            traceback.print_exc()
            self._error(f"Unexpected error: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.analysis = an
        self._update_banner(an)
        self._fill_tables(an)
        for t in self.plots:
            t.set_metrics(an)
            t.dirty = True
        self._plot_tab_changed()
        mc = f" · MC N={an.settings.mc_n:,}" if an.mc is not None else ""
        self.statusBar().showMessage(
            f"Computed in {an.elapsed * 1000:.0f} ms · {len(an.dims)} "
            f"toleranced dimensions · {len(an.keys)} metrics{mc}")

    def _error(self, msg):
        self.banner.setText(f"<b>INPUT ERROR</b> — {msg}")
        self.banner.setStyleSheet("QLabel { background:#7a1f1f; color:white; "
                                  "padding:8px; border-radius:4px; }")

    def _plot_tab_changed(self, *_):
        t = self.plot_tabs.currentWidget()
        if t is not None and t.dirty:
            t.redraw()

    def _update_banner(self, an: Analysis):
        st = an.status()
        bg = {"PASS": "#14532d", "WARN": "#7a5a12", "FAIL": "#7a1f1f"}[st]
        u = self.units
        k = MM_PER_IN if u == "mm" else 1.0
        un = unit_name(u)
        fin = an.nom("final_od")
        lines = [f"<b>{st}</b> — judged on <b>{METHOD_LABELS[an.method]}</b>"
                 f" &nbsp; | &nbsp; final OD nominal <b>{fin * k:.4f} {un}</b>"
                 f" ({fin * MM_PER_IN:.3f} mm, {french(fin):.2f} Fr)"]
        for m in ("wc", "rss", "mrss", "mc"):
            if m in an.bands:
                lo, hi = an.band("final_od", m)
                lines.append(f"{METHOD_LABELS[m]:<14}: {lo * k:.4f} – "
                             f"{hi * k:.4f} {un}  (±{(hi - lo) / 2 * k:.4f})")
        usl = an.design.acceptance.od_usl
        if usl > 0:
            lo, hi = an.band("final_od")
            mg = usl - hi
            lines.append(f"OD ceiling {usl * k:.4f} {un} → margin "
                         f"{mg * k:+.4f} {un} ({mg * MM_PER_IN * 1000:+.0f} µm)")
        nf = sum(1 for c in an.checks if not c.passed and c.severity == "fail")
        nw = sum(1 for c in an.checks if not c.passed and c.severity == "warn")
        lines.append(f"{len(an.checks)} checks: {nf} failed, {nw} warnings")
        self.banner.setText("<br>".join(
            l if i == 0 else f"<span style='font-family:Consolas,monospace'>"
            f"{l.replace(' ', '&nbsp;')}</span>" for i, l in enumerate(lines)))
        self.banner.setStyleSheet(f"QLabel {{ background:{bg}; color:white; "
                                  f"padding:8px; border-radius:4px; }}")

    def _fill_metric_table(self, t: QTableWidget, an: Analysis, keys,
                           names=None):
        u = self.units
        methods = [m for m in ("wc", "rss", "mrss", "mc") if m in an.bands]
        head = ["Metric", "Nominal"]
        for m in methods:
            head += [f"{METHOD_LABELS[m]} min", f"{METHOD_LABELS[m]} max"]
        head.append("Δ nominal" if names else "Unit")
        t.clear()
        t.setColumnCount(len(head))
        t.setHorizontalHeaderLabels(head)
        t.setRowCount(len(keys))
        bold = QFont()
        bold.setBold(True)
        prev = 0.0
        sel_cols = set()
        for j, m in enumerate(methods):
            if m == an.method:
                sel_cols = {2 + 2 * j, 3 + 2 * j}
        for r, key in enumerate(keys):
            info = an.info[key]
            vals = [names[r] if names else info.label,
                    _rfmt(an.nom(key), info.kind, u)]
            for m in methods:
                lo, hi = an.band(key, m)
                vals += [_rfmt(lo, info.kind, u), _rfmt(hi, info.kind, u)]
            if names:
                vals.append(_rfmt(an.nom(key) - prev, "len", u))
                prev = an.nom(key)
            else:
                vals.append(P._ulab(info.kind, u))
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c:
                    it.setTextAlignment(ALIGN_R)
                if c in sel_cols:
                    it.setFont(bold)
                t.setItem(r, c, it)
        t.resizeColumnsToContents()

    def _fill_tables(self, an: Analysis):
        rows = an.layer_rows()
        self._fill_metric_table(self.tbl_build, an, [r["key"] for r in rows],
                                [r["name"] for r in rows])
        self._fill_metric_table(self.tbl_all, an, an.keys)
        # checks
        u = self.units
        t = self.tbl_checks
        head = ["Status", "Check", "Spec", "Nominal",
                f"Bound ({METHOD_LABELS[an.method]})", "Margin", "MC yield",
                "Cpk"]
        t.clear()
        t.setColumnCount(len(head))
        t.setHorizontalHeaderLabels(head)
        order = sorted(an.checks, key=lambda c: (c.passed, c.severity != "fail"))
        t.setRowCount(len(order))
        for r, c in enumerate(order):
            st = "PASS" if c.passed else ("WARN" if c.severity == "warn"
                                          else "FAIL")
            spec = []
            if c.lsl is not None:
                spec.append(f"≥ {_rfmt(c.lsl, c.kind, u)}")
            if c.usl is not None:
                spec.append(f"≤ {_rfmt(c.usl, c.kind, u)}")
            bound = c.lo if (c.lsl is not None and
                             (c.usl is None or c.lo - c.lsl <= c.usl - c.hi)) \
                else c.hi
            vals = [st, c.label, " & ".join(spec), _rfmt(c.nominal, c.kind, u),
                    _rfmt(bound, c.kind, u), _rfmt(c.margin, c.kind, u),
                    f"{100 * c.mc_yield:.3f} %" if c.mc_yield is not None
                    else "—",
                    f"{c.cpk:.2f}" if c.cpk is not None else "—"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j >= 2:
                    it.setTextAlignment(ALIGN_R)
                if j == 0:
                    it.setBackground(QColor({"PASS": "#1f8a4c",
                                             "WARN": "#d68910",
                                             "FAIL": "#c0392b"}[st]))
                    it.setForeground(QColor("white"))
                if j == 7 and c.cpk_src:
                    it.setToolTip(f"Source: {c.cpk_src}")
                t.setItem(r, j, it)
        t.resizeColumnsToContents()
        # notes
        lines = [f"Design: {an.design.name}", ""]
        if an.warnings:
            lines.append("WARNINGS")
            lines += [f"  ! {w}" for w in an.warnings]
            lines.append("")
        if an.mc_invalid:
            lines.append(f"  ! {an.mc_invalid} Monte-Carlo samples produced "
                         f"invalid geometry and were excluded.")
        lines.append("TOP CONTRIBUTORS TO FINAL OD (RSS variance share)")
        for r in an.contributions("final_od")[:8]:
            lines.append(f"  {r['var_pct']:5.1f} %  s={r['sens']:+.3f}  "
                         f"{r['label']}")
        lines.append("")
        lines.append("CHANNEL SETS")
        for cs in an.nom_ctx.channel_sets:
            lines.append(f"  {cs.layer.name}: {len(cs.items)} channels, "
                         f"envelope Ø{_rfmt(cs.env, 'len', u)} "
                         f"{unit_name(u)}, embedded in "
                         f"{cs.host or '— (not covered)'}")
        self.txt_notes.setPlainText("\n".join(lines))

    # ------------------------------------------------------------- help
    def show_model_help(self):
        txt = ["<h3>Layer models</h3>"]
        for cls in LAYER_TYPES.values():
            txt.append(f"<p><b>{cls.LABEL}</b><br>"
                       f"{cls.DESCRIPTION.replace(chr(10), '<br>')}</p>")
        txt.append("<h3>Analysis</h3><p>All methods evaluate the same "
                   "non-linear layer model. Sensitivities are central "
                   "differences, so substrate cross-terms (e.g. reflow "
                   "attenuation of upstream variation) are included.</p>")
        m = QMessageBox(self)
        m.setWindowTitle("Model reference")
        m.setTextFormat(Qt.TextFormat.RichText)
        m.setText("\n".join(txt))
        m.exec()

    def show_about(self):
        QMessageBox.about(
            self, APP_NAME,
            f"<b>{APP_NAME}</b> v{__version__}<br>Radial tolerance stack-up "
            f"for layered multi-lumen catheter shafts.<br><br>Run "
            f"<code>python main.py --selftest</code> for engine "
            f"verification.")

    def closeEvent(self, ev):
        if self._confirm_discard():
            ev.accept()
        else:
            ev.ignore()


def run(argv=None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName(APP_NAME)
    w = MainWindow()
    w.show()
    return app.exec()
