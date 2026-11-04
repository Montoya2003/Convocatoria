"""
Ventana principal de ClairautDiffViewer (capa de presentación / controlador).

Panel izquierdo : casos de estudio, entrada de f(p) con validación en tiempo
                  real, rangos, botón de acción y procedimiento paso a paso.
Panel derecho   : lienzo Matplotlib con barra de navegación y controles de
                  visibilidad (envolvente, familia, tangencias, rectas por C).

El cálculo simbólico se ejecuta en un QThread para que la interfaz nunca se
congele, incluso con expresiones costosas para SymPy.
"""

from __future__ import annotations

from typing import Optional

import sympy as sp
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.core.clairaut_solver import (
    ClairautParseError,
    ClairautSolution,
    ClairautSolver,
    SingularStatus,
    SolutionStep,
    build_c_values,
    latex_y_sing,
)
from app.ui import theme
from app.ui.latex_widget import LatexWidget
from app.ui.plot_canvas import PlotCanvas, PlotConfig, PlotReport
from app.utils.presets import PRESETS, preset_by_text

CUSTOM_LABEL = "✎ Personalizada"


# ============================================================================
# Hilo de trabajo para el cálculo simbólico
# ============================================================================
class SolveWorker(QThread):
    """Ejecuta ClairautSolver.solve() fuera del hilo de la interfaz."""

    succeeded = pyqtSignal(object, object)  # (ClairautSolver, ClairautSolution)
    failed = pyqtSignal(str)

    def __init__(self, f_text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.f_text = f_text

    def run(self) -> None:  # noqa: D401 - API Qt
        try:
            solver = ClairautSolver(self.f_text)
            solution = solver.solve()
            self.succeeded.emit(solver, solution)
        except ClairautParseError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - nunca debe colapsar la app
            self.failed.emit(f"Error inesperado durante el cálculo: {exc}")


# ============================================================================
# Widgets auxiliares
# ============================================================================
def _spin(value: float, lo: float, hi: float, step: float = 0.5, decimals: int = 2) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)  # emite valueChanged al confirmar, no por tecla
    box.setMinimumWidth(80)
    return box


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    label.setWordWrap(True)
    return label


class StepCard(QFrame):
    """Tarjeta visual de un paso del procedimiento matemático."""

    def __init__(self, step: SolutionStep, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StepCard")
        color = theme.STEP_COLORS.get(step.kind, theme.ACCENT)
        self.setStyleSheet(
            f"QFrame#StepCard {{ border-left: 4px solid {color}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel(step.title)
        title.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 10.5pt;")
        title.setWordWrap(True)
        layout.addWidget(title)

        if step.description:
            desc = _muted(step.description)
            desc.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(desc)

        for formula in step.formulas:
            layout.addWidget(LatexWidget(formula, fontsize=13))


# ============================================================================
# Ventana principal
# ============================================================================
class MainWindow(QMainWindow):
    """Controlador principal: conecta la vista con el motor matemático."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ClairautDiffViewer — Ecuaciones Diferenciales de Clairaut")
        self.resize(1480, 900)
        self.setMinimumSize(1100, 700)

        self._solver: Optional[ClairautSolver] = None
        self._solution: Optional[ClairautSolution] = None
        self._worker: Optional[SolveWorker] = None
        self._input_valid: bool = False
        self._updating_c_list: bool = False

        # Temporizadores de "debounce" para validación y re-graficado.
        self._validate_timer = QTimer(self, singleShot=True, interval=250)
        self._validate_timer.timeout.connect(self._validate_input)
        self._replot_timer = QTimer(self, singleShot=True, interval=200)
        self._replot_timer.timeout.connect(self._replot_if_possible)

        self._build_ui()
        self._connect_signals()

        # Arranque con el caso obligatorio f(p) = p².
        self.preset_combo.setCurrentIndex(1)
        self._load_preset(1)
        QTimer.singleShot(150, self._on_calculate)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 1000])
        self.setCentralWidget(splitter)

        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Listo.")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(400)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 6, 10)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)

        # ---------- Controles (desplazables) ----------
        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(0, 0, 6, 0)
        cl.setSpacing(8)

        title = QLabel("ClairautDiffViewer")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Solución analítica y gráfica de ecuaciones de Clairaut")
        subtitle.setObjectName("AppSubtitle")
        cl.addWidget(title)
        cl.addWidget(subtitle)
        cl.addWidget(LatexWidget(r"y = x\,y' + f(y')\quad\Longleftrightarrow\quad y = x\,p + f(p)",
                                 fontsize=14, color=theme.ACCENT_HOVER))

        # Ecuación
        eq_box = QGroupBox("Ecuación")
        eq = QVBoxLayout(eq_box)
        eq.addWidget(_muted("Caso de estudio"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem(CUSTOM_LABEL)
        for preset in PRESETS:
            self.preset_combo.addItem(preset.name)
        eq.addWidget(self.preset_combo)
        self.preset_desc = _muted()
        eq.addWidget(self.preset_desc)

        row = QHBoxLayout()
        f_label = QLabel("f(p) =")
        f_label.setStyleSheet("font-weight: 700; font-size: 11pt;")
        self.f_input = QLineEdit()
        self.f_input.setObjectName("FInput")
        self.f_input.setPlaceholderText("p**2,  sqrt(p),  1/p,  exp(p), ...")
        self.f_input.setClearButtonEnabled(True)
        self.f_input.setToolTip(
            "Expresión en la variable p. Admite + - * / ** ^ y las funciones\n"
            "sqrt, cbrt, exp, log/ln, sin, cos, tan, asin, acos, atan,\n"
            "sinh, cosh, tanh, abs y las constantes pi, e."
        )
        row.addWidget(f_label)
        row.addWidget(self.f_input, 1)
        eq.addLayout(row)
        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        eq.addWidget(self.validation_label)
        self.preview_latex = LatexWidget(fontsize=13)
        eq.addWidget(self.preview_latex)
        cl.addWidget(eq_box)

        # Rangos
        range_box = QGroupBox("Rango de visualización")
        grid = QGridLayout(range_box)
        self.x_min = _spin(-6, -1000, 1000)
        self.x_max = _spin(6, -1000, 1000)
        self.y_auto = QCheckBox("Y automático")
        self.y_auto.setChecked(True)
        self.y_min = _spin(-10, -1e5, 1e5)
        self.y_max = _spin(10, -1e5, 1e5)
        grid.addWidget(QLabel("X mín"), 0, 0)
        grid.addWidget(self.x_min, 0, 1)
        grid.addWidget(QLabel("X máx"), 0, 2)
        grid.addWidget(self.x_max, 0, 3)
        grid.addWidget(self.y_auto, 1, 0, 1, 4)
        grid.addWidget(QLabel("Y mín"), 2, 0)
        grid.addWidget(self.y_min, 2, 1)
        grid.addWidget(QLabel("Y máx"), 2, 2)
        grid.addWidget(self.y_max, 2, 3)
        cl.addWidget(range_box)

        # Valores de C
        c_box = QGroupBox("Familia general  y = Cx + f(C)")
        cg = QGridLayout(c_box)
        self.c_min = _spin(-3, -100, 100, 1)
        self.c_max = _spin(3, -100, 100, 1)
        self.c_step = _spin(1, 0.05, 50, 0.25)
        cg.addWidget(QLabel("C mín"), 0, 0)
        cg.addWidget(self.c_min, 0, 1)
        cg.addWidget(QLabel("C máx"), 0, 2)
        cg.addWidget(self.c_max, 0, 3)
        cg.addWidget(QLabel("Paso"), 1, 0)
        cg.addWidget(self.c_step, 1, 1)
        self.c_reset = QPushButton("Restablecer [-3, 3]")
        cg.addWidget(self.c_reset, 1, 2, 1, 2)
        self.c_summary = _muted()
        cg.addWidget(self.c_summary, 2, 0, 1, 4)
        cl.addWidget(c_box)

        # Acción principal
        self.calc_button = QPushButton("Calcular y Graficar")
        self.calc_button.setObjectName("PrimaryButton")
        self.calc_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.calc_button.setToolTip("Atajo: Enter en el campo f(p) o Ctrl+Enter")
        cl.addWidget(self.calc_button)
        self.feedback_label = QLabel()
        self.feedback_label.setWordWrap(True)
        cl.addWidget(self.feedback_label)
        cl.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls)

        # ---------- Procedimiento matemático ----------
        proc_box = QGroupBox("Procedimiento matemático")
        pl = QVBoxLayout(proc_box)
        pl.setContentsMargins(6, 12, 6, 6)
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 6, 0)
        self.steps_layout.setSpacing(8)
        self.steps_layout.addStretch(1)
        steps_scroll = QScrollArea()
        steps_scroll.setWidgetResizable(True)
        steps_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        steps_scroll.setWidget(self.steps_container)
        self.steps_scroll = steps_scroll
        pl.addWidget(steps_scroll)

        vsplit.addWidget(controls_scroll)
        vsplit.addWidget(proc_box)
        vsplit.setSizes([520, 420])
        outer.addWidget(vsplit)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 10, 10, 10)
        layout.setSpacing(6)

        self.canvas = PlotCanvas(panel)
        self.toolbar = NavigationToolbar2QT(self.canvas, panel)
        self.toolbar.setIconSize(QSize(20, 20))

        top = QHBoxLayout()
        top.addWidget(self.toolbar)
        top.addStretch(1)
        layout.addLayout(top)

        options = QHBoxLayout()
        self.chk_singular = QCheckBox("Solución singular (envolvente)")
        self.chk_family = QCheckBox("Familia general")
        self.chk_tangents = QCheckBox("Puntos de tangencia")
        self.chk_legend = QCheckBox("Leyenda")
        for chk in (self.chk_singular, self.chk_family, self.chk_tangents, self.chk_legend):
            chk.setChecked(True)
            options.addWidget(chk)
        self.chk_singular.setStyleSheet(f"QCheckBox {{ color: {theme.ENVELOPE}; font-weight: 600; }}")
        options.addStretch(1)
        layout.addLayout(options)

        body = QHBoxLayout()
        body.addWidget(self.canvas, 1)

        side = QVBoxLayout()
        side.addWidget(_muted("Rectas por C"))
        self.c_list = QListWidget()
        self.c_list.setFixedWidth(150)
        self.c_list.setToolTip("Marca/desmarca cada recta de la familia individualmente")
        side.addWidget(self.c_list, 1)
        btns = QHBoxLayout()
        self.btn_all = QPushButton("Todas")
        self.btn_none = QPushButton("Ninguna")
        btns.addWidget(self.btn_all)
        btns.addWidget(self.btn_none)
        side.addLayout(btns)
        body.addLayout(side)
        layout.addLayout(body, 1)
        return panel

    def _connect_signals(self) -> None:
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.f_input.textEdited.connect(self._on_text_edited)
        self.f_input.textChanged.connect(lambda _t: self._validate_timer.start())
        self.f_input.returnPressed.connect(self._on_calculate)
        self.calc_button.clicked.connect(self._on_calculate)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._on_calculate)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self._on_calculate)

        for box in (self.x_min, self.x_max, self.y_min, self.y_max, self.c_min, self.c_max, self.c_step):
            box.valueChanged.connect(self._on_range_changed)
        self.y_auto.toggled.connect(self._on_y_auto_toggled)
        self.c_reset.clicked.connect(self._reset_c_range)
        self._on_y_auto_toggled(True)
        self._update_c_summary()

        self.chk_singular.toggled.connect(self.canvas.set_singular_visible)
        self.chk_family.toggled.connect(self._on_family_toggled)
        self.chk_tangents.toggled.connect(self.canvas.set_tangents_visible)
        self.chk_legend.toggled.connect(self.canvas.set_legend_visible)
        self.c_list.itemChanged.connect(self._on_c_item_changed)
        self.btn_all.clicked.connect(lambda: self._set_all_c(True))
        self.btn_none.clicked.connect(lambda: self._set_all_c(False))

    # ------------------------------------------------------------ presets
    def _on_preset_changed(self, index: int) -> None:
        if index <= 0:
            self.preset_desc.setText("Escribe tu propia f(p) en el campo inferior.")
            return
        self._load_preset(index)
        self._on_calculate()

    def _load_preset(self, index: int) -> None:
        preset = PRESETS[index - 1]
        self.preset_desc.setText(f"{preset.description}\nSolución singular esperada: {preset.expected_singular}")
        self.f_input.setText(preset.f_text)
        for box, value in ((self.x_min, preset.x_range[0]), (self.x_max, preset.x_range[1])):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self._validate_input()

    def _on_text_edited(self, text: str) -> None:
        """Al editar a mano, el combo refleja si coincide con algún preset."""
        idx = preset_by_text(text)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(idx + 1 if idx >= 0 else 0)
        self.preset_combo.blockSignals(False)
        if idx < 0:
            self.preset_desc.setText("Expresión personalizada.")

    # ------------------------------------------------------------ validación
    def _validate_input(self) -> None:
        """Validación en tiempo real (sólo parseo, muy rápido)."""
        text = self.f_input.text()
        try:
            expr = ClairautSolver.parse(text)
        except ClairautParseError as exc:
            self._input_valid = False
            self._set_input_state("invalid")
            self.validation_label.setText(f"✖ {exc}")
            self.validation_label.setStyleSheet(f"color: {theme.ERROR};")
            self.preview_latex.set_latex("")
        else:
            self._input_valid = True
            self._set_input_state("valid")
            self.validation_label.setText("✔ Expresión válida")
            self.validation_label.setStyleSheet(f"color: {theme.SUCCESS};")
            self.preview_latex.set_latex(r"y = x\,p + " + ClairautSolver.wrap_latex(sp.latex(expr)))
        self._update_button_state()

    def _set_input_state(self, state: str) -> None:
        self.f_input.setProperty("state", state)
        self.f_input.style().unpolish(self.f_input)
        self.f_input.style().polish(self.f_input)

    def _update_button_state(self) -> None:
        busy = self._worker is not None and self._worker.isRunning()
        self.calc_button.setEnabled(self._input_valid and not busy)
        self.calc_button.setText("Calculando…" if busy else "Calcular y Graficar")

    # ------------------------------------------------------------ rangos
    def _on_y_auto_toggled(self, checked: bool) -> None:
        self.y_min.setEnabled(not checked)
        self.y_max.setEnabled(not checked)
        self._replot_timer.start()

    def _reset_c_range(self) -> None:
        for box, value in ((self.c_min, -3), (self.c_max, 3), (self.c_step, 1)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self._on_range_changed()

    def _on_range_changed(self, *_args: object) -> None:
        self._update_c_summary()
        self._replot_timer.start()

    def _update_c_summary(self) -> None:
        try:
            values = build_c_values(self.c_min.value(), self.c_max.value(), self.c_step.value())
        except ValueError as exc:
            self.c_summary.setText(f"⚠ {exc}")
            self.c_summary.setStyleSheet(f"color: {theme.WARNING};")
            return
        shown = ", ".join(f"{v:g}" for v in values) if len(values) <= 9 else (
            f"{values[0]:g}, {values[1]:g}, …, {values[-1]:g}")
        self.c_summary.setText(f"{len(values)} rectas: C ∈ {{{shown}}}")
        self.c_summary.setStyleSheet("")

    def _read_config(self) -> PlotConfig:
        """Lee y valida los rangos; lanza ValueError con mensaje claro."""
        x_min, x_max = self.x_min.value(), self.x_max.value()
        if x_min >= x_max:
            raise ValueError("X mín debe ser menor que X máx.")
        y_auto = self.y_auto.isChecked()
        y_min, y_max = self.y_min.value(), self.y_max.value()
        if not y_auto and y_min >= y_max:
            raise ValueError("Y mín debe ser menor que Y máx.")
        c_values = build_c_values(self.c_min.value(), self.c_max.value(), self.c_step.value())
        return PlotConfig(x_min, x_max, c_values, y_auto, y_min, y_max)

    # ------------------------------------------------------------ cálculo
    def _on_calculate(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._validate_input()
        if not self._input_valid:
            self._show_feedback(self.validation_label.text(), theme.ERROR)
            return
        try:
            self._read_config()
        except ValueError as exc:
            self._show_feedback(f"⚠ {exc}", theme.WARNING)
            return

        text = self.f_input.text().strip()
        if self._solver is not None and self._solver.f_text == text and self._solution is not None:
            self._replot_if_possible()  # misma f(p): sólo se vuelve a graficar
            return

        self._show_feedback("Resolviendo simbólicamente…", theme.TEXT_MUTED)
        self.statusBar().showMessage(f"Calculando para f(p) = {text} …")
        self._worker = SolveWorker(text, self)
        self._worker.succeeded.connect(self._on_solved)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._update_button_state)
        self._worker.start()
        self._update_button_state()

    def _on_solved(self, solver: ClairautSolver, solution: ClairautSolution) -> None:
        self._solver, self._solution = solver, solution
        self.canvas.reset_curve_visibility()
        self._render_steps(solution)
        self._replot_if_possible()

    def _on_failed(self, message: str) -> None:
        self._solver, self._solution = None, None
        self._show_feedback(f"✖ {message}", theme.ERROR)
        self.statusBar().showMessage("Error en el cálculo.")
        self._clear_steps()
        self._add_card(SolutionStep("Error", message, [], kind="error"))
        self.canvas.show_placeholder("No se pudo calcular: revise f(p)")
        self.c_list.clear()

    def _replot_if_possible(self) -> None:
        if self._solver is None or self._solution is None:
            return
        try:
            cfg = self._read_config()
        except ValueError as exc:
            self._show_feedback(f"⚠ {exc}", theme.WARNING)
            return
        try:
            report = self.canvas.plot_solution(self._solver, self._solution, cfg)
        except Exception as exc:  # noqa: BLE001 - errores numéricos no deben cerrar la app
            self._show_feedback(f"✖ Error al graficar: {exc}", theme.ERROR)
            self.canvas.show_placeholder("Error al graficar")
            return
        self._populate_c_list(report)
        self._report_plot(report)

    def _report_plot(self, report: PlotReport) -> None:
        sol = self._solution
        assert sol is not None
        parts = [f"{len(report.drawn_c)} rectas graficadas"]
        color = theme.SUCCESS
        if report.skipped_c:
            skipped = ", ".join(f"{c:g}" for c, _ in report.skipped_c)
            parts.append(f"omitidas C = {skipped} (f(C) fuera del dominio real)")
            color = theme.WARNING
        if sol.status is SingularStatus.NONE:
            parts.append("sin solución singular")
            color = theme.WARNING
        elif report.envelope_drawn:
            parts.append(f"envolvente {report.envelope_mode}")
        else:
            parts.append("envolvente fuera del rango de x")
            color = theme.WARNING
        self._show_feedback("✔ " + " · ".join(parts), color)
        self.statusBar().showMessage(f"f(p) = {sol.f_text}  ·  " + " · ".join(parts))

    def _show_feedback(self, text: str, color: str) -> None:
        self.feedback_label.setText(text)
        self.feedback_label.setStyleSheet(f"color: {color};")

    # ------------------------------------------------------------ procedimiento
    def _clear_steps(self) -> None:
        while self.steps_layout.count() > 1:
            item = self.steps_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_card(self, step: SolutionStep) -> None:
        self.steps_layout.insertWidget(self.steps_layout.count() - 1, StepCard(step))

    def _render_steps(self, sol: ClairautSolution) -> None:
        self._clear_steps()
        # Tarjeta resumen con los resultados principales.
        if sol.status is SingularStatus.EXPLICIT:
            n = len(sol.branches)
            sing_formulas = [latex_y_sing(i, b.y_expr, n) for i, b in enumerate(sol.branches)]
            kind = "success"
        elif sol.status is SingularStatus.PARAMETRIC:
            sing_formulas = [sol.latex_parametric]
            kind = "warning"
        else:
            sing_formulas = [r"\mathrm{Sin\ envolvente:}\ f''(p) \equiv 0"]
            kind = "warning"
        summary = SolutionStep(
            "Resultado",
            "Solución general (familia de rectas) y solución singular (envolvente).",
            [sol.latex_general] + sing_formulas,
            kind=kind,
        )
        self._add_card(summary)
        for step in sol.steps:
            self._add_card(step)
        for msg in sol.messages:
            self._add_card(SolutionStep("Observación", msg, [], kind="warning"))
        self.steps_scroll.verticalScrollBar().setValue(0)

    # ------------------------------------------------------------ visibilidad
    def _populate_c_list(self, report: PlotReport) -> None:
        self._updating_c_list = True
        self.c_list.clear()
        for c, color in report.colors.items():
            drawn = c in report.drawn_c
            item = QListWidgetItem(f"C = {c:g}" + ("" if drawn else "  (n/d)"))
            item.setIcon(self._color_icon(color if drawn else theme.BORDER))
            if drawn:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if self.canvas.is_curve_visible(c)
                                   else Qt.CheckState.Unchecked)
            else:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                reason = next((r for cc, r in report.skipped_c if cc == c), "")
                item.setToolTip(reason)
            item.setData(Qt.ItemDataRole.UserRole, c)
            self.c_list.addItem(item)
        self._updating_c_list = False

    @staticmethod
    def _color_icon(color: str) -> QIcon:
        pix = QPixmap(14, 14)
        pix.fill(QColor(color))
        return QIcon(pix)

    def _on_c_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating_c_list:
            return
        c = item.data(Qt.ItemDataRole.UserRole)
        self.canvas.set_curve_visible(float(c), item.checkState() == Qt.CheckState.Checked)

    def _set_all_c(self, visible: bool) -> None:
        state = Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked
        values: list[float] = []
        self._updating_c_list = True
        for i in range(self.c_list.count()):
            item = self.c_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)
                values.append(float(item.data(Qt.ItemDataRole.UserRole)))
        self._updating_c_list = False
        self.canvas.set_curves_visible(values, visible)

    def _on_family_toggled(self, checked: bool) -> None:
        self.canvas.set_family_visible(checked)
        self.c_list.setEnabled(checked)

    # ------------------------------------------------------------ cierre
    def closeEvent(self, event) -> None:  # noqa: N802 - API Qt
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().closeEvent(event)


__all__ = ["MainWindow", "SolveWorker"]
