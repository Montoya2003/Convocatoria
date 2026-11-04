"""
Visualizador gráfico: lienzo Matplotlib embebido en Qt (FigureCanvasQTAgg).

Dibuja en un mismo sistema de ejes:
  * la familia de soluciones generales  y = C·x + f(C)  (paleta continua),
  * la solución singular / envolvente (trazo grueso carmesí),
  * los puntos de tangencia (x_t, y_t) = (-f'(C), C·x_t + f(C)),
y permite mostrar/ocultar cada elemento sin recalcular.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt6.QtWidgets import QSizePolicy, QWidget

from app.core.clairaut_solver import ClairautSolution, ClairautSolver, SingularStatus
from app.ui import theme
from app.ui.latex_widget import is_renderable, to_mathtext


@dataclass
class PlotConfig:
    """Parámetros de visualización seleccionados en la interfaz."""

    x_min: float
    x_max: float
    c_values: list[float]
    y_auto: bool = True
    y_min: float = -10.0
    y_max: float = 10.0
    samples: int = 2400


@dataclass
class PlotReport:
    """Resumen de lo graficado (para informar al usuario)."""

    drawn_c: list[float] = field(default_factory=list)
    skipped_c: list[tuple[float, str]] = field(default_factory=list)
    envelope_drawn: bool = False
    envelope_mode: str = ""           # "explícita" | "paramétrica" | ""
    colors: dict[float, str] = field(default_factory=dict)
    y_limits: tuple[float, float] = (0.0, 0.0)


def _break_jumps(ys: np.ndarray, threshold: float) -> np.ndarray:
    """Inserta NaN donde hay saltos verticales enormes (asíntotas) para no
    unir con una línea vertical ramas separadas por una discontinuidad."""
    out = ys.copy()
    if out.size < 3:
        return out
    jumps = np.abs(np.diff(out)) > threshold
    out[1:][jumps] = np.nan
    return out


class PlotCanvas(FigureCanvasQTAgg):
    """Lienzo interactivo con estado de visibilidad por elemento."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor=theme.PLOT_BG, layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)

        # Artistas y estado de visibilidad
        self._general: dict[float, Line2D] = {}
        self._tangents: dict[float, Line2D] = {}
        self._singular: list[Line2D] = []
        self._glow: list[Line2D] = []
        self._hidden_c: set[float] = set()
        self.show_family: bool = True
        self.show_singular: bool = True
        self.show_tangents: bool = True
        self.show_legend: bool = True

        self.show_placeholder("Introduce f(p) y pulsa «Calcular y Graficar»")

    # ------------------------------------------------------------ estilos
    def _style_axes(self) -> None:
        ax = self.ax
        ax.set_facecolor(theme.PLOT_BG)
        for spine in ax.spines.values():
            spine.set_color(theme.PLOT_GRID)
        ax.tick_params(colors=theme.TEXT_MUTED, labelsize=9)
        ax.grid(True, color=theme.PLOT_GRID, alpha=0.45, linestyle="--", linewidth=0.6)
        ax.set_xlabel("x", color=theme.TEXT, fontsize=12)
        ax.set_ylabel("y", color=theme.TEXT, fontsize=12, rotation=0, labelpad=12)
        # Ejes coordenados cruzando por el origen (tono sutil)
        ax.axhline(0.0, color=theme.PLOT_AXES, linewidth=1.1, zorder=1)
        ax.axvline(0.0, color=theme.PLOT_AXES, linewidth=1.1, zorder=1)

    def show_placeholder(self, message: str) -> None:
        """Estado vacío o de error del lienzo."""
        self.ax.clear()
        self._reset_artists()
        self._style_axes()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.text(0.5, 0.5, message, transform=self.ax.transAxes, ha="center", va="center",
                     color=theme.TEXT_MUTED, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.6", fc=theme.SURFACE, ec=theme.PLOT_GRID))
        self.ax.set_title("ClairautDiffViewer", color=theme.TEXT, fontsize=13)
        self.draw_idle()

    def _reset_artists(self) -> None:
        self._general.clear()
        self._tangents.clear()
        self._singular.clear()
        self._glow.clear()

    # ------------------------------------------------------------ graficación
    def plot_solution(self, solver: ClairautSolver, solution: ClairautSolution, cfg: PlotConfig) -> PlotReport:
        """Dibuja la familia general, la envolvente y los puntos de tangencia."""
        report = PlotReport()
        ax = self.ax
        ax.clear()
        self._reset_artists()
        self._style_axes()

        xs = np.linspace(cfg.x_min, cfg.x_max, cfg.samples)
        span_x = cfg.x_max - cfg.x_min

        # --- a) Familia general ------------------------------------------------
        curves = solver.evaluate_general(cfg.c_values, xs)
        cmap = mpl.colormaps[theme.COLORMAP]
        n = max(len(curves) - 1, 1)
        line_samples: list[np.ndarray] = []
        tangent_ys: list[float] = []
        for i, curve in enumerate(curves):
            color = mpl.colors.to_hex(cmap(0.15 + 0.85 * i / n))
            report.colors[curve.c_value] = color
            if curve.y is None:
                report.skipped_c.append((curve.c_value, curve.reason))
                continue
            (line,) = ax.plot(xs, curve.y, color=color, alpha=0.65, linewidth=1.3,
                              label=f"C = {curve.c_value:g}", zorder=2)
            self._general[curve.c_value] = line
            line_samples.append(curve.y)
            report.drawn_c.append(curve.c_value)
            if curve.tangent_point is not None:
                xt, yt = curve.tangent_point
                (pt,) = ax.plot([xt], [yt], "o", color=color, markersize=6,
                                markeredgecolor="white", markeredgewidth=0.9, zorder=6)
                self._tangents[curve.c_value] = pt
                if cfg.x_min <= xt <= cfg.x_max:
                    tangent_ys.append(yt)

        # --- b) Solución singular -----------------------------------------------
        env_segments: list[tuple[np.ndarray, np.ndarray]] = []
        if solution.status is SingularStatus.EXPLICIT:
            for ys in solver.evaluate_singular(xs):
                if np.isfinite(ys).any():
                    env_segments.append((xs, ys))
            report.envelope_mode = "explícita"
        if solution.status is SingularStatus.PARAMETRIC or (
            solution.status is SingularStatus.EXPLICIT and not env_segments
        ):
            env_segments = solver.evaluate_parametric(cfg.x_min, cfg.x_max)
            report.envelope_mode = "paramétrica"

        # --- Límites del eje Y (robustos frente a asíntotas) -------------------
        y_lo, y_hi = self._compute_ylim(cfg, env_segments, line_samples, tangent_ys)
        report.y_limits = (y_lo, y_hi)
        threshold = 0.6 * (y_hi - y_lo)

        for k, (ex, ey) in enumerate(env_segments):
            ey = _break_jumps(ey, threshold)
            label = "Solución singular (envolvente)" if k == 0 else None
            (glow,) = ax.plot(ex, ey, color=theme.ENVELOPE, linewidth=7, alpha=0.15,
                              solid_capstyle="round", zorder=7)
            (env,) = ax.plot(ex, ey, color=theme.ENVELOPE, linewidth=2.5, label=label,
                             solid_capstyle="round", zorder=8)
            self._glow.append(glow)
            self._singular.append(env)
        report.envelope_drawn = bool(self._singular)

        if solution.status is SingularStatus.NONE:
            self._annotate("Sin solución singular: f''(p) ≡ 0 (haz de rectas concurrentes)")
        elif not report.envelope_drawn:
            self._annotate("La envolvente no tiene puntos reales en el rango de x elegido")

        ax.set_xlim(cfg.x_min, cfg.x_max)
        ax.set_ylim(y_lo, y_hi)
        self._set_title(solution)
        self._apply_visibility()
        self.draw_idle()
        # Se guarda la vista para que "Home" de la toolbar regrese aquí.
        if self.toolbar is not None:
            self.toolbar.update()
        return report

    def _compute_ylim(
        self,
        cfg: PlotConfig,
        env_segments: list[tuple[np.ndarray, np.ndarray]],
        line_samples: list[np.ndarray],
        tangent_ys: list[float],
    ) -> tuple[float, float]:
        if not cfg.y_auto:
            return float(cfg.y_min), float(cfg.y_max)
        pieces: list[np.ndarray] = []
        env_vals = np.concatenate([ey[np.isfinite(ey)] for _, ey in env_segments]) if env_segments else np.array([])
        if env_vals.size:
            pieces.append(np.nanpercentile(env_vals, [1, 99]))
        if line_samples:
            all_lines = np.concatenate(line_samples)
            all_lines = all_lines[np.isfinite(all_lines)]
            if all_lines.size:
                q = [10, 90] if env_vals.size else [2, 98]
                pieces.append(np.nanpercentile(all_lines, q))
        if tangent_ys:
            pieces.append(np.asarray(tangent_ys))
        if not pieces:
            return -5.0, 5.0
        vals = np.concatenate(pieces)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return -5.0, 5.0
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.10 * (hi - lo)
        return lo - pad, hi + pad

    def _annotate(self, text: str) -> None:
        self.ax.text(0.98, 0.02, text, transform=self.ax.transAxes, ha="right", va="bottom",
                     color=theme.WARNING, fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.4", fc=theme.SURFACE, ec=theme.WARNING, alpha=0.9),
                     zorder=10)

    def _set_title(self, solution: ClairautSolution) -> None:
        latex_title = r"y = x\,p + " + ClairautSolver.wrap_latex(solution.latex_f)
        if is_renderable(latex_title):
            title = f"Ecuación de Clairaut:  ${to_mathtext(latex_title)}$"
        else:
            title = f"Ecuación de Clairaut:  y = x·p + {solution.f_text}"
        self.ax.set_title(title, color=theme.TEXT, fontsize=13, pad=10)

    # ------------------------------------------------------------ visibilidad
    def set_family_visible(self, visible: bool) -> None:
        self.show_family = visible
        self._apply_visibility()

    def set_singular_visible(self, visible: bool) -> None:
        self.show_singular = visible
        self._apply_visibility()

    def set_tangents_visible(self, visible: bool) -> None:
        self.show_tangents = visible
        self._apply_visibility()

    def set_legend_visible(self, visible: bool) -> None:
        self.show_legend = visible
        self._apply_visibility()

    def set_curve_visible(self, c_value: float, visible: bool) -> None:
        if visible:
            self._hidden_c.discard(c_value)
        else:
            self._hidden_c.add(c_value)
        self._apply_visibility()

    def is_curve_visible(self, c_value: float) -> bool:
        return c_value not in self._hidden_c

    def set_curves_visible(self, c_values: list[float], visible: bool) -> None:
        """Muestra/oculta de una vez un conjunto de rectas de la familia."""
        if visible:
            self._hidden_c.difference_update(c_values)
        else:
            self._hidden_c.update(c_values)
        self._apply_visibility()

    def reset_curve_visibility(self) -> None:
        """Vuelve a mostrar todas las rectas (al calcular una nueva f(p))."""
        self._hidden_c.clear()

    def _apply_visibility(self) -> None:
        for c, line in self._general.items():
            line.set_visible(self.show_family and c not in self._hidden_c)
        for c, pt in self._tangents.items():
            pt.set_visible(self.show_tangents and self.show_family and c not in self._hidden_c)
        for artist in self._singular + self._glow:
            artist.set_visible(self.show_singular)
        self._rebuild_legend()
        self.draw_idle()

    def _rebuild_legend(self) -> None:
        old = self.ax.get_legend()
        if old is not None:
            old.remove()
        if not self.show_legend:
            return
        handles: list[Line2D] = [ln for ln in self._general.values() if ln.get_visible()]
        if self._singular and self.show_singular:
            handles.append(self._singular[0])
        if any(pt.get_visible() for pt in self._tangents.values()):
            handles.append(Line2D([], [], marker="o", linestyle="none", color=theme.TEXT_MUTED,
                                  markeredgecolor="white", label="Puntos de tangencia"))
        if not handles:
            return
        ncol = int(np.ceil(len(handles) / 14))
        legend = self.ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=ncol,
                                framealpha=0.85, facecolor=theme.SURFACE, edgecolor=theme.PLOT_GRID,
                                labelcolor=theme.TEXT)
        legend.set_zorder(20)
