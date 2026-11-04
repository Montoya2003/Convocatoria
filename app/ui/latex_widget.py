"""
Componente Qt para renderizar fórmulas matemáticas a partir de LaTeX.

Se usa el motor *mathtext* de Matplotlib (no requiere instalación de TeX):
la fórmula se rasteriza a PNG transparente con alta resolución (según el
device-pixel-ratio de la pantalla) y se pinta escalada para encajar en el
ancho disponible, manteniendo la nitidez en pantallas HiDPI.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from PyQt6.QtCore import QRectF, QSize
from PyQt6.QtGui import QGuiApplication, QPainter, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget

from app.ui import theme

_MATH_RC = {"mathtext.fontset": "cm", "mathtext.rm": "serif"}
_parser = MathTextParser("path")


def to_mathtext(latex: str) -> str:
    r"""
    Adapta el LaTeX generado por SymPy al subconjunto soportado por mathtext.
    p. ej. ``\operatorname{acos}`` → ``\mathrm{acos}``.
    """
    s = latex
    s = re.sub(r"\\operatorname\{([^{}]*)\}", r"\\mathrm{\1}", s)
    s = s.replace(r"\displaystyle", "").replace(r"\limits", "")
    s = s.replace(r"\text{", r"\mathrm{")
    s = s.replace(r"\Rightarrow", r"\Rightarrow ")
    return s


def is_renderable(latex: str, fontsize: float = 12) -> bool:
    """True si mathtext puede componer la fórmula sin errores."""
    try:
        with mpl.rc_context(_MATH_RC):
            _parser.parse(f"${to_mathtext(latex)}$", dpi=72, prop=FontProperties(size=fontsize))
        return True
    except Exception:  # noqa: BLE001 - cualquier error de parseo => no renderizable
        return False


def render_to_pixmap(
    latex: str, fontsize: float = 13, color: str = theme.TEXT, dpr: float = 2.0, math: bool = True
) -> QPixmap:
    """
    Rasteriza una fórmula (o texto plano si ``math=False``) a QPixmap con
    fondo transparente. El tamaño lógico del pixmap es independiente del DPR.
    """
    text = f"${to_mathtext(latex)}$" if math else latex.replace("$", r"\$")
    with mpl.rc_context(_MATH_RC):
        fig = Figure(figsize=(0.01, 0.01), dpi=100)
        fig.patch.set_alpha(0.0)
        FigureCanvasAgg(fig)
        fig.text(0, 0, text, fontsize=fontsize, color=color)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100 * dpr, transparent=True,
                    bbox_inches="tight", pad_inches=0.04)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue(), "PNG")
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


class LatexWidget(QWidget):
    """
    Muestra una fórmula LaTeX. Si mathtext no la soporta (p. ej. entornos
    ``cases``) se degrada elegantemente a texto plano en lugar de fallar.
    El contenido se reduce proporcionalmente si no cabe en el ancho.
    """

    def __init__(
        self,
        latex: str = "",
        fontsize: float = 13,
        color: str = theme.TEXT,
        center: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._fontsize = fontsize
        self._color = color
        self._center = center
        self._pixmap: Optional[QPixmap] = None
        self._latex = ""
        sp_ = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sp_.setHeightForWidth(True)
        self.setSizePolicy(sp_)
        if latex:
            self.set_latex(latex)

    # -- API pública ---------------------------------------------------------
    def latex(self) -> str:
        return self._latex

    def set_latex(self, latex: str, color: Optional[str] = None) -> None:
        """Renderiza `latex`; vacío limpia el widget."""
        self._latex = latex
        if color:
            self._color = color
        if not latex:
            self._pixmap = None
        else:
            dpr = self._screen_dpr()
            try:
                if not is_renderable(latex, self._fontsize):
                    raise ValueError("mathtext no soporta la fórmula")
                self._pixmap = render_to_pixmap(latex, self._fontsize, self._color, dpr)
            except Exception:  # noqa: BLE001 - degradación a texto plano
                self._pixmap = render_to_pixmap(latex, self._fontsize * 0.8, self._color, dpr, math=False)
            self.setToolTip(latex)
        self.updateGeometry()
        self.update()

    # -- Geometría -----------------------------------------------------------
    def _logical_size(self) -> QSize:
        if self._pixmap is None or self._pixmap.isNull():
            return QSize(0, 0)
        dpr = self._pixmap.devicePixelRatio()
        return QSize(int(self._pixmap.width() / dpr), int(self._pixmap.height() / dpr))

    def sizeHint(self) -> QSize:  # noqa: N802 - API Qt
        return self._logical_size()

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        size = self._logical_size()
        return QSize(min(size.width(), 60), self.heightForWidth(min(size.width(), 60)) if size.width() else 0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        size = self._logical_size()
        if size.width() <= 0:
            return 0
        scale = min(1.0, width / size.width())
        return max(1, int(round(size.height() * scale)))

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self._pixmap is None or self._pixmap.isNull():
            return
        size = self._logical_size()
        scale = min(1.0, self.width() / max(1, size.width()))
        w, h = size.width() * scale, size.height() * scale
        left = (self.width() - w) / 2 if self._center else 0.0
        top = (self.height() - h) / 2
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(QRectF(left, top, w, h), self._pixmap, QRectF(self._pixmap.rect()))
        painter.end()

    @staticmethod
    def _screen_dpr() -> float:
        screen = QGuiApplication.primaryScreen()
        dpr = screen.devicePixelRatio() if screen is not None else 1.0
        # Se renderiza al menos a 2x para que al reducir siga siendo nítida.
        return max(2.0, float(dpr))


__all__ = ["LatexWidget", "render_to_pixmap", "is_renderable", "to_mathtext"]
