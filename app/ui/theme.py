"""
Paleta de colores y hoja de estilos (QSS) del tema oscuro de la aplicación.
Centralizar el tema permite mantener coherencia entre Qt y Matplotlib.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

# --- Paleta base -------------------------------------------------------------
BG = "#0F1117"          # fondo de ventana
SURFACE = "#171A23"     # paneles / tarjetas
SURFACE_2 = "#1F2330"   # entradas
BORDER = "#2C3242"
TEXT = "#E6E8EF"
TEXT_MUTED = "#9AA3B5"
ACCENT = "#4C8DFF"      # azul primario
ACCENT_HOVER = "#6BA1FF"
SUCCESS = "#3DDC97"
WARNING = "#FFB547"
ERROR = "#FF4B4B"

# --- Gráfica -----------------------------------------------------------------
PLOT_BG = "#12151D"
PLOT_AXES = "#5A6275"
PLOT_GRID = "#3A4153"
ENVELOPE = "#FF4B4B"    # carmesí llamativo para la solución singular
COLORMAP = "viridis"

STEP_COLORS = {"info": ACCENT, "success": SUCCESS, "warning": WARNING, "error": ERROR}


STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-size: 10pt;
}}
QMainWindow::separator, QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
}}
QLabel {{ background: transparent; }}
QLabel#AppTitle {{ font-size: 17pt; font-weight: 700; color: {TEXT}; }}
QLabel#AppSubtitle {{ color: {TEXT_MUTED}; }}
QLabel#Muted {{ color: {TEXT_MUTED}; }}
QLineEdit, QDoubleSpinBox, QComboBox {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QLineEdit#FInput {{ font-family: "JetBrains Mono", "Fira Code", "DejaVu Sans Mono", monospace; font-size: 11pt; }}
QLineEdit#FInput[state="valid"] {{ border: 1px solid {SUCCESS}; }}
QLineEdit#FInput[state="invalid"] {{ border: 1px solid {ERROR}; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}
QDoubleSpinBox:disabled {{ color: {TEXT_MUTED}; background-color: {SURFACE}; }}
QPushButton {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 12px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton#PrimaryButton {{
    background-color: {ACCENT};
    border: none;
    color: white;
    font-weight: 700;
    font-size: 11pt;
    padding: 10px;
}}
QPushButton#PrimaryButton:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#PrimaryButton:pressed {{ background-color: #3A73D9; }}
QPushButton#PrimaryButton:disabled {{ background-color: #2A3550; color: {TEXT_MUTED}; }}
QCheckBox {{ spacing: 6px; background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QFrame#StepCard {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QListWidget {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QListWidget::item {{ padding: 2px 6px; }}
QToolBar {{ background: {SURFACE}; border: none; spacing: 2px; }}
QStatusBar {{ background: {SURFACE}; color: {TEXT_MUTED}; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {BG}; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QToolTip {{ background-color: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""


def apply_dark_theme(app: QApplication) -> None:
    """Aplica estilo Fusion + paleta oscura + QSS (la toolbar de Matplotlib
    detecta el fondo oscuro y usa iconos claros automáticamente)."""
    app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: SURFACE_2,
        QPalette.ColorRole.AlternateBase: SURFACE,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: SURFACE,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.ToolTipBase: SURFACE_2,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: TEXT_MUTED,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)

