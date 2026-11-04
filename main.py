"""
Punto de entrada de ClairautDiffViewer.

Ejecutar:
    python main.py
"""

from __future__ import annotations

import sys

import matplotlib

# El backend debe fijarse ANTES de importar cualquier módulo de Matplotlib/Qt.
matplotlib.use("QtAgg")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QFont, QGuiApplication  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.theme import apply_dark_theme  # noqa: E402


def main() -> int:
    """Crea la aplicación Qt con escalado HiDPI nítido y tema oscuro."""
    # Escalado fraccionario exacto (125 %, 150 %...) sin redondeos borrosos.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("ClairautDiffViewer")
    app.setOrganizationName("ClairautDiffViewer")
    app.setFont(QFont("Inter, Segoe UI, Ubuntu, Noto Sans, sans-serif", 10))
    apply_dark_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

