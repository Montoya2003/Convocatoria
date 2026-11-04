"""
Casos de estudio predeterminados de ClairautDiffViewer.

Cada preset incluye la f(p), un rango de x recomendado (donde la envolvente
es real y visible) y la solución singular esperada como referencia docente.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    """Caso de estudio de una ecuación de Clairaut y = x·p + f(p)."""

    name: str
    f_text: str
    description: str
    expected_singular: str
    x_range: tuple[float, float] = (-5.0, 5.0)


PRESETS: list[Preset] = [
    Preset(
        name="1. Cuadrática (obligatorio): f(p) = p²",
        f_text="p**2",
        description="Caso clásico: familia y = Cx + C² con envolvente parabólica.",
        expected_singular="y = -x²/4",
        x_range=(-6.0, 6.0),
    ),
    Preset(
        name="2. Cúbica: f(p) = p³",
        f_text="p**3",
        description="x = -3p² ⇒ sólo hay envolvente real para x ≤ 0 (dos ramas ±).",
        expected_singular="y = ±(2√3/9)·(-x)^(3/2)",
        x_range=(-6.0, 2.0),
    ),
    Preset(
        name="3. Racional: f(p) = 1/p",
        f_text="1/p",
        description="C = 0 queda excluido (división entre cero). Envolvente: y² = 4x.",
        expected_singular="y = ±2√x",
        x_range=(-1.0, 6.0),
    ),
    Preset(
        name="4. Desplazada: f(p) = p² + 1",
        f_text="p**2 + 1",
        description="Igual que el caso 1 trasladado una unidad hacia arriba.",
        expected_singular="y = 1 - x²/4",
        x_range=(-6.0, 6.0),
    ),
    Preset(
        name="5. Radical: f(p) = √p",
        f_text="sqrt(p)",
        description="Dominio real p ≥ 0: sólo las rectas con C ≥ 0 existen; envolvente en x < 0.",
        expected_singular="y = -1/(4x),  x < 0",
        x_range=(-4.0, 1.0),
    ),
    Preset(
        name="6. Circular: f(p) = √(1 + p²)",
        f_text="sqrt(1 + p**2)",
        description="Las rectas son tangentes a la semicircunferencia unitaria superior.",
        expected_singular="y = √(1 - x²),  |x| < 1",
        x_range=(-2.0, 2.0),
    ),
    Preset(
        name="7. Exponencial: f(p) = eᵖ",
        f_text="exp(p)",
        description="x = -eᵖ ⇒ p = ln(-x); envolvente definida para x < 0.",
        expected_singular="y = x·ln(-x) - x",
        x_range=(-6.0, 1.0),
    ),
    Preset(
        name="8. Lineal (degenerado): f(p) = 2p + 1",
        f_text="2*p + 1",
        description="f''(p) = 0: no hay solución singular; haz de rectas por (-2, 1).",
        expected_singular="No existe",
        x_range=(-6.0, 4.0),
    ),
]


def preset_by_text(f_text: str) -> int:
    """Índice del preset cuya f(p) coincide con el texto (o -1)."""
    norm = f_text.replace(" ", "")
    for i, preset in enumerate(PRESETS):
        if preset.f_text.replace(" ", "") == norm:
            return i
    return -1

