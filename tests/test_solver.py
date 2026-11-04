"""Pruebas unitarias del motor matemático (ejecutar: python -m unittest -v)."""

from __future__ import annotations

import unittest

import numpy as np
import sympy as sp

from app.core.clairaut_solver import (
    ClairautParseError,
    ClairautSolver,
    SingularStatus,
    build_c_values,
    x,
)
from app.utils.presets import PRESETS


class TestParsing(unittest.TestCase):
    def test_valid_expressions(self) -> None:
        for text in ["p**2", "p^3", "1/p", "sqrt(p)", "2p + 1", "exp(p)", "ln(p)", "sin(p)*cos(p)", "2e3*p"]:
            with self.subTest(text=text):
                ClairautSolver.parse(text)

    def test_invalid_expressions(self) -> None:
        for text in ["", "p+", "x*p", "import os", "__import__('os')", "foo(p)", "1/0", "p;1", "(p", "p)"]:
            with self.subTest(text=text):
                with self.assertRaises(ClairautParseError):
                    ClairautSolver.parse(text)


class TestSolver(unittest.TestCase):
    def _singular_values(self, f_text: str, xs: np.ndarray) -> list[np.ndarray]:
        solver = ClairautSolver(f_text)
        return solver.evaluate_singular(xs)

    def test_mandatory_case(self) -> None:
        sol = ClairautSolver("p**2").solve()
        self.assertIs(sol.status, SingularStatus.EXPLICIT)
        self.assertEqual(len(sol.branches), 1)
        self.assertEqual(sp.simplify(sol.branches[0].y_expr + x**2 / 4), 0)
        self.assertIn("C^{2}", sol.latex_general)

    def test_shifted_case(self) -> None:
        sol = ClairautSolver("p**2 + 1").solve()
        self.assertEqual(sp.simplify(sol.branches[0].y_expr - (1 - x**2 / 4)), 0)

    def test_rational_two_branches(self) -> None:
        xs = np.array([1.0, 4.0])
        values = sorted(self._singular_values("1/p", xs), key=lambda a: a[0])
        np.testing.assert_allclose(values[0], [-2.0, -4.0])
        np.testing.assert_allclose(values[1], [2.0, 4.0])

    def test_cubic_domain(self) -> None:
        xs = np.array([-3.0, 3.0])
        for ys in self._singular_values("p**3", xs):
            self.assertTrue(np.isfinite(ys[0]))
            self.assertTrue(np.isnan(ys[1]))  # sin envolvente real para x > 0

    def test_radical(self) -> None:
        xs = np.array([-1.0, -0.5, 1.0])
        (ys,) = self._singular_values("sqrt(p)", xs)
        np.testing.assert_allclose(ys[:2], [0.25, 0.5])
        self.assertTrue(np.isnan(ys[2]))

    def test_linear_has_no_singular(self) -> None:
        sol = ClairautSolver("2*p + 1").solve()
        self.assertIs(sol.status, SingularStatus.NONE)
        self.assertTrue(sol.messages)

    def test_tangency(self) -> None:
        """La recta de parámetro C toca la envolvente en x_t = -f'(C)."""
        solver = ClairautSolver("p**2")
        curves = solver.evaluate_general([-2, 1, 3], np.linspace(-5, 5, 5))
        for curve in curves:
            xt, yt = curve.tangent_point
            self.assertAlmostEqual(yt, -xt**2 / 4)

    def test_domain_skips(self) -> None:
        solver = ClairautSolver("1/p")
        curves = solver.evaluate_general([-1, 0, 1], np.linspace(-1, 1, 3))
        self.assertIsNone(curves[1].y)  # C = 0 → f(0) = 1/0

    def test_all_presets_solve(self) -> None:
        for preset in PRESETS:
            with self.subTest(preset=preset.name):
                solver = ClairautSolver(preset.f_text)
                sol = solver.solve()
                self.assertGreaterEqual(len(sol.steps), 6)
                xs = np.linspace(*preset.x_range, 200)
                if sol.status is SingularStatus.EXPLICIT:
                    self.assertTrue(any(np.isfinite(a).any() for a in solver.evaluate_singular(xs)))

    def test_parametric_fallback(self) -> None:
        solver = ClairautSolver("p*exp(p)")
        sol = solver.solve()
        self.assertIn(sol.status, (SingularStatus.EXPLICIT, SingularStatus.PARAMETRIC))
        self.assertTrue(solver.evaluate_parametric(-5, 5))


class TestCValues(unittest.TestCase):
    def test_default(self) -> None:
        self.assertEqual(build_c_values(-3, 3, 1), [-3, -2, -1, 0, 1, 2, 3])

    def test_float_noise(self) -> None:
        self.assertEqual(build_c_values(0, 0.3, 0.1), [0.0, 0.1, 0.2, 0.3])

    def test_errors(self) -> None:
        with self.assertRaises(ValueError):
            build_c_values(0, 1, 0)
        with self.assertRaises(ValueError):
            build_c_values(2, 1, 1)


if __name__ == "__main__":
    unittest.main()

