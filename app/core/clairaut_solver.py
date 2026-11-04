"""
Motor simbólico y numérico de ClairautDiffViewer.

FUNDAMENTO TEÓRICO (resumen para la sustentación)
--------------------------------------------------
Una Ecuación Diferencial de Clairaut tiene la forma

        y = x·y' + f(y')                                   (1)

Haciendo el cambio de variable p = y' se obtiene la forma algebraica

        y = x·p + f(p)                                      (2)

Derivando (2) respecto de x (recordando que p = p(x)):

        y' = p + x·p' + f'(p)·p'
        p  = p + p'·(x + f'(p))
        0  = p'·(x + f'(p))                                 (3)

La ecuación (3) se anula de dos formas:

  * Caso A — p' = 0  ⇒  p = C (constante). Sustituyendo en (2):
        y = C·x + f(C)          → SOLUCIÓN GENERAL (familia de rectas).

  * Caso B — x + f'(p) = 0  ⇒  x = -f'(p). Despejando p = p(x) y
    sustituyendo en (2) se obtiene la SOLUCIÓN SINGULAR, que no contiene
    constantes arbitrarias y es la ENVOLVENTE de la familia de rectas:
    cada recta y = C·x + f(C) es tangente a la curva singular en el punto
        x_t = -f'(C),   y_t = C·x_t + f(C).

  Si el despeje explícito de p no es posible, la envolvente siempre admite
  la representación PARAMÉTRICA (parámetro p):
        x(p) = -f'(p),     y(p) = f(p) - p·f'(p).

  Si f''(p) ≡ 0 (f lineal en p), x + f'(p) = 0 no depende de p y no existe
  envolvente: todas las rectas pasan por un mismo punto (haz de rectas).

Este módulo NO depende de la interfaz gráfica: puede usarse y probarse de
forma aislada (capa de dominio de la arquitectura limpia).
"""

from __future__ import annotations

import keyword
import re
import threading
import tokenize
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ---------------------------------------------------------------------------
# Símbolos globales del problema
# ---------------------------------------------------------------------------
#: Variable independiente.
x: sp.Symbol = sp.Symbol("x", real=True)
#: Variable dependiente (sólo para representación).
y: sp.Symbol = sp.Symbol("y")
#: p = y' (pendiente). Se declara sin supuestos para que `solve` no descarte
#: ramas; la validez real se verifica numéricamente después.
p: sp.Symbol = sp.Symbol("p")
#: Constante arbitraria de la solución general.
C: sp.Symbol = sp.Symbol("C")

#: Tiempo máximo (s) para operaciones simbólicas potencialmente costosas.
SOLVE_TIMEOUT_S: float = 8.0
SIMPLIFY_TIMEOUT_S: float = 4.0

#: Tolerancia para decidir si un valor complejo es "real".
_IMAG_TOL: float = 1e-9

# ---------------------------------------------------------------------------
# Lista blanca de identificadores permitidos en f(p) (parseo SEGURO)
# ---------------------------------------------------------------------------
_ALLOWED_NAMES: dict[str, Any] = {
    "p": p,
    # Constantes
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    # Raíces, exponenciales y logaritmos
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "root": sp.root,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    # Trigonométricas y sus inversas
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    # Hiperbólicas
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    # Otras
    "abs": sp.Abs,
    "Abs": sp.Abs,
}

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,  # permite escribir p^2 en lugar de p**2
)

_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]*$")
_NUMBER_RE = re.compile(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_MAX_INPUT_LEN = 300


class ClairautParseError(ValueError):
    """Error de validación/sintaxis al interpretar f(p)."""


class SingularStatus(Enum):
    """Resultado de la búsqueda de la solución singular."""

    EXPLICIT = "explicit"          # y_sing(x) hallada explícitamente
    PARAMETRIC = "parametric"      # sólo forma paramétrica x(p), y(p)
    NONE = "none"                  # no existe envolvente (f'' ≡ 0)


@dataclass
class SolutionStep:
    """Un paso del procedimiento matemático, listo para mostrarse en la UI."""

    title: str
    description: str
    formulas: list[str] = field(default_factory=list)  # cadenas LaTeX
    kind: str = "info"  # "info" | "success" | "warning" | "error"


@dataclass
class SingularBranch:
    """Una rama de la solución singular: p = p_i(x)  →  y = y_i(x)."""

    p_expr: sp.Expr
    y_expr: sp.Expr
    p_func: Callable[[np.ndarray], np.ndarray]
    y_func: Callable[[np.ndarray], np.ndarray]

    @property
    def latex(self) -> str:
        return sp.latex(self.y_expr)


@dataclass
class GeneralCurve:
    """Recta de la familia general evaluada numéricamente para un valor C."""

    c_value: float
    y: Optional[np.ndarray]           # None si f(C) no es real/finito
    tangent_point: Optional[tuple[float, float]]
    reason: str = ""                  # motivo si no es evaluable


@dataclass
class ClairautSolution:
    """Contenedor inmutable del resultado completo del análisis."""

    f_text: str
    f_expr: sp.Expr
    fp_expr: sp.Expr
    fpp_expr: sp.Expr
    status: SingularStatus
    branches: list[SingularBranch]
    steps: list[SolutionStep]
    messages: list[str]
    # Latex principales
    latex_original: str
    latex_p_form: str
    latex_general: str
    latex_f: str
    latex_fp: str
    latex_envelope_condition: str
    latex_parametric: str
    latex_singular: list[str]

    @property
    def has_envelope(self) -> bool:
        return self.status is not SingularStatus.NONE


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _run_with_timeout(func: Callable[[], Any], timeout: float) -> tuple[bool, Any]:
    """
    Ejecuta `func` en un hilo *daemon* y espera como máximo `timeout` segundos.

    Devuelve (terminó, resultado_o_excepción). Si la operación simbólica se
    excede del tiempo (p. ej. `solve` de una ecuación trascendente muy
    compleja) la aplicación NO se bloquea: se sigue con la forma paramétrica.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = func()
        except Exception as exc:  # noqa: BLE001 - se reporta al llamador
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return False, TimeoutError(f"operación simbólica excedió {timeout:.0f} s")
    if "error" in box:
        return True, box["error"]
    return True, box.get("result")


def _lambdify(var: sp.Symbol, expr: sp.Expr) -> Callable[[np.ndarray], np.ndarray]:
    """
    Convierte una expresión SymPy en una función NumPy vectorizada.

    Se usa SciPy si está disponible (funciones especiales como LambertW);
    en otro caso sólo NumPy. El resultado se "difunde" siempre al tamaño de la
    entrada, de modo que expresiones constantes también devuelvan arreglos.
    """
    modules: list[Any] = ["numpy"]
    try:  # pragma: no cover - depende del entorno
        import scipy  # noqa: F401

        modules = ["scipy", "numpy"]
    except ImportError:
        pass
    raw = sp.lambdify(var, expr, modules=modules)

    def vectorized(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        with np.errstate(all="ignore"):
            out = raw(arr)
        return np.broadcast_to(np.asarray(out, dtype=complex), arr.shape).copy()

    return vectorized


def real_part_where_real(values: np.ndarray) -> np.ndarray:
    """
    Devuelve la parte real donde el número es (numéricamente) real y finito;
    NaN en el resto. Así se respeta el dominio real: √(negativo), log(≤0),
    divisiones entre cero, etc., quedan fuera de la gráfica sin errores.
    """
    vals = np.asarray(values, dtype=complex)
    re_, im_ = vals.real, vals.imag
    ok = np.isfinite(re_) & np.isfinite(im_) & (np.abs(im_) <= _IMAG_TOL * (1.0 + np.abs(re_)))
    out = np.full(vals.shape, np.nan, dtype=float)
    out[ok] = re_[ok]
    return out


def _contains_invalid_numbers(expr: sp.Expr) -> bool:
    return bool(expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I))


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------
class ClairautSolver:
    """
    Resuelve analíticamente la ecuación de Clairaut  y = x·p + f(p)  y ofrece
    evaluadores numéricos vectorizados para graficar la familia general y la
    envolvente (solución singular).

    Uso típico::

        solver = ClairautSolver("p**2")
        sol = solver.solve()
        sol.latex_singular        # ['- \\frac{x^{2}}{4}']
    """

    def __init__(self, f_text: str) -> None:
        self.f_text: str = f_text.strip()
        #: f(p) simbólica; lanza ClairautParseError si la entrada es inválida.
        self.f_expr: sp.Expr = self.parse(self.f_text)
        self._solution: Optional[ClairautSolution] = None

        # Funciones numéricas de f y f' (se construyen una sola vez).
        self.fp_expr: sp.Expr = sp.diff(self.f_expr, p)
        self._f_num = _lambdify(p, self.f_expr)
        self._fp_num = _lambdify(p, self.fp_expr)

    # ------------------------------------------------------------------ parse
    @staticmethod
    def parse(text: str) -> sp.Expr:
        """
        Parseo SEGURO de f(p).

        `parse_expr` usa internamente `eval`, por lo que antes de invocarlo se
        aplican tres filtros:
          1. Longitud máxima y lista blanca de caracteres (sin ';', '[', '_'
             dobles, comillas, etc.).
          2. Todos los identificadores deben pertenecer a `_ALLOWED_NAMES`
             (p, constantes y funciones matemáticas estándar).
          3. El resultado debe depender, a lo sumo, de p y no contener
             valores indeterminados (zoo, nan, ∞) ni números complejos.
        """
        if text is None or not str(text).strip():
            raise ClairautParseError("La expresión f(p) está vacía.")
        text = str(text).strip()
        if len(text) > _MAX_INPUT_LEN:
            raise ClairautParseError(f"Expresión demasiado larga (máx. {_MAX_INPUT_LEN} caracteres).")
        if not _ALLOWED_CHARS.match(text):
            bad = sorted({ch for ch in text if not _ALLOWED_CHARS.match(ch)})
            raise ClairautParseError(f"Caracteres no permitidos: {' '.join(bad)}")
        if "__" in text:
            raise ClairautParseError("Secuencia '__' no permitida.")

        # Verificación de identificadores (se eliminan antes los literales numéricos
        # para que '2e3' no se interprete como el identificador 'e3').
        names = set(_IDENT_RE.findall(_NUMBER_RE.sub(" ", text)))
        unknown = sorted(n for n in names if n not in _ALLOWED_NAMES)
        if unknown:
            raise ClairautParseError(
                "Identificador(es) desconocido(s): " + ", ".join(unknown)
                + ". Use la variable p y funciones como sqrt, exp, log, sin, cos."
            )
        if any(keyword.iskeyword(n) for n in names):
            raise ClairautParseError("Palabras reservadas de Python no permitidas.")

        # Paréntesis balanceados (mensaje más claro que el de SyntaxError).
        depth = 0
        for ch in text:
            depth += ch == "("
            depth -= ch == ")"
            if depth < 0:
                raise ClairautParseError("Paréntesis de cierre ')' sin apertura.")
        if depth != 0:
            raise ClairautParseError("Paréntesis sin cerrar.")

        try:
            expr = parse_expr(
                text,
                local_dict=dict(_ALLOWED_NAMES),
                transformations=_TRANSFORMATIONS,
                evaluate=True,
            )
        except (SyntaxError, TypeError, tokenize.TokenError) as exc:
            raise ClairautParseError(f"Sintaxis inválida: {exc.__class__.__name__}") from exc
        except ZeroDivisionError as exc:
            raise ClairautParseError("División entre cero en la expresión.") from exc
        except Exception as exc:  # noqa: BLE001
            raise ClairautParseError(f"No se pudo interpretar la expresión: {exc}") from exc

        if not isinstance(expr, sp.Expr):
            raise ClairautParseError("La entrada no es una expresión matemática escalar.")
        if expr.atoms(AppliedUndef):
            raise ClairautParseError("La expresión contiene funciones no definidas.")
        extra = expr.free_symbols - {p}
        if extra:
            raise ClairautParseError(
                "f debe depender sólo de p; símbolos extra: " + ", ".join(map(str, extra))
            )
        if _contains_invalid_numbers(expr):
            raise ClairautParseError(
                "La expresión no es real/finita (división entre cero o valor indeterminado)."
            )
        return expr

    # ------------------------------------------------------------------ solve
    def solve(self) -> ClairautSolution:
        """
        Ejecuta el análisis analítico completo y construye el procedimiento
        paso a paso. Nunca lanza excepciones por falta de solución singular:
        en ese caso la reporta con `SingularStatus.PARAMETRIC` o `NONE`.
        """
        if self._solution is not None:
            return self._solution

        f, fp = self.f_expr, self.fp_expr
        fpp = sp.simplify(sp.diff(fp, p))
        messages: list[str] = []
        steps: list[SolutionStep] = []

        # --- Representaciones LaTeX básicas --------------------------------
        latex_f = sp.latex(f)
        latex_fp = sp.latex(fp)
        latex_original = r"y = x\,y' + " + self.wrap_latex(sp.latex(f.subs(p, sp.Symbol("y'"))))
        latex_p_form = r"y = x\,p + " + self.wrap_latex(latex_f)
        # Se compone a mano para conservar el orden didáctico "C x + f(C)".
        latex_general = r"y = C\,x + " + self.wrap_latex(sp.latex(f.subs(p, C)))
        latex_condition = r"x + " + self.wrap_latex(latex_fp) + r" = 0"
        param_y = sp.simplify(f - p * fp)
        latex_parametric = (
            r"x(p) = " + sp.latex(sp.simplify(-fp)) + r",\quad y(p) = " + sp.latex(param_y)
        )

        # --- Paso 1: cambio de variable -------------------------------------
        steps.append(SolutionStep(
            "Paso 1 · Cambio de variable p = y'",
            "La ecuación de Clairaut se escribe con la pendiente p = dy/dx, "
            "convirtiéndose en una relación algebraica entre x, y y p.",
            [latex_original, latex_p_form],
        ))

        # --- Paso 2: derivación y factorización ------------------------------
        steps.append(SolutionStep(
            "Paso 2 · Derivar respecto a x y factorizar",
            "Al derivar y = xp + f(p) respecto a x (con p = p(x)) los términos p "
            "se cancelan y queda un producto igual a cero. Cada factor genera un "
            "tipo de solución.",
            [
                r"p = p + x\,\frac{dp}{dx} + f'(p)\,\frac{dp}{dx}",
                r"\frac{dp}{dx}\,\left(x + f'(p)\right) = 0",
            ],
        ))

        # --- Paso 3: solución general ---------------------------------------
        steps.append(SolutionStep(
            "Paso 3 · Solución general (dp/dx = 0)",
            "Si dp/dx = 0 entonces p = C es constante. Sustituyendo en la ecuación "
            "se obtiene una familia uniparamétrica de RECTAS de pendiente C y "
            "ordenada al origen f(C).",
            [r"p = C \;\Rightarrow\; " + latex_general],
            kind="success",
        ))

        # --- Paso 4: derivada y condición de envolvente ---------------------
        steps.append(SolutionStep(
            "Paso 4 · Derivada f'(p) y condición de envolvente",
            "El segundo factor, x + f'(p) = 0, es la condición que define la "
            "envolvente de la familia de rectas.",
            [
                r"f(p) = " + latex_f,
                r"f'(p) = \frac{d}{dp}\left[" + latex_f + r"\right] = " + latex_fp,
                latex_condition,
            ],
        ))

        branches: list[SingularBranch] = []
        latex_singular: list[str] = []

        # --- Caso degenerado: f'' ≡ 0 ----------------------------------------
        if fpp == 0 or not fp.has(p):
            status = SingularStatus.NONE
            point_x = sp.simplify(-fp)
            point_y = sp.simplify(f - p * fp)
            msg = (
                "f'(p) no depende de p (f es lineal en p): la condición x + f'(p) = 0 "
                "no permite despejar p. NO existe solución singular; todas las rectas "
                f"pasan por el punto ({point_x}, {point_y})."
            )
            messages.append(msg)
            steps.append(SolutionStep(
                "Paso 5 · Despeje de p", msg,
                [r"f''(p) = " + sp.latex(fpp) + r"\;\Rightarrow\;\mathrm{sin\ envolvente}"],
                kind="warning",
            ))
            steps.append(SolutionStep(
                "Paso 6 · Solución singular",
                "La familia es un haz de rectas concurrentes; la envolvente degenera "
                "en un punto y no hay curva singular.",
                [r"\left(x_0, y_0\right) = \left(" + sp.latex(point_x) + ", "
                 + sp.latex(point_y) + r"\right)"],
                kind="warning",
            ))
        else:
            status, branches, despeje_step, singular_step = self._solve_singular(f, fp, messages)
            latex_singular = [b.latex for b in branches]
            steps.append(despeje_step)
            steps.append(singular_step)

            # --- Paso 7: forma paramétrica y tangencia ----------------------
            steps.append(SolutionStep(
                "Paso 7 · Forma paramétrica y tangencia",
                "Usando p como parámetro, la envolvente siempre puede escribirse "
                "así. Cada recta y = Cx + f(C) toca a la envolvente en el punto "
                "correspondiente a p = C, donde ambas comparten pendiente (tangencia).",
                [
                    latex_parametric,
                    r"\left(x_t, y_t\right) = \left(-f'(C),\; C\,x_t + f(C)\right)",
                ],
                kind="info",
            ))

        self._solution = ClairautSolution(
            f_text=self.f_text,
            f_expr=f,
            fp_expr=fp,
            fpp_expr=fpp,
            status=status,
            branches=branches,
            steps=steps,
            messages=messages,
            latex_original=latex_original,
            latex_p_form=latex_p_form,
            latex_general=latex_general,
            latex_f=latex_f,
            latex_fp=latex_fp,
            latex_envelope_condition=latex_condition,
            latex_parametric=latex_parametric,
            latex_singular=latex_singular,
        )
        return self._solution

    def _solve_singular(
        self, f: sp.Expr, fp: sp.Expr, messages: list[str]
    ) -> tuple[SingularStatus, list[SingularBranch], SolutionStep, SolutionStep]:
        """
        Despeja p de x + f'(p) = 0 (posiblemente varias ramas) y sustituye en
        y = x·p + f(p) para obtener y_sing(x).
        """
        equation = sp.Eq(x + fp, 0)
        roots, failure = self._solve_for_p(fp)

        branches: list[SingularBranch] = []
        for root in roots:
            y_raw = x * root + f.subs(p, root)
            if _contains_invalid_numbers(y_raw) and not y_raw.has(sp.I):
                continue
            try:
                branch = SingularBranch(
                    p_expr=root,
                    y_expr=y_raw,
                    p_func=_lambdify(x, root),
                    y_func=_lambdify(x, y_raw),
                )
            except Exception:  # noqa: BLE001 - expresión no traducible a NumPy
                continue
            if not self._branch_has_real_points(branch):
                continue
            self._simplify_branch(branch)
            branches.append(branch)

        if branches:
            n = len(branches)
            despeje = SolutionStep(
                "Paso 5 · Despeje de p en términos de x",
                f"Se resuelve x + f'(p) = 0 para p. Se obtuvo{'n' if n > 1 else ''} "
                f"{n} rama{'s' if n > 1 else ''} con valores reales.",
                [latex_cond_p(i, b.p_expr, n) for i, b in enumerate(branches)],
            )
            singular = SolutionStep(
                "Paso 6 · Solución singular (envolvente)",
                "Sustituyendo p(x) en y = xp + f(p) y simplificando se obtiene la "
                "solución singular. No contiene constantes arbitrarias y no se "
                "obtiene de la general para ningún valor de C.",
                [latex_y_sing(i, b.y_expr, n) for i, b in enumerate(branches)],
                kind="success",
            )
            return SingularStatus.EXPLICIT, branches, despeje, singular

        reason = failure or "Las ramas halladas no producen valores reales."
        messages.append(reason + " Se usa la forma paramétrica de la envolvente.")
        despeje = SolutionStep(
            "Paso 5 · Despeje de p en términos de x",
            reason,
            [sp.latex(equation.lhs) + " = 0"],
            kind="warning",
        )
        singular = SolutionStep(
            "Paso 6 · Solución singular (envolvente)",
            "No hay expresión explícita y_sing(x); la envolvente se describe "
            "mediante las ecuaciones paramétricas del Paso 7 y así se grafica.",
            [r"x = -f'(p),\quad y = f(p) - p\,f'(p)"],
            kind="warning",
        )
        return SingularStatus.PARAMETRIC, [], despeje, singular

    def _solve_for_p(self, fp: sp.Expr) -> tuple[list[sp.Expr], Optional[str]]:
        """
        Resuelve x + f'(p) = 0 para p.

        Primero sin supuestos sobre p (obtiene todas las ramas, p. ej. ±√).
        Si SymPy no halla raíces —típico con √p o log p, donde la comprobación
        de soluciones falla por no conocer el signo— se reintenta suponiendo
        p > 0 y p < 0 por separado. Cada candidata se valida luego
        numéricamente con el residuo |x + f'(p(x))|.
        """
        p_pos = sp.Symbol("p", positive=True)
        p_neg = sp.Symbol("p", negative=True)
        roots: list[sp.Expr] = []
        failure: Optional[str] = None
        for sym in (p, p_pos, p_neg):
            eq = sp.Eq(x + fp.subs(p, sym), 0)
            # En los reintentos con supuestos se omite la comprobación simbólica
            # (check=False): la validación numérica posterior es más fiable.
            finished, result = _run_with_timeout(
                lambda e=eq, s=sym: sp.solve(e, s, check=(s is p)), SOLVE_TIMEOUT_S
            )
            if not finished:
                failure = "El despeje simbólico excedió el tiempo límite."
                continue
            if isinstance(result, Exception):
                failure = f"SymPy no pudo despejar p ({result.__class__.__name__})."
                continue
            # Sólo interesan raíces que dependan de x (p constante no es envolvente).
            found = [r for r in (result or []) if isinstance(r, sp.Expr) and r.has(x)]
            for r in found:
                if r not in roots:
                    roots.append(r)
            if found and sym is p:
                break  # la solución sin supuestos ya contiene todas las ramas
        if not roots and failure is None:
            failure = "No existe una expresión explícita p(x) que satisfaga x + f'(p) = 0."
        return roots, failure

    def _simplify_branch(self, branch: SingularBranch) -> None:
        """
        Busca la forma más compacta de y_sing(x) que sea NUMÉRICAMENTE idéntica
        a la original en todo su dominio válido. Se prueban `simplify` general y
        simplificaciones bajo los supuestos x > 0 y x < 0 (p. ej. -2/√(1/x)
        → -2√x); el control numérico impide aceptar formas que cambien de rama.
        """
        y_raw = branch.y_expr
        x_pos = sp.Symbol("x", positive=True)
        x_neg = sp.Symbol("x", negative=True)
        builders: list[Callable[[], sp.Expr]] = [
            lambda: sp.simplify(y_raw),
            lambda: sp.simplify(sp.powdenest(y_raw, force=True)),
            lambda: sp.simplify(y_raw.subs(x, x_pos)).subs(x_pos, x),
            lambda: sp.simplify(y_raw.subs(x, x_neg)).subs(x_neg, x),
        ]
        xs = np.concatenate([-np.logspace(-3, 3, 150), np.logspace(-3, 3, 150)])
        reference = self._eval_branch(branch, xs)
        valid = np.isfinite(reference)
        # Si la rama tiene signo constante, y = ±√(y²) suele dar la forma más
        # limpia de expresiones radicales (p. ej. la semicircunferencia √(1-x²)).
        if valid.any() and (np.all(reference[valid] >= 0) or np.all(reference[valid] <= 0)):
            sign = 1 if np.all(reference[valid] >= 0) else -1
            builders.append(lambda: sign * sp.sqrt(sp.factor(sp.simplify(y_raw**2))))
            builders.append(lambda: sign * sp.sqrt(sp.expand(sp.simplify(y_raw**2))))
        best_expr, best_func = y_raw, branch.y_func
        best_ops = sp.count_ops(y_raw)
        for build in builders:
            done, cand = _run_with_timeout(build, SIMPLIFY_TIMEOUT_S)
            if not done or not isinstance(cand, sp.Expr) or cand.has(sp.zoo, sp.nan):
                continue
            ops = sp.count_ops(cand)
            if ops >= best_ops:
                continue
            try:
                func = _lambdify(x, cand)
                vals = real_part_where_real(func(xs.astype(complex)))
            except Exception:  # noqa: BLE001
                continue
            if np.all(np.isfinite(vals[valid])) and np.allclose(
                vals[valid], reference[valid], rtol=1e-7, atol=1e-9
            ):
                best_expr, best_func, best_ops = cand, func, ops
        branch.y_expr, branch.y_func = best_expr, best_func

    def _branch_has_real_points(self, branch: SingularBranch) -> bool:
        """Comprueba numéricamente que la rama tenga puntos reales válidos."""
        xs = np.concatenate([-np.logspace(-3, 3, 200), np.logspace(-3, 3, 200)])
        try:
            return bool(np.isfinite(self._eval_branch(branch, xs)).any())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------- evaluación numérica
    def _eval_branch(self, branch: SingularBranch, xs: np.ndarray) -> np.ndarray:
        """
        Evalúa una rama y_sing(x) aplicando dos filtros de validez:
          * p(x) e y(x) deben ser reales y finitos;
          * residuo |x + f'(p(x))| ≈ 0 (descarta tramos espurios que `solve`
            introduce al elevar al cuadrado, p. ej. con f(p) = √p).
        """
        xs = np.asarray(xs, dtype=float)
        xc = xs.astype(complex)
        p_vals = branch.p_func(xc)
        y_vals = real_part_where_real(branch.y_func(xc))
        p_real = real_part_where_real(p_vals)
        with np.errstate(all="ignore"):
            residual = np.abs(xs + real_part_where_real(self._fp_num(p_real.astype(complex))))
        valid = np.isfinite(p_real) & np.isfinite(y_vals) & (residual <= 1e-6 * (1.0 + np.abs(xs)))
        out = np.where(valid, y_vals, np.nan)
        return out

    def evaluate_f(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """f(p) real (NaN fuera del dominio)."""
        return real_part_where_real(self._f_num(np.asarray(values, dtype=complex)))

    def evaluate_fp(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """f'(p) real (NaN fuera del dominio)."""
        return real_part_where_real(self._fp_num(np.asarray(values, dtype=complex)))

    def evaluate_general(self, c_values: Sequence[float], xs: np.ndarray) -> list[GeneralCurve]:
        """
        Evalúa la familia general y = C·x + f(C) para cada C, junto con el punto
        de tangencia con la envolvente (x_t, y_t) = (-f'(C), C·x_t + f(C)).
        """
        xs = np.asarray(xs, dtype=float)
        cs = np.asarray(list(c_values), dtype=float)
        f_c = self.evaluate_f(cs)
        fp_c = self.evaluate_fp(cs)
        envelope_exists = self.solve().has_envelope
        curves: list[GeneralCurve] = []
        for c, fc, fpc in zip(cs, f_c, fp_c):
            if not np.isfinite(fc):
                curves.append(GeneralCurve(float(c), None, None,
                                           f"f({c:g}) no es real/finito (fuera del dominio)"))
                continue
            tangent: Optional[tuple[float, float]] = None
            if envelope_exists and np.isfinite(fpc):
                xt = -fpc
                tangent = (float(xt), float(c * xt + fc))
            curves.append(GeneralCurve(float(c), c * xs + fc, tangent))
        return curves

    def evaluate_singular(self, xs: np.ndarray) -> list[np.ndarray]:
        """Evalúa cada rama explícita de la solución singular sobre `xs`."""
        sol = self.solve()
        if sol.status is not SingularStatus.EXPLICIT:
            return []
        results: list[np.ndarray] = []
        for branch in sol.branches:
            try:
                results.append(self._eval_branch(branch, xs))
            except Exception:  # noqa: BLE001 - p. ej. función especial sin NumPy
                continue
        return results

    def evaluate_parametric(
        self, x_min: float, x_max: float, samples: int = 6000
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Evalúa la envolvente paramétrica x(p) = -f'(p), y(p) = f(p) - p·f'(p)
        con un muestreo denso y simétrico de p (lineal cerca de 0, logarítmico
        lejos). Devuelve segmentos continuos dentro de [x_min, x_max].
        """
        if not self.solve().has_envelope:
            return []
        half = samples // 2
        pos = np.unique(np.concatenate([np.linspace(1e-6, 5, half // 2), np.logspace(-6, 4, half // 2)]))
        ps = np.concatenate([-pos[::-1], [0.0], pos])
        fp_v = self.evaluate_fp(ps)
        f_v = self.evaluate_f(ps)
        with np.errstate(all="ignore"):
            xs = -fp_v
            ys = f_v - ps * fp_v
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= x_min) & (xs <= x_max)
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return segments
        # Corta en huecos de índice (cambios de dominio) y saltos grandes en x.
        span = max(x_max - x_min, 1e-12)
        breaks = np.flatnonzero((np.diff(idx) > 1) | (np.abs(np.diff(xs[idx])) > 0.25 * span)) + 1
        for chunk in np.split(idx, breaks):
            if chunk.size >= 2:
                segments.append((xs[chunk], ys[chunk]))
        return segments

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def wrap_latex(latex: str) -> str:
        """Agrupa con paréntesis si f empieza con signo, para 'x p + (-...)'."""
        latex = latex.strip()
        if latex.startswith("-"):
            return r"\left(" + latex + r"\right)"
        return latex


def latex_cond_p(index: int, expr: sp.Expr, total: int) -> str:
    """LaTeX de la rama p_i(x)."""
    name = f"p_{{{index + 1}}}" if total > 1 else "p"
    return f"{name}(x) = " + sp.latex(expr)


def latex_y_sing(index: int, expr: sp.Expr, total: int) -> str:
    """LaTeX de la rama y_sing,i(x)."""
    name = r"y_{\mathrm{sing}," + str(index + 1) + "}" if total > 1 else r"y_{\mathrm{sing}}"
    return f"{name}(x) = " + sp.latex(expr)


def build_c_values(c_min: float, c_max: float, step: float, max_count: int = 61) -> list[float]:
    """
    Genera los valores de C en [c_min, c_max] con paso `step` (inclusive),
    redondeados para evitar ruido de coma flotante (p. ej. 0.30000000000000004).
    """
    if step <= 0:
        raise ValueError("El paso de C debe ser positivo.")
    if c_min > c_max:
        raise ValueError("C mínimo no puede ser mayor que C máximo.")
    count = int(np.floor((c_max - c_min) / step + 1e-9)) + 1
    if count > max_count:
        raise ValueError(f"Demasiados valores de C ({count}); máximo {max_count}. Aumente el paso.")
    return [float(np.round(c_min + i * step, 10)) for i in range(count)]
