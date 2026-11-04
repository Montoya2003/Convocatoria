# ClairautDiffViewer

Software de escritorio para **resolver analíticamente y graficar de forma interactiva** ecuaciones
diferenciales de Clairaut:

$$y = x\,y' + f(y') \quad\Longleftrightarrow\quad y = x\,p + f(p),\qquad p = y'$$

- **Motor simbólico:** SymPy (derivación, despeje con múltiples ramas, simplificación, LaTeX).
- **Motor numérico y gráfico:** NumPy + Matplotlib embebido en PyQt6 (tema oscuro, HiDPI).
- **Arquitectura:** capas separadas (núcleo matemático independiente de la UI).

---

## 1. Instalación

Requisitos: **Python 3.10 o superior**.

```bash
# 1) (Recomendado) crear y activar un entorno virtual
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (PowerShell / CMD)

# 2) instalar dependencias
pip install -r requirements.txt
```

> En Linux, si Qt indica que falta el plugin `xcb`, instale las bibliotecas del sistema, p. ej.
> `sudo apt install libxcb-cursor0` (Ubuntu/Debian).

## 2. Ejecución

```bash
python main.py
```

Pruebas unitarias del motor matemático:

```bash
python -m unittest discover -s tests -t . -v
```

---

## 3. Uso

| Zona | Descripción |
|------|-------------|
| **Caso de estudio** | Presets: `p**2` (obligatorio), `p**3`, `1/p`, `p**2 + 1`, `sqrt(p)` y extras (circular, exponencial, lineal degenerado). |
| **f(p) =** | Entrada libre validada en tiempo real (borde verde/rojo + vista previa LaTeX). Admite `+ - * / ** ^`, multiplicación implícita (`2p`), `sqrt, cbrt, exp, log/ln, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, abs`, `pi`, `e`. |
| **Rango de visualización** | X mín / X máx; Y automático (robusto ante asíntotas) o manual. Los cambios re-grafican al instante. |
| **Familia general** | C mín, C máx y paso (por defecto `[-3, 3]`, paso 1 → C ∈ {-3,…,3}). |
| **Calcular y Graficar** | `Enter` en el campo f(p) o `Ctrl+Enter`. El cálculo corre en segundo plano (la UI no se congela). |
| **Procedimiento matemático** | Resultado + Pasos 1–7 con fórmulas renderizadas. |
| **Visualizador** | Barra de Matplotlib (zoom, pan, inicio, guardar PNG/PDF/SVG), casillas para envolvente, familia, puntos de tangencia y leyenda, y lista para mostrar/ocultar **cada recta C** individualmente. |

---

## 4. Fundamento matemático (resumen para la sustentación)

1. **Cambio de variable:** `p = y'` ⇒ `y = xp + f(p)`.
2. **Derivar respecto a x:** `p = p + x p' + f'(p) p'` ⇒ `p' (x + f'(p)) = 0`.
3. **Solución general (p' = 0):** `p = C` ⇒ `y = Cx + f(C)` (familia de rectas).
4. **Condición de envolvente:** `x + f'(p) = 0`.
5. **Despeje de p(x):** puede tener varias ramas (p. ej. `±√`). El solver prueba sin supuestos y,
   si SymPy no halla raíces, con `p > 0` y `p < 0`. Cada rama se **valida numéricamente** con el
   residuo `|x + f'(p(x))| ≈ 0`, descartando tramos espurios.
6. **Solución singular:** `y_sing(x) = x·p(x) + f(p(x))`, simplificada (se elige la forma más
   compacta que sea numéricamente idéntica en su dominio).
7. **Forma paramétrica (siempre válida):** `x = -f'(p)`, `y = f(p) - p f'(p)`. Se usa como respaldo
   para graficar cuando no hay despeje explícito. La recta de parámetro C es tangente a la envolvente
   en `(x_t, y_t) = (-f'(C), C x_t + f(C))` (puntos marcados en la gráfica).

Si `f''(p) ≡ 0` (f lineal), no existe envolvente: las rectas forman un haz concurrente y el
programa lo informa explícitamente.

| f(p) | Solución general | Solución singular |
|------|------------------|-------------------|
| `p²` | `y = Cx + C²` | `y = -x²/4` |
| `p³` | `y = Cx + C³` | `y = ±(2√3/9)(-x)^{3/2}`, x ≤ 0 |
| `1/p` | `y = Cx + 1/C` (C ≠ 0) | `y = ±2√x` (y² = 4x) |
| `p² + 1` | `y = Cx + C² + 1` | `y = 1 - x²/4` |
| `√p` | `y = Cx + √C` (C ≥ 0) | `y = -1/(4x)`, x < 0 |

---

## 5. Robustez

- **Parseo seguro:** lista blanca de caracteres e identificadores antes de `parse_expr`
  (impide inyección de código); detección de paréntesis desbalanceados, símbolos ajenos a `p`,
  funciones desconocidas, divisiones entre cero y valores indeterminados.
- **Dominio real:** la evaluación numérica se hace en aritmética compleja y sólo se grafican los
  valores reales y finitos (√ de negativos, log de no positivos, 1/0 → se omiten sin errores).
  Las rectas con f(C) no real se listan como "n/d" y se informa al usuario.
- **Tiempo límite** para `solve`/`simplify`; si se excede se usa la forma paramétrica.
- Las asíntotas se cortan para no dibujar segmentos verticales falsos.

---

## 6. Estructura del proyecto

```
ClairautDiffViewer/
├── main.py                     # Punto de entrada (HiDPI + tema oscuro)
├── requirements.txt
├── README.md
├── app/
│   ├── core/
│   │   └── clairaut_solver.py  # Motor simbólico y evaluación numérica (sin dependencias de UI)
│   ├── ui/
│   │   ├── main_window.py      # Ventana principal, layouts, controlador, hilo de cálculo
│   │   ├── plot_canvas.py      # Visualizador Matplotlib (FigureCanvasQTAgg)
│   │   ├── latex_widget.py     # Render de fórmulas LaTeX (mathtext, sin instalar TeX)
│   │   └── theme.py            # Paleta y hoja de estilos del tema oscuro
│   └── utils/
│       └── presets.py          # Casos de estudio predeterminados
└── tests/
    └── test_solver.py          # Pruebas unitarias del motor
```

