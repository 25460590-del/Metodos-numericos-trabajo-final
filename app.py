# -*- coding: utf-8 -*-
"""
==============================================================================
 SOLUCIONADOR DE SISTEMAS DE ECUACIONES LINEALES 3x3
==============================================================================
 Materia   : Métodos Numéricos (AEC-1046)
 Institución: Instituto Tecnológico de Colima (TecNM)
 Docente   : Mtro. Ricardo Olivera Guerrero
 Integrantes: Karol Gabriel Cobian Zamora
              Kalel Isahid Ramos Espinoza

 Ejecución : streamlit run app.py

 NOTAS DE DISEÑO
 ---------------
 * Los tres métodos (LU-Doolittle, Gauss-Seidel y Krylov) están programados
   DESDE CERO con ciclos y operaciones aritméticas de Python. No se usa
   numpy.linalg.solve, scipy.linalg.lu ni equivalentes. NumPy sólo se usa para
   construir la malla de la gráfica 3D; el determinante se calcula a mano
   (regla de Sarrus / cofactores).
 * Cada método devuelve una lista de "pasos" (objetos Paso). Un Paso es una
   lista de bloques (texto, ecuación LaTeX, matrices, tabla, aviso). La misma
   estructura se dibuja en pantalla (Streamlit) y en el PDF, así que el PDF
   contiene exactamente el procedimiento que se ve en la aplicación.
 * Todos los valores calculados se muestran con exactamente 6 decimales.
==============================================================================
"""

import inspect
import itertools
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

# ==============================================================================
# 0. CONSTANTES GENERALES
# ==============================================================================
EPS = 1e-12                 # Umbral para considerar un número "igual a cero"
TOL_COMPROBACION = 1e-4     # Tolerancia para marcar ✓ al sustituir la solución
TOL_COINCIDENCIA = 1e-4     # Tolerancia para decir que dos métodos coinciden
VARS = ["x", "y", "z"]      # Nombres de las incógnitas

M_LU = "LU (Doolittle)"
M_GS = "Gauss-Seidel"
M_KR = "Krylov"
METODOS = [M_LU, M_GS, M_KR]

# Paleta pastel beige (UI, gráficas y PDF)
BEIGE_FONDO = "#FAF6EF"
BEIGE_TARJETA = "#F3EADB"
BEIGE_BORDE = "#E4D6C0"
BEIGE_ACENTO = "#C8A27C"
CAFE_TEXTO = "#4A3F35"
CAFE_TITULO = "#6B5440"
COLORES_PLANOS = ["#D4A373", "#A3B18A", "#B5838D"]   # caramelo, salvia, rosa antiguo

# Sistemas de prueba obligatorios (se cargan con el botón "Cargar Ejemplo")
EJ_LU = "1 · Prueba LU (Doolittle)"
EJ_GS = "2 · Prueba Gauss-Seidel (requiere reordenamiento)"
EJ_KR = "3 · Prueba Krylov"
EJ_PIV = "4 · Prueba Validación (pivote cero)"
EJEMPLOS = {
    EJ_LU: {"A": [[1, 2, 3], [1, 0, 8], [2, 5, 3]], "b": [14, 25, 21]},
    EJ_GS: {"A": [[1, 10, 1], [2, 2, 10], [10, 1, 1]], "b": [13, 14, 12]},
    EJ_KR: {"A": [[4, 1, 0], [1, 3, -1], [0, 2, 5]], "b": [6, 4, 19]},
    EJ_PIV: {"A": [[0, 2, 1], [1, 1, 1], [2, 1, 3]], "b": [4, 3, 7]},
}
# Ejemplo sugerido para cada método (se preselecciona en el menú desplegable)
EJEMPLO_POR_METODO = {M_LU: EJ_LU, M_GS: EJ_GS, M_KR: EJ_KR}


# ==============================================================================
# 1. FORMATO NUMÉRICO Y DE ECUACIONES
# ==============================================================================
def n6(v):
    """Número con EXACTAMENTE 6 decimales (evita el feo '-0.000000')."""
    v = float(v)
    if abs(v) < 5e-7:
        v = 0.0
    return f"{v:.6f}"


def g(v):
    """Formato compacto para los datos CAPTURADOS (2, -1, 0.5...) en ecuaciones."""
    v = float(v)
    if abs(v) < EPS:
        return "0"
    if abs(v - round(v)) < EPS and abs(v) < 1e12:
        return str(int(round(v)))
    return f"{v:.10g}"


def ecuacion_partes(fila, rhs):
    """Devuelve (lado_izquierdo, lado_derecho) de una ecuación, ej. ('2x + 3y - z', '5')."""
    lhs = ""
    for j, c in enumerate(fila):
        if abs(c) < EPS:
            continue                      # los términos con coeficiente 0 se omiten
        mag = abs(c)
        coef = "" if abs(mag - 1) < EPS else g(mag)
        termino = f"{coef}{VARS[j]}"
        if lhs == "":
            lhs = ("-" if c < 0 else "") + termino
        else:
            lhs += (" - " if c < 0 else " + ") + termino
    return (lhs if lhs else "0"), g(rhs)


def ecuacion_str(fila, rhs):
    """Ecuación como texto plano: '2x + 3y - z = 5'."""
    lhs, r = ecuacion_partes(fila, rhs)
    return f"{lhs} = {r}"


def latex_sistema(A, b):
    """Sistema completo en LaTeX (ecuaciones alineadas en el '=')."""
    filas = []
    for i in range(3):
        lhs, r = ecuacion_partes(A[i], b[i])
        filas.append(f"{lhs} &= {r}")
    return r"\begin{aligned}" + r" \\ ".join(filas) + r"\end{aligned}"


def sistema_plano(A, b):
    """Sistema como texto plano (para el PDF)."""
    return "\n".join(ecuacion_str(A[i], b[i]) for i in range(3))


def latex_matrix(M, fmt=n6):
    """Matriz en LaTeX con corchetes (bmatrix)."""
    filas = [" & ".join(fmt(v) for v in fila) for fila in M]
    return r"\begin{bmatrix}" + r" \\ ".join(filas) + r"\end{bmatrix}"


def latex_aug(M, col, fmt=n6):
    """Matriz aumentada [M | col] en LaTeX."""
    filas = [" & ".join(fmt(v) for v in M[i]) + " & " + fmt(col[i]) for i in range(len(M))]
    return r"\left[\begin{array}{ccc|c}" + r" \\ ".join(filas) + r"\end{array}\right]"


def forma_matricial_latex(A, b):
    """AX = B en LaTeX con los datos capturados (formato compacto)."""
    vec_x = r"\begin{bmatrix} x \\ y \\ z \end{bmatrix}"
    return (latex_matrix(A, g) + vec_x + " = " +
            latex_matrix([[v] for v in b], g))


def nombre_latex(nombre):
    """Convierte el nombre de una matriz (texto plano) a LaTeX."""
    return nombre.replace("·", r"\cdot ")


def sistema_generico(M, rhs, vars_tex, vars_txt):
    """
    Sistema M·v = rhs con coeficientes a 6 decimales (se omiten los ceros).
    Devuelve (latex, texto_plano). Sirve para L·Y=B, U·X=Y y el sistema de Krylov.
    """
    filas_tex, filas_txt = [], []
    for i in range(3):
        tex, txt = "", ""
        for j in range(3):
            c = M[i][j]
            if abs(c) < EPS:
                continue
            if tex == "":
                pref = "-" if c < 0 else ""
            else:
                pref = " - " if c < 0 else " + "
            tex += f"{pref}{n6(abs(c))}\\,{vars_tex[j]}"
            txt += f"{pref}{n6(abs(c))} {vars_txt[j]}"
        if tex == "":
            tex, txt = "0", "0"
        filas_tex.append(f"{tex} &= {n6(rhs[i])}")
        filas_txt.append(f"{txt} = {n6(rhs[i])}")
    return (r"\begin{aligned}" + r" \\ ".join(filas_tex) + r"\end{aligned}",
            "\n".join(filas_txt))


# ==============================================================================
# 2. ÁLGEBRA BÁSICA (hecha a mano con ciclos)
# ==============================================================================
def copia(M):
    """Copia profunda de una matriz (lista de listas)."""
    return [fila[:] for fila in M]


def identidad(n=3):
    """Matriz identidad n x n."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def det3(M):
    """Determinante 3x3 por la regla de Sarrus (sin usar numpy.linalg)."""
    return (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] +
            M[0][2] * M[1][0] * M[2][1] - M[0][2] * M[1][1] * M[2][0] -
            M[0][0] * M[1][2] * M[2][1] - M[0][1] * M[1][0] * M[2][2])


def mat_vec(A, v):
    """Producto matriz por vector A·v con ciclos."""
    res = []
    for i in range(3):
        s = 0.0
        for j in range(3):
            s += A[i][j] * v[j]
        res.append(s)
    return res


def mat_mul(A, B):
    """Producto de matrices A·B con tres ciclos anidados."""
    C = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i][j] += A[i][k] * B[k][j]
    return C


def es_singular(A):
    """True si det(A) es (numéricamente) cero. Devuelve (singular, determinante)."""
    d = det3(A)
    escala = max(abs(v) for fila in A for v in fila)
    umbral = 1e-12 * (escala ** 3 if escala > 0 else 1.0)
    return abs(d) <= umbral, d


# ==============================================================================
# 3. ESTRUCTURA "PASO" (la comparten la pantalla y el PDF)
# ==============================================================================
class Paso:
    """
    Un paso del procedimiento = título + lista de bloques. Tipos de bloque:
      ("texto", markdown)              texto explicativo
      ("eq", latex, texto_plano)       ecuación(es): LaTeX en pantalla, texto en PDF
      ("mats", [(nombre, M, aug), ...]) una o varias matrices (aug = columna extra o None)
      ("tabla", DataFrame)             tabla
      ("aviso", tipo, mensaje)         tipo: info | ok | warn | error
    """

    def __init__(self, titulo):
        self.titulo = titulo
        self.bloques = []

    def texto(self, md):
        self.bloques.append(("texto", md))

    def eq(self, latex, plano):
        self.bloques.append(("eq", latex, plano))

    def mats(self, *items):
        # IMPORTANTE: se guarda una COPIA de cada matriz. Los métodos modifican U "en sitio"
        # durante la eliminación; sin la copia, todos los pasos mostrarían el valor final.
        copias = [(nombre, [list(fila) for fila in M], None if aug is None else list(aug))
                  for nombre, M, aug in items]
        self.bloques.append(("mats", copias))

    def tabla(self, df):
        self.bloques.append(("tabla", df))

    def aviso(self, tipo, msg):
        self.bloques.append(("aviso", tipo, msg))


def nuevo(pasos, titulo):
    """Crea un Paso numerado automáticamente y lo agrega a la lista."""
    p = Paso(f"Paso {len(pasos) + 1} · {titulo}")
    pasos.append(p)
    return p


def resultado_base(nombre):
    """Estructura estándar que devuelve cada método."""
    return {"metodo": nombre, "pasos": [], "solucion": None, "error": None, "extra": {}}


# ==============================================================================
# 4. MÉTODO A: DESCOMPOSICIÓN LU (DOOLITTLE)
# ==============================================================================
def metodo_lu(A, b):
    """
    Descomposición LU de Doolittle (L con 1 en la diagonal) mediante eliminación
    de Gauss. Si un pivote es 0 se aplica pivoteo parcial (intercambio de
    renglones) y se trabaja con P·A = L·U.
    """
    n = 3
    res = resultado_base(M_LU)
    pasos = res["pasos"]

    U = copia(A)              # U empieza siendo A y se irá triangularizando
    L = identidad(n)          # L guardará los multiplicadores m_ij
    bp = list(b)              # B reordenado si hay intercambios de renglón
    perm = list(range(n))     # permutación acumulada de renglones
    n_swaps = 0

    # ---------------- Paso 1: A y B originales -------------------------------
    p = nuevo(pasos, "Matriz A y vector B originales")
    p.texto("Se resuelve **AX = B** buscando la factorización **A = L·U**, "
            "donde L es triangular inferior con 1s en la diagonal y U es triangular superior.")
    p.mats(("A", A, None), ("B", [[v] for v in b], None))

    # ---------------- Pasos 2..n: eliminación de Gauss -----------------------
    for k in range(n - 1):                      # k = columna del pivote actual
        p = nuevo(pasos, f"Eliminación de Gauss — columna {k + 1}")

        # --- Validación del pivote: si es 0 se hace pivoteo parcial ---
        if abs(U[k][k]) < EPS:
            candidatos = [i for i in range(k + 1, n) if abs(U[i][k]) > EPS]
            if not candidatos:
                res["error"] = (f"El pivote a{k + 1}{k + 1} es 0 y no existe un renglón inferior "
                                "para intercambiar: la matriz es singular.")
                p.aviso("error", res["error"])
                return res
            fila_p = max(candidatos, key=lambda i: abs(U[i][k]))   # mayor |valor|
            p.aviso("warn", f"El pivote a{k + 1}{k + 1} = 0. Se aplica pivoteo parcial: "
                            f"intercambio F{k + 1} ↔ F{fila_p + 1}.")
            U[k], U[fila_p] = U[fila_p], U[k]                       # intercambio en U
            bp[k], bp[fila_p] = bp[fila_p], bp[k]                   # ... en B
            perm[k], perm[fila_p] = perm[fila_p], perm[k]           # ... en P
            for j in range(k):                                       # ... y en los m_ij ya calculados
                L[k][j], L[fila_p][j] = L[fila_p][j], L[k][j]
            n_swaps += 1
            p.mats(("A", U, None))

        pivote = U[k][k]
        p.eq(r"\text{Pivote: } a_{" + f"{k + 1}{k + 1}" + r"} = " + n6(pivote),
             f"Pivote: a{k + 1}{k + 1} = {n6(pivote)}")

        for i in range(k + 1, n):               # renglones debajo del pivote
            # Multiplicador m_ik = a_ik / a_kk
            m = U[i][k] / pivote
            L[i][k] = m                          # se guarda en L
            idx_i, idx_k = f"{i + 1}{k + 1}", f"{k + 1}{k + 1}"
            p.eq(r"m_{" + idx_i + r"} = \dfrac{a_{" + idx_i + r"}}{a_{" + idx_k + r"}} = \dfrac{"
                 + n6(U[i][k]) + "}{" + n6(pivote) + "} = " + n6(m),
                 f"m{idx_i} = a{idx_i} / a{idx_k} = {n6(U[i][k])} / {n6(pivote)} = {n6(m)}")
            # Operación de renglón: F_i <- F_i - m_ik * F_k
            for j in range(k, n):
                U[i][j] -= m * U[k][j]
            U[i][k] = 0.0                        # el elemento eliminado queda exactamente en 0
            p.eq(r"F_{" + str(i + 1) + r"} \leftarrow F_{" + str(i + 1) + r"} - (" + n6(m)
                 + r")\,F_{" + str(k + 1) + "}",
                 f"F{i + 1} <- F{i + 1} - ({n6(m)}) * F{k + 1}")
        p.mats((f"A({k + 1})", U, None))        # matriz resultante tras esta columna

    # ---------------- Paso: matrices U y L -----------------------------------
    p = nuevo(pasos, "Matrices U (triangular superior) y L (triangular inferior)")
    if n_swaps > 0:
        P = [[1.0 if perm[i] == j else 0.0 for j in range(n)] for i in range(n)]
        p.aviso("warn", "Hubo intercambio de renglones, por lo que la factorización es P·A = L·U.")
        p.mats(("P", P, None))
    p.mats(("L", L, None), ("U", U, None))

    # ---------------- Paso: verificación L·U = A -----------------------------
    p = nuevo(pasos, "Verificación matemática: L × U = A")
    LU = mat_mul(L, U)
    PA = [A[perm[i]][:] for i in range(n)]
    nombre_ref = "A" if n_swaps == 0 else "P·A"
    err = max(abs(LU[i][j] - PA[i][j]) for i in range(n) for j in range(n))
    p.mats(("L·U", LU, None), (nombre_ref, PA, None))
    if err < 1e-9:
        p.aviso("ok", f"✓ L·U coincide con {nombre_ref} (error máximo = {n6(err)}).")
    else:
        p.aviso("warn", f"L·U difiere de {nombre_ref} (error máximo = {n6(err)}).")

    # ---------------- Etapa 1: L·Y = B (sustitución hacia adelante) ----------
    p = nuevo(pasos, "Etapa 1 — Resolver L·Y = B (sustitución hacia adelante)")
    tex, txt = sistema_generico(L, bp, ["y_1", "y_2", "y_3"], ["y1", "y2", "y3"])
    p.eq(tex, txt)
    Y = [0.0] * n
    for i in range(n):
        s = bp[i]
        expr = n6(bp[i])
        for j in range(i):                       # restar l_ij * y_j ya conocidos
            s -= L[i][j] * Y[j]
            expr += f" - ({n6(L[i][j])})({n6(Y[j])})"
        Y[i] = s                                 # L tiene 1 en la diagonal: no se divide
        p.eq(f"y_{i + 1} = {expr} = {n6(Y[i])}", f"y{i + 1} = {expr} = {n6(Y[i])}")
    p.mats(("Y", [[v] for v in Y], None))

    # ---------------- Etapa 2: U·X = Y (sustitución hacia atrás) -------------
    p = nuevo(pasos, "Etapa 2 — Resolver U·X = Y (sustitución hacia atrás)")
    tex, txt = sistema_generico(U, Y, VARS, VARS)
    p.eq(tex, txt)
    X = [0.0] * n
    for i in range(n - 1, -1, -1):               # z, luego y, luego x
        if abs(U[i][i]) < EPS:
            res["error"] = f"División entre cero: u{i + 1}{i + 1} = 0 (sistema singular)."
            p.aviso("error", res["error"])
            return res
        s = Y[i]
        expr = n6(Y[i])
        for j in range(i + 1, n):                # restar u_ij * x_j ya conocidos
            s -= U[i][j] * X[j]
            expr += f" - ({n6(U[i][j])})({n6(X[j])})"
        X[i] = s / U[i][i]
        p.eq(f"{VARS[i]} = " + r"\dfrac{" + expr + "}{" + n6(U[i][i]) + "} = " + n6(X[i]),
             f"{VARS[i]} = ({expr}) / {n6(U[i][i])} = {n6(X[i])}")

    res["solucion"] = X
    res["extra"] = {"detalle": "Factorización LU" + (" con pivoteo parcial" if n_swaps else ""),
                    "L": L, "U": U}
    return res


# ==============================================================================
# 5. MÉTODO B: GAUSS-SEIDEL
# ==============================================================================
def dominancia(M):
    """
    Evalúa dominancia diagonal por renglones. Devuelve (estricta, debil):
      estricta: |a_ii| > Σ|a_ij| en TODOS los renglones.
      debil   : |a_ii| >= Σ|a_ij| en todos y estricta en al menos uno.
    """
    estricta, debil, alguna = True, True, False
    for i in range(3):
        d = abs(M[i][i])
        s = sum(abs(M[i][j]) for j in range(3) if j != i)
        if not d > s + EPS:
            estricta = False
        if d + EPS < s:
            debil = False
        if d > s + EPS:
            alguna = True
    return estricta, (debil and alguna)


def df_dominancia(M):
    """Tabla |a_ii| vs Σ|a_ij| por renglón (6 decimales)."""
    filas = []
    for i in range(3):
        d = abs(M[i][i])
        s = sum(abs(M[i][j]) for j in range(3) if j != i)
        veredicto = "Sí" if d > s + EPS else ("Igual" if abs(d - s) <= EPS else "No")
        filas.append({"Fila": str(i + 1), "|a_ii|": n6(d),
                      "Σ|a_ij| (j≠i)": n6(s), "¿|a_ii| > Σ?": veredicto})
    return pd.DataFrame(filas)


def buscar_reordenamiento(A):
    """
    Prueba las 6 permutaciones de renglones. Prioridad: (1) estrictamente
    dominante, (2) débilmente dominante, (3) la que maximiza el producto de la
    diagonal (se avisará que podría no converger).
    Devuelve (permutación, tipo) con tipo en {"estricta", "débil", None}.
    """
    permutaciones = list(itertools.permutations(range(3)))   # (0,1,2) va primero
    for buscar_estricta in (True, False):
        for pr in permutaciones:
            B = [A[i] for i in pr]
            est, deb = dominancia(B)
            if (est if buscar_estricta else deb):
                return pr, ("estricta" if buscar_estricta else "débil")
    mejor, mejor_val = permutaciones[0], -1.0
    for pr in permutaciones:
        val = abs(A[pr[0]][0] * A[pr[1]][1] * A[pr[2]][2])
        if val > mejor_val:
            mejor, mejor_val = pr, val
    return mejor, None


def formula_despeje(A, b, i):
    """Fórmula de iteración de la variable i. Devuelve (latex, texto_plano)."""
    var = VARS[i]
    num_tex, num_txt = g(b[i]), g(b[i])
    for j in range(3):
        if j == i:
            continue
        c = A[i][j]
        if abs(c) < EPS:
            continue
        nivel = "(k+1)" if j < i else "(k)"       # variables ya actualizadas usan k+1
        coef = "" if abs(abs(c) - 1) < EPS else g(abs(c))
        signo = " - " if c > 0 else " + "          # al despejar cambia el signo
        num_tex += f"{signo}{coef}{VARS[j]}^{{{nivel}}}"
        num_txt += f"{signo}{coef}{VARS[j]}{nivel}"
    return (f"{var}^{{(k+1)}} = " + r"\dfrac{" + num_tex + "}{" + g(A[i][i]) + "}",
            f"{var}(k+1) = ({num_txt}) / {g(A[i][i])}")


def metodo_gauss_seidel(A, b, x0, tol, max_iter):
    """
    Gauss-Seidel: reordena para lograr dominancia diagonal, despeja cada
    variable y itera usando SIEMPRE los valores más recientes.
    """
    n = 3
    res = resultado_base(M_GS)
    pasos = res["pasos"]

    # ---------------- Paso 1: dominancia diagonal y reordenamiento -----------
    p = nuevo(pasos, "Revisión de dominancia diagonal y reordenamiento")
    p.texto("Condición suficiente de convergencia: |a_ii| > Σ|a_ij| (j ≠ i) en cada renglón.")
    p.tabla(df_dominancia(A))
    est, _ = dominancia(A)
    if est:
        p.aviso("ok", "El sistema ya es estrictamente diagonalmente dominante: no se reordena.")
        pr, tipo = (0, 1, 2), "estricta"
    else:
        pr, tipo = buscar_reordenamiento(A)
        if tipo is not None:
            p.aviso("warn", "El sistema NO es diagonalmente dominante. Se reordenan las ecuaciones "
                            "colocando en la diagonal el coeficiente dominante.")
        else:
            p.aviso("warn", "No existe ningún reordenamiento diagonalmente dominante: "
                            "El método podría no converger. Se continúa con el mejor reordenamiento posible.")
    A2 = [A[i][:] for i in pr]
    b2 = [b[i] for i in pr]
    if tuple(pr) != (0, 1, 2):
        mapa = ", ".join(f"nueva ecuación {i + 1} = ecuación original {pr[i] + 1}" for i in range(n))
        p.texto(f"Reordenamiento aplicado: {mapa}.")
        p.eq(latex_sistema(A2, b2), sistema_plano(A2, b2))
        p.tabla(df_dominancia(A2))
        if tipo == "débil":
            p.aviso("info", "El sistema reordenado es débilmente dominante (≥ en todos los renglones y > en al menos uno).")
        elif tipo == "estricta":
            p.aviso("ok", "El sistema reordenado es estrictamente diagonalmente dominante ✓.")

    # El despeje requiere diagonal distinta de cero
    if any(abs(A2[i][i]) < EPS for i in range(n)):
        res["error"] = "Hay un 0 en la diagonal tras reordenar: no se puede despejar (división entre cero)."
        p.aviso("error", res["error"])
        return res

    # ---------------- Paso 2: fórmulas de iteración --------------------------
    p = nuevo(pasos, "Despeje de las variables (fórmulas de iteración)")
    p.texto("Se despeja x del renglón 1, y del renglón 2 y z del renglón 3. "
            "Cada fórmula usa los valores ya actualizados en la misma iteración (índice k+1).")
    tex_f, txt_f = [], []
    for i in range(n):
        t, s = formula_despeje(A2, b2, i)
        tex_f.append(t)
        txt_f.append(s)
    p.eq(r"\begin{aligned}" + r" \\ ".join(f.replace("=", "&=", 1) for f in tex_f) + r"\end{aligned}",
         "\n".join(txt_f))

    # ---------------- Paso 3: configuración ----------------------------------
    p = nuevo(pasos, "Condiciones iniciales y criterio de paro")
    p.eq(r"x^{(0)} = " + n6(x0[0]) + r",\quad y^{(0)} = " + n6(x0[1]) + r",\quad z^{(0)} = " + n6(x0[2]),
         f"x(0) = {n6(x0[0])}, y(0) = {n6(x0[1])}, z(0) = {n6(x0[2])}")
    p.texto(f"Tolerancia ε = {tol:g} · Máximo de iteraciones = {max_iter}. "
            "Se detiene cuando |Δx|, |Δy| y |Δz| son menores que ε, o al llegar al máximo.")

    # ---------------- Paso 4: primera iteración explícita --------------------
    p4 = nuevo(pasos, "Primera iteración con sustitución explícita")
    p5 = None   # se crea después del ciclo para conservar el orden

    # ---------------- Ciclo de iteraciones -----------------------------------
    x = list(x0)
    tabla = [{"k": "0", "x(k)": n6(x[0]), "y(k)": n6(x[1]), "z(k)": n6(x[2]),
              "|Δx|": "—", "|Δy|": "—", "|Δz|": "—"}]
    ks, err_max, err_var = [], [], []
    motivo, k_final = None, 0

    for k in range(1, max_iter + 1):
        x_ant = x[:]                 # valores de la iteración anterior (k-1)
        x_nuevo = x[:]               # aquí se van "pisando" con los valores más recientes
        for i in range(n):
            s = b2[i]
            expr = n6(b2[i])
            for j in range(n):
                if j != i:
                    s -= A2[i][j] * x_nuevo[j]     # x_nuevo[j] ya es k+1 si j<i (Gauss-Seidel)
                    expr += f" - ({n6(A2[i][j])})({n6(x_nuevo[j])})"
            x_nuevo[i] = s / A2[i][i]
            if k == 1:                              # sustitución explícita (1ª iteración)
                p4.eq(f"{VARS[i]}^{{(1)}} = " + r"\dfrac{" + expr + "}{" + n6(A2[i][i]) + "} = " + n6(x_nuevo[i]),
                      f"{VARS[i]}(1) = ({expr}) / {n6(A2[i][i])} = {n6(x_nuevo[i])}")
        errores = [abs(x_nuevo[i] - x_ant[i]) for i in range(n)]
        k_final = k
        # Protección contra divergencia (desbordamiento / NaN)
        if not all(math.isfinite(v) for v in x_nuevo + errores):
            motivo = "diverge"
            break
        x = x_nuevo
        tabla.append({"k": str(k), "x(k)": n6(x[0]), "y(k)": n6(x[1]), "z(k)": n6(x[2]),
                      "|Δx|": n6(errores[0]), "|Δy|": n6(errores[1]), "|Δz|": n6(errores[2])})
        ks.append(k)
        err_var.append(errores)
        err_max.append(max(errores))
        if max(errores) < tol:                       # criterio de paro por tolerancia
            motivo = "tol"
            break
    if motivo is None:
        motivo = "max"

    p4.texto("Se parte de x⁽⁰⁾ = (" + ", ".join(n6(v) for v in x0) + ") y se sustituye en las fórmulas de iteración.")

    # ---------------- Paso 5: tabla de iteraciones ---------------------------
    p5 = nuevo(pasos, "Tabla de iteraciones")
    p5.tabla(pd.DataFrame(tabla))

    # ---------------- Paso 6: criterio de paro -------------------------------
    p6 = nuevo(pasos, "Criterio de paro")
    if motivo == "tol":
        p6.aviso("ok", f"El método se detuvo en la iteración k = {k_final}: los tres errores "
                       f"|Δx|, |Δy|, |Δz| son menores que ε = {tol:g}.")
    elif motivo == "max":
        p6.aviso("warn", f"Se alcanzó el máximo de {max_iter} iteraciones sin cumplir la tolerancia "
                         f"ε = {tol:g}. El resultado es aproximado (el método podría no converger).")
    else:
        res["error"] = (f"El método DIVERGE (los valores se desbordaron en la iteración {k_final}). "
                        "El sistema no es adecuado para Gauss-Seidel.")
        p6.aviso("error", res["error"])
        res["extra"] = {"ks": ks, "err_max": err_max, "err_var": err_var, "tol": tol,
                        "detalle": "Diverge"}
        return res

    res["solucion"] = x
    res["extra"] = {"ks": ks, "err_max": err_max, "err_var": err_var, "tol": tol,
                    "convergio": motivo == "tol", "iteraciones": k_final,
                    "detalle": f"{k_final} iteraciones (ε = {tol:g})" +
                               ("" if motivo == "tol" else " — NO alcanzó la tolerancia")}
    return res


# ==============================================================================
# 6. MÉTODO C: KRYLOV
# ==============================================================================
def mult_detalle(A, v, nombre_tex, nombre_txt):
    """Multiplica A·v mostrando cada componente. Devuelve (vector, [(latex, texto)])."""
    out, lineas = [], []
    for i in range(3):
        s = 0.0
        partes = []
        for j in range(3):
            s += A[i][j] * v[j]
            partes.append(f"({n6(A[i][j])})({n6(v[j])})")
        out.append(s)
        lineas.append((f"({nombre_tex})_{i + 1} = " + " + ".join(partes) + " = " + n6(s),
                       f"({nombre_txt}){i + 1} = " + " + ".join(partes) + " = " + n6(s)))
    return out, lineas


def metodo_krylov(A, b):
    """
    Método de Krylov. Por Cayley-Hamilton: A³ + c1·A² + c2·A + c3·I = 0.
    Multiplicando por b:   A³b + c1·A²b + c2·Ab + c3·b = 0   (sistema para c1, c2, c3).
    Multiplicando por x (Ax = b):  c3·x = -(A²b + c1·Ab + c2·b)  →  x = -(A²b + c1·Ab + c2·b) / c3.
    """
    n = 3
    res = resultado_base(M_KR)
    pasos = res["pasos"]

    # ---------------- Paso 1: planteamiento ----------------------------------
    p = nuevo(pasos, "Planteamiento del sistema")
    p.texto("Se resuelve **A·x = b**. Con el teorema de Cayley-Hamilton la matriz A satisface su propio "
            "polinomio característico, lo que permite despejar x a partir de los vectores de Krylov "
            "b, Ab, A²b y A³b.")
    p.mats(("A", A, None), ("b", [[v] for v in b], None))
    p.eq(r"A\,\mathbf{x} = \mathbf{b} \;\Rightarrow\; " + latex_matrix(A) +
         r"\begin{bmatrix} x \\ y \\ z \end{bmatrix} = " + latex_matrix([[v] for v in b]),
         "A x = b, con x = [x, y, z]^T")

    # ---------------- Paso 2: vectores de Krylov -----------------------------
    p = nuevo(pasos, "Cálculo de los vectores Ab, A²b y A³b")
    v0 = list(b)
    v1, l1 = mult_detalle(A, v0, "Ab", "Ab")
    p.texto("**Ab = A · b**")
    for t, s in l1:
        p.eq(t, s)
    v2, l2 = mult_detalle(A, v1, r"A^2 b", "A^2 b")
    p.texto("**A²b = A · (Ab)**")
    for t, s in l2:
        p.eq(t, s)
    v3, l3 = mult_detalle(A, v2, r"A^3 b", "A^3 b")
    p.texto("**A³b = A · (A²b)**")
    for t, s in l3:
        p.eq(t, s)
    p.mats(("Ab", [[v] for v in v1], None), ("A^2 b", [[v] for v in v2], None),
           ("A^3 b", [[v] for v in v3], None))

    # ---------------- Paso 3: relación de Krylov -----------------------------
    p = nuevo(pasos, "Relación de Krylov: sistema para c₁, c₂, c₃")
    p.eq(r"A^3 b + c_1 A^2 b + c_2 A b + c_3 b = 0",
         "A^3 b + c1 A^2 b + c2 A b + c3 b = 0")
    K = [[v2[i], v1[i], v0[i]] for i in range(n)]       # columnas: A²b, Ab, b
    rhs = [-v3[i] for i in range(n)]                    # lado derecho: -A³b
    tex, txt = sistema_generico(K, rhs, ["c_1", "c_2", "c_3"], ["c1", "c2", "c3"])
    p.texto("Reacomodando: [A²b | Ab | b] · [c₁, c₂, c₃]ᵀ = −A³b")
    p.eq(tex, txt)

    # ---------------- Paso 4: resolver subsistema (regla de Cramer) ----------
    p = nuevo(pasos, "Resolución del subsistema para c₁, c₂, c₃ (determinantes)")
    dK = det3(K)
    escala = max(abs(v) for fila in K for v in fila)
    if abs(dK) <= 1e-10 * (escala ** 3 if escala > 0 else 1.0):
        res["error"] = ("Los vectores b, Ab y A²b son linealmente dependientes (det K = 0): "
                        "el método de Krylov con este vector b no puede determinar c₁, c₂, c₃. "
                        "Usa otro método (LU o Gauss-Seidel).")
        p.eq(r"\det(K) = " + n6(dK), f"det(K) = {n6(dK)}")
        p.aviso("error", res["error"])
        return res
    p.eq(r"\det(K) = " + n6(dK), f"det(K) = {n6(dK)}")
    c = []
    mats_k = []
    for j in range(3):
        Kj = copia(K)
        for i in range(n):
            Kj[i][j] = rhs[i]                       # se reemplaza la columna j por -A³b
        dj = det3(Kj)
        c.append(dj / dK)
        mats_k.append((f"K_{j + 1}", Kj, None))
    p.mats(*mats_k[:2])
    p.mats(mats_k[2])
    for j in range(3):
        dj = det3(mats_k[j][1])
        p.eq(f"c_{j + 1} = " + r"\dfrac{\det(K_" + str(j + 1) + r")}{\det(K)} = \dfrac{" + n6(dj) + "}{" + n6(dK) + "} = " + n6(c[j]),
             f"c{j + 1} = det(K{j + 1}) / det(K) = {n6(dj)} / {n6(dK)} = {n6(c[j])}")
    p.aviso("info", f"Comprobación: c3 debe ser igual a −det(A). −det(A) = {n6(-det3(A))} ; c3 = {n6(c[2])}.")
    if abs(c[2]) < EPS:
        res["error"] = "c3 = 0: la matriz A es singular, no se puede despejar x."
        p.aviso("error", res["error"])
        return res

    # ---------------- Paso 5: despeje de x -----------------------------------
    p = nuevo(pasos, "Despeje de x")
    p.eq(r"\mathbf{x} = -\dfrac{A^2 b + c_1 A b + c_2 b}{c_3}",
         "x = -(A^2 b + c1 A b + c2 b) / c3")

    # ---------------- Paso 6: sustitución y resultado ------------------------
    p = nuevo(pasos, "Sustitución y resultado final")
    X = []
    for i in range(n):
        num = v2[i] + c[0] * v1[i] + c[1] * v0[i]
        X.append(-num / c[2])
        expr = f"{n6(v2[i])} + ({n6(c[0])})({n6(v1[i])}) + ({n6(c[1])})({n6(v0[i])})"
        p.eq(f"{VARS[i]} = -" + r"\dfrac{" + expr + "}{" + n6(c[2]) + "} = " + n6(X[i]),
             f"{VARS[i]} = -({expr}) / {n6(c[2])} = {n6(X[i])}")
    p.mats(("x", [[v] for v in X], None))

    res["solucion"] = X
    res["extra"] = {"detalle": f"c1 = {n6(c[0])}, c2 = {n6(c[1])}, c3 = {n6(c[2])}", "c": c}
    return res


# ==============================================================================
# 7. COMPROBACIÓN, COMPARACIÓN Y GRÁFICAS
# ==============================================================================
def sustitucion_str(fila, sol):
    """Texto de la sustitución, ej. '1(1.000000) + 2(2.000000) + 3(3.000000)'."""
    out = ""
    for j, c in enumerate(fila):
        if abs(c) < EPS:
            continue
        term = f"{g(abs(c))}({n6(sol[j])})"
        if out == "":
            out = ("-" if c < 0 else "") + term
        else:
            out += (" - " if c < 0 else " + ") + term
    return out if out else "0"


def tabla_comprobacion(A, b, sol, tol_check=TOL_COMPROBACION):
    """Tabla [Ecuación | Sustitución | Resultado | ✓] al sustituir la solución."""
    filas = []
    for i in range(3):
        lhs = sum(A[i][j] * sol[j] for j in range(3))
        ok = abs(lhs - b[i]) <= tol_check
        filas.append({"Ecuación": ecuacion_str(A[i], b[i]),
                      "Sustitución": sustitucion_str(A[i], sol),
                      "Resultado": n6(lhs),
                      "✓": "✓" if ok else "✗"})
    return pd.DataFrame(filas)


def paso_resultado(A, b, sol):
    """Paso 'Resultado y comprobación' (se usa en el PDF)."""
    p = Paso("Resultado final y comprobación")
    p.eq(r"\begin{aligned} x &= " + n6(sol[0]) + r" \\ y &= " + n6(sol[1]) + r" \\ z &= " + n6(sol[2]) + r"\end{aligned}",
         f"x = {n6(sol[0])}\ny = {n6(sol[1])}\nz = {n6(sol[2])}")
    p.tabla(tabla_comprobacion(A, b, sol))
    return p


def ejecutar_los_tres(A, b, cfg):
    """Corre LU, Gauss-Seidel y Krylov. Cada uno se protege con try/except."""
    resultados = []
    for nombre in METODOS:
        try:
            if nombre == M_LU:
                r = metodo_lu(A, b)
            elif nombre == M_GS:
                r = metodo_gauss_seidel(A, b, cfg["x0"], cfg["tol"], cfg["max_iter"])
            else:
                r = metodo_krylov(A, b)
        except Exception as e:                    # nunca debe tronar la aplicación
            r = resultado_base(nombre)
            r["error"] = f"Error inesperado: {e}"
        resultados.append(r)
    return resultados


def tabla_comparativa(resultados):
    """Tabla comparativa de los tres métodos (referencia: el primer resultado válido, LU)."""
    ref = next((r["solucion"] for r in resultados if r["solucion"] is not None), None)
    filas = []
    for r in resultados:
        sol = r["solucion"]
        if sol is None or ref is None:
            filas.append({"Método": r["metodo"], "x": "—", "y": "—", "z": "—",
                          "Detalle": r["error"] or "Sin resultado",
                          "Dif. máx. vs LU": "—", "¿Coincide?": "✗"})
        else:
            dif = max(abs(sol[i] - ref[i]) for i in range(3))
            filas.append({"Método": r["metodo"], "x": n6(sol[0]), "y": n6(sol[1]), "z": n6(sol[2]),
                          "Detalle": r["extra"].get("detalle", ""),
                          "Dif. máx. vs LU": n6(dif),
                          "¿Coincide?": "✓" if dif < TOL_COINCIDENCIA else "✗"})
    return pd.DataFrame(filas)


def figura_planos(A, b, sol):
    """Gráfica 3D interactiva: los 3 planos y la solución como punto rojo."""
    r = max(3.0, 1.5 * max(abs(v) for v in sol))          # semi-rango de los ejes
    s = np.linspace(-r, r, 25)
    U_, V_ = np.meshgrid(s, s)
    fig = go.Figure()
    for i in range(3):
        # Se despeja la variable con mayor |coeficiente| (evita dividir entre 0 y planos verticales)
        idx = max(range(3), key=lambda j: abs(A[i][j]))
        o1, o2 = [j for j in range(3) if j != idx]
        P = sol[o1] + U_
        Q = sol[o2] + V_
        R = (b[i] - A[i][o1] * P - A[i][o2] * Q) / A[i][idx]
        coords = [None, None, None]
        coords[o1], coords[o2], coords[idx] = P, Q, R
        col = COLORES_PLANOS[i]
        fig.add_trace(go.Surface(x=coords[0], y=coords[1], z=coords[2], opacity=0.55,
                                 showscale=False, colorscale=[[0, col], [1, col]],
                                 name=f"Plano {i + 1}: {ecuacion_str(A[i], b[i])}",
                                 showlegend=True, hoverinfo="name"))
    fig.add_trace(go.Scatter3d(x=[sol[0]], y=[sol[1]], z=[sol[2]], mode="markers+text",
                               marker=dict(size=7, color="#D62828", line=dict(color="#7A1010", width=1)),
                               text=["Solución"], textposition="top center",
                               name=f"Solución ({n6(sol[0])}, {n6(sol[1])}, {n6(sol[2])})"))
    eje = dict(backgroundcolor=BEIGE_TARJETA, gridcolor=BEIGE_BORDE, zerolinecolor=BEIGE_ACENTO,
               color=CAFE_TEXTO)
    fig.update_layout(
        paper_bgcolor=BEIGE_FONDO, plot_bgcolor=BEIGE_FONDO, font=dict(color=CAFE_TEXTO),
        margin=dict(l=0, r=0, t=30, b=0), height=560,
        legend=dict(bgcolor="rgba(243,234,219,0.8)", bordercolor=BEIGE_BORDE, borderwidth=1,
                    yanchor="top", y=0.98, xanchor="left", x=0.01),
        scene=dict(xaxis=dict(title="x", range=[sol[0] - r, sol[0] + r], **eje),
                   yaxis=dict(title="y", range=[sol[1] - r, sol[1] + r], **eje),
                   zaxis=dict(title="z", range=[sol[2] - r, sol[2] + r], **eje),
                   aspectmode="cube"))
    return fig


def figura_convergencia(extra):
    """Gráfica 2D de convergencia de Gauss-Seidel: error vs iteración (escala log)."""
    ks, err_var, err_max, tol = extra["ks"], extra["err_var"], extra["err_max"], extra["tol"]
    clip = lambda v: max(v, 1e-16)           # la escala log no admite ceros
    fig = go.Figure()
    for idx, (nombre, col) in enumerate(zip(["|Δx|", "|Δy|", "|Δz|"], COLORES_PLANOS)):
        fig.add_trace(go.Scatter(x=ks, y=[clip(e[idx]) for e in err_var], mode="lines+markers",
                                 name=nombre, line=dict(color=col, width=1.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=ks, y=[clip(e) for e in err_max], mode="lines+markers",
                             name="Error máximo", line=dict(color="#6B5440", width=3), marker=dict(size=6)))
    fig.add_hline(y=tol, line_dash="dash", line_color="#D62828",
                  annotation_text=f"ε = {tol:g}", annotation_position="top right")
    fig.update_yaxes(type="log", title="Error", gridcolor=BEIGE_BORDE)
    fig.update_xaxes(title="Iteración k", gridcolor=BEIGE_BORDE)
    fig.update_layout(paper_bgcolor=BEIGE_FONDO, plot_bgcolor=BEIGE_TARJETA, font=dict(color=CAFE_TEXTO),
                      height=380, margin=dict(l=10, r=10, t=30, b=10), title="Convergencia: Error vs Iteración")
    return fig


def mostrar_plotly(fig, key):
    """Compatibilidad entre versiones de Streamlit (width='stretch' vs use_container_width)."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


# ==============================================================================
# 8. EXPORTAR A PDF (FPDF)
# ==============================================================================
_PDF_MAP = {"←": "<-", "→": "->", "↔": "<->", "✓": "OK", "✗": "X", "ε": "eps", "Δ": "Delta",
            "Σ": "Suma", "≈": "~", "−": "-", "—": "-", "–": "-", "≠": "!=", "≤": "<=", "≥": ">=",
            "…": "...", "“": '"', "”": '"', "‘": "'", "’": "'", "•": "-", "₁": "1", "₂": "2",
            "₃": "3", "⁽⁰⁾": "(0)", "ᵀ": "^T", "**": "", "`": ""}


def safe(t):
    """Adapta el texto a latin-1 (fuentes base de FPDF no soportan Unicode completo)."""
    t = str(t)
    for k, v in _PDF_MAP.items():
        t = t.replace(k, v)
    return t.encode("latin-1", "replace").decode("latin-1")


class PDFReporte(FPDF):
    """PDF con fondo beige y encabezado/pie de página."""

    def header(self):
        self.set_fill_color(250, 246, 239)
        self.rect(0, 0, self.w, self.h, "F")                       # fondo beige
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(139, 115, 85)
        self.cell(0, 6, safe("TecNM · Instituto Tecnológico de Colima — Métodos Numéricos (AEC-1046)"),
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(139, 115, 85)
        self.cell(0, 8, f"Página {self.page_no()}", align="C")


def _pdf_espacio(pdf, alto):
    """Salta de página si no cabe un bloque del alto indicado."""
    if pdf.get_y() + alto > pdf.h - pdf.b_margin:
        pdf.add_page()


def _pdf_texto(pdf, txt, size=10, estilo="", color=(74, 63, 53), fuente="Helvetica", h=5):
    pdf.set_font(fuente, estilo, size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, h, safe(txt), new_x="LMARGIN", new_y="NEXT")


def _pdf_mats(pdf, items):
    """Dibuja una o varias matrices como cuadrículas, una junto a otra si caben."""
    cw, rh, gap = 22, 6, 8
    disponible = pdf.w - pdf.l_margin - pdf.r_margin
    lineas, ancho = [[]], 0
    for it in items:
        ncols = len(it[1][0]) + (1 if it[2] is not None else 0)
        w = ncols * cw
        if lineas[-1] and ancho + w > disponible:
            lineas.append([])
            ancho = 0
        lineas[-1].append((it, w))
        ancho += w + gap
    for linea in lineas:
        alto = max(5 + len(it[1]) * rh for it, _ in linea) + 3
        _pdf_espacio(pdf, alto)
        y0, x = pdf.get_y(), pdf.l_margin
        for (nombre, M, aug), w in linea:
            pdf.set_xy(x, y0)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(107, 84, 64)
            pdf.cell(w, 5, safe(nombre) + " =", new_x="RIGHT", new_y="TOP")
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(74, 63, 53)
            for i, fila in enumerate(M):
                pdf.set_xy(x, y0 + 5 + i * rh)
                pdf.set_fill_color(255, 253, 248)
                for v in fila:
                    pdf.cell(cw, rh, n6(v), border=1, align="R", fill=True, new_x="RIGHT", new_y="TOP")
                if aug is not None:
                    pdf.set_fill_color(243, 234, 219)
                    pdf.cell(cw, rh, n6(aug[i]), border=1, align="R", fill=True, new_x="RIGHT", new_y="TOP")
            x += w + gap
        pdf.set_xy(pdf.l_margin, y0 + alto)


def _pdf_tabla(pdf, df):
    """Dibuja un DataFrame de texto como tabla con anchos proporcionales."""
    enc = [safe(c) for c in df.columns]
    datos = [[safe(v) for v in fila] for fila in df.values.tolist()]
    size = 8
    pdf.set_font("Helvetica", "", size)
    anchos = [max([pdf.get_string_width(enc[j])] + [pdf.get_string_width(f[j]) for f in datos]) + 5
              for j in range(len(enc))]
    disponible = pdf.w - pdf.l_margin - pdf.r_margin
    total = sum(anchos)
    if total > disponible:
        anchos = [a * disponible / total for a in anchos]
        size = 7
    rh = 6

    def ajustar(txt, ancho):
        pdf.set_font("Helvetica", "", size)
        while txt and pdf.get_string_width(txt) + 2 > ancho:
            txt = txt[:-1]
        return txt

    def fila_encabezado():
        pdf.set_font("Helvetica", "B", size)
        pdf.set_fill_color(233, 220, 198)
        pdf.set_text_color(74, 63, 53)
        for j, t in enumerate(enc):
            pdf.cell(anchos[j], rh, ajustar(t, anchos[j]), border=1, fill=True, align="C",
                     new_x="RIGHT", new_y="TOP")
        pdf.ln(rh)

    _pdf_espacio(pdf, rh * 2)
    fila_encabezado()
    for f in datos:
        if pdf.get_y() + rh > pdf.h - pdf.b_margin:
            pdf.add_page()
            fila_encabezado()
        pdf.set_font("Helvetica", "", size)
        pdf.set_fill_color(255, 253, 248)
        for j, t in enumerate(f):
            pdf.cell(anchos[j], rh, ajustar(t, anchos[j]), border=1, fill=True, align="C",
                     new_x="RIGHT", new_y="TOP")
        pdf.ln(rh)
    pdf.ln(2)


_COLOR_AVISO = {"info": (232, 240, 247), "ok": (226, 240, 226), "warn": (252, 240, 210), "error": (250, 222, 216)}


def _pdf_paso(pdf, paso):
    """Dibuja un Paso completo en el PDF."""
    _pdf_espacio(pdf, 20)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(233, 220, 198)
    pdf.set_text_color(107, 84, 64)
    pdf.multi_cell(0, 7, safe(paso.titulo), fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    for blq in paso.bloques:
        tipo = blq[0]
        if tipo == "texto":
            _pdf_texto(pdf, blq[1])
            pdf.ln(1)
        elif tipo == "eq":
            _pdf_texto(pdf, blq[2], size=9, fuente="Courier")
            pdf.ln(1)
        elif tipo == "mats":
            _pdf_mats(pdf, blq[1])
        elif tipo == "tabla":
            _pdf_tabla(pdf, blq[1])
        elif tipo == "aviso":
            pdf.set_fill_color(*_COLOR_AVISO[blq[1]])
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(74, 63, 53)
            pdf.multi_cell(0, 5.5, safe(blq[2]), fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
    pdf.ln(3)


def generar_pdf(A, b, resultados, comparacion=None):
    """Genera el PDF (bytes) con el procedimiento completo de los métodos indicados."""
    pdf = PDFReporte()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    _pdf_texto(pdf, "Solucionador de Sistemas de Ecuaciones Lineales 3x3", size=16, estilo="B",
               color=(107, 84, 64), h=9)
    _pdf_texto(pdf, "Materia: Métodos Numéricos (AEC-1046)\n"
                    "Docente: Mtro. Ricardo Olivera Guerrero\n"
                    "Integrantes: Karol Gabriel Cobian Zamora, Kalel Isahid Ramos Espinoza y Iker Orlando Salas Jimenes\n"
                    f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", size=10)
    pdf.ln(3)
    _pdf_texto(pdf, "Sistema a resolver", size=12, estilo="B", color=(107, 84, 64), h=7)
    _pdf_texto(pdf, sistema_plano(A, b), size=10, fuente="Courier")
    _pdf_mats(pdf, [("A", A, b)])
    d = det3(A)
    _pdf_texto(pdf, f"det(A) = {n6(d)}  (distinto de cero: el sistema tiene solución única)", size=9)
    pdf.ln(3)

    for r in resultados:
        pdf.add_page()
        _pdf_texto(pdf, f"Método: {r['metodo']}", size=14, estilo="B", color=(107, 84, 64), h=8)
        pdf.ln(1)
        for paso in r["pasos"]:
            _pdf_paso(pdf, paso)
        if r["error"]:
            _pdf_paso(pdf, _paso_error(r["error"]))
        elif r["solucion"] is not None:
            _pdf_paso(pdf, paso_resultado(A, b, r["solucion"]))
    if comparacion is not None:
        pdf.add_page()
        p = Paso("Comparación de métodos")
        p.tabla(comparacion)
        _pdf_paso(pdf, p)
    _pdf_texto(pdf, "Nota: las gráficas interactivas (planos 3D y convergencia) se consultan en la aplicación.",
               size=8, estilo="I")
    return bytes(pdf.output())


def _paso_error(msg):
    p = Paso("Resultado")
    p.aviso("error", msg)
    return p


# ==============================================================================
# 9. INTERFAZ STREAMLIT — ESTILO Y UTILIDADES
# ==============================================================================
def inyectar_css():
    """Tema beige pastel minimalista (fondos, botones, tarjetas, pestañas, campos)."""
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {BEIGE_FONDO}; color: {CAFE_TEXTO}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    h1, h2, h3, h4, h5 {{ color: {CAFE_TITULO}; }}
    /* Botones */
    div.stButton > button, div.stDownloadButton > button {{
        background-color: #EADFCB; color: {CAFE_TEXTO}; border: 1px solid #D8C4A5;
        border-radius: 10px; transition: all .15s ease-in-out;
    }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
        background-color: #DCC8A6; border-color: #B08968; color: #3A2F26;
    }}
    div.stButton > button:disabled {{ background-color: #F1EADD; color: #B5A58F; border-color: #E8DDC8; }}
    /* Tarjetas colapsables */
    div[data-testid="stExpander"] {{
        background-color: {BEIGE_TARJETA}; border: 1px solid {BEIGE_BORDE}; border-radius: 12px;
    }}
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #EFE4D0; border-radius: 10px 10px 0 0; padding: 8px 18px; color: {CAFE_TEXTO};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {BEIGE_ACENTO}; color: white; }}
    /* Campos de entrada */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{
        background-color: #FFFDF8; border-color: {BEIGE_BORDE};
    }}
    /* Tarjetas propias */
    .tarjeta {{
        background-color: {BEIGE_TARJETA}; border: 1px solid {BEIGE_BORDE}; border-radius: 14px;
        padding: 18px 22px; text-align: center; margin-bottom: 10px;
    }}
    .tarjeta .icono {{ font-size: 42px; }}
    .tarjeta .titulo {{ font-weight: 700; font-size: 18px; color: {CAFE_TITULO}; }}
    .tarjeta .sub {{ font-size: 13px; color: #8B7355; }}
    [data-testid="stMetric"] {{
        background-color: {BEIGE_TARJETA}; border: 1px solid {BEIGE_BORDE}; border-radius: 12px; padding: 10px 14px;
    }}
    </style>
    """, unsafe_allow_html=True)


def mostrar_aviso(tipo, msg):
    {"info": st.info, "ok": st.success, "warn": st.warning, "error": st.error}[tipo](msg)


def render_paso(paso):
    """Dibuja un Paso en pantalla."""
    for blq in paso.bloques:
        tipo = blq[0]
        if tipo == "texto":
            st.markdown(blq[1])
        elif tipo == "eq":
            st.latex(blq[1])
        elif tipo == "mats":
            partes = []
            for nombre, M, aug in blq[1]:
                cuerpo = latex_aug(M, aug) if aug is not None else latex_matrix(M)
                partes.append(f"{nombre_latex(nombre)} = {cuerpo}")
            st.latex(r" \quad ".join(partes))
        elif tipo == "tabla":
            st.dataframe(blq[1], hide_index=True)
        elif tipo == "aviso":
            mostrar_aviso(blq[1], blq[2])


# ------------------------- callbacks de botones ------------------------------
def cb_cargar_ejemplo():
    """Carga el sistema de prueba elegido en la cuadrícula."""
    ej = EJEMPLOS[st.session_state["ejemplo_sel"]]
    for i in range(3):
        for j in range(3):
            st.session_state[f"c_{i}_{j}"] = float(ej["A"][i][j])
        st.session_state[f"c_{i}_3"] = float(ej["b"][i])


def cb_limpiar():
    """Vacía toda la cuadrícula."""
    for i in range(3):
        for j in range(4):
            st.session_state[f"c_{i}_{j}"] = None
    st.session_state["modo"] = "todo"
    st.session_state["paso"] = 1


def cb_cambio_metodo():
    """Al cambiar de método se preselecciona el ejemplo sugerido para ese método."""
    st.session_state["ejemplo_sel"] = EJEMPLO_POR_METODO[st.session_state["metodo"]]


def cb_nav(accion):
    """Navegación paso a paso (usa st.session_state)."""
    if accion == "reiniciar":
        st.session_state["modo"], st.session_state["paso"] = "pasos", 1
    elif accion == "anterior":
        st.session_state["paso"] = max(1, st.session_state["paso"] - 1)
    elif accion == "siguiente":
        st.session_state["paso"] += 1          # se recorta al total en pantalla
    elif accion == "todo":
        st.session_state["modo"] = "todo"


def inicializar_estado():
    st.session_state.setdefault("metodo", M_LU)
    st.session_state.setdefault("ejemplo_sel", EJEMPLO_POR_METODO[M_LU])
    st.session_state.setdefault("modo", "todo")
    st.session_state.setdefault("paso", 1)
    st.session_state.setdefault("firma", None)
    st.session_state.setdefault("comparar", False)
    st.session_state.setdefault("gs_cfg", {"x0": [0.0, 0.0, 0.0], "tol": 1e-6, "max_iter": 50})


# ==============================================================================
# 10. PANTALLAS
# ==============================================================================
def pantalla_inicio():
    """Pantalla de inicio: título, logotipos, materia, docente e integrantes."""
    st.markdown("<h1 style='text-align:center;'>Solucionador de Sistemas de Ecuaciones Lineales 3x3</h1>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    # Logotipos: si existen los archivos logo_tecnm.png / logo_itcolima.png se usan; si no, emojis.
    for col, archivo, icono, titulo, sub in (
            (c1, "logo_tecnm.png", "🎓", "TecNM", "Tecnológico Nacional de México"),
            (c2, "logo_itcolima.png", "🏛️", "Instituto Tecnológico de Colima", "Villa de Álvarez, Colima")):
        with col:
            if os.path.exists(archivo):
                st.image(archivo, width=140)
            else:
                st.markdown(f"<div class='tarjeta'><div class='icono'>{icono}</div>"
                            f"<div class='titulo'>{titulo}</div><div class='sub'>{sub}</div></div>",
                            unsafe_allow_html=True)
    st.markdown(f"""
    <div class='tarjeta' style='text-align:left;'>
      <p><b>📘 Materia:</b> Métodos Numéricos (AEC-1046)</p>
      <p><b>👨‍🏫 Docente:</b> Mtro. Ricardo Olivera Guerrero</p>
      <p style='margin-bottom:0'><b>👥 Integrantes:</b> Karol Gabriel Cobian Zamora · Kalel Isahid Ramos Espinoza · Iker Orlando Salas Jiménez</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### ¿Cómo usar la aplicación?")
    st.markdown(
        "1. Ve a la pestaña **✏️ Resolver**, captura la matriz A y el vector B (o carga un ejemplo).\n"
        "2. Elige el método: **LU (Doolittle)**, **Gauss-Seidel** o **Krylov**.\n"
        "3. Navega el procedimiento **paso a paso** o pulsa **Mostrar todo**.\n"
        "4. Revisa la solución, la comprobación y las gráficas; exporta el procedimiento a **PDF**.\n"
        "5. En **⚖️ Comparar métodos** corre los tres algoritmos y verifica que lleguen a lo mismo.")


def cuadricula_captura():
    """Cuadrícula 3x4 de number_input. Devuelve (A, b, lista_de_campos_vacíos)."""
    anchos = [0.9, 1, 1, 1, 1.2]
    cab = st.columns(anchos)
    for col, txt in zip(cab, ["", "Coef. de x", "Coef. de y", "Coef. de z", "Término indep. (b)"]):
        col.markdown(f"**{txt}**")
    A = [[None] * 3 for _ in range(3)]
    b = [None] * 3
    vacios = []
    for i in range(3):
        cols = st.columns(anchos)
        cols[0].markdown(f"**Ecuación {i + 1}**")
        for j in range(4):
            val = cols[j + 1].number_input(f"Ecuación {i + 1}, columna {j + 1}", value=None, step=1.0,
                                           format="%g", key=f"c_{i}_{j}", placeholder="0",
                                           label_visibility="collapsed")
            if val is None:
                vacios.append((i, j))
            if j < 3:
                A[i][j] = val
            else:
                b[i] = val
    return A, b, vacios


def leer_config_gs():
    """Entradas de configuración de Gauss-Seidel. Devuelve dict o None si hay error."""
    with st.container(border=True):
        st.markdown("**⚙️ Configuración de Gauss-Seidel**")
        c = st.columns(3)
        x0 = [c[i].number_input(f"Condición inicial {VARS[i]}⁽⁰⁾", value=0.0, step=1.0, format="%g",
                                key=f"gs_x0_{i}") for i in range(3)]
        c2 = st.columns(2)
        tol_txt = c2[0].text_input("Tolerancia ε (ej. 1e-6)", value="1e-6", key="gs_tol")
        max_iter = c2[1].number_input("Máximo de iteraciones", min_value=1, max_value=2000, value=50,
                                      step=1, key="gs_max")
    # Validación de las entradas (sin dejar que la app truene)
    try:
        if any(v is None for v in x0):
            raise ValueError("Las condiciones iniciales no pueden estar vacías.")
        tol = float(tol_txt.strip().replace(",", "."))
        if not (tol > 0 and math.isfinite(tol)):
            raise ValueError("La tolerancia debe ser un número positivo.")
        if max_iter is None:
            raise ValueError("Indica el máximo de iteraciones.")
    except ValueError as e:
        msg = str(e)
        if "could not convert" in msg:
            msg = "La tolerancia no es un número válido (usa por ejemplo 1e-6 o 0.000001)."
        st.error(f"⚠️ Configuración inválida: {msg}")
        return None
    cfg = {"x0": [float(v) for v in x0], "tol": tol, "max_iter": int(max_iter)}
    st.session_state["gs_cfg"] = cfg        # la pestaña de comparación reutiliza esta configuración
    return cfg


def pantalla_resolver():
    """Captura/validación, selección de método, navegación paso a paso, resultados y gráficas."""
    st.header("✏️ Captura y validación")

    # ----- Ejemplos y limpieza -----
    c1, c2, c3 = st.columns([3, 1.2, 1])
    c1.selectbox("Sistema de ejemplo", list(EJEMPLOS.keys()), key="ejemplo_sel")
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    c2.button("📥 Cargar Ejemplo", on_click=cb_cargar_ejemplo)
    c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    c3.button("🧹 Limpiar", on_click=cb_limpiar)

    # ----- Cuadrícula de captura -----
    A, b, vacios = cuadricula_captura()

    # Validación 1: campos vacíos
    if vacios:
        if len(vacios) == 12:
            st.info("Captura los coeficientes de A y el vector B, o usa **Cargar Ejemplo**.")
        else:
            st.warning(f"⚠️ Hay {len(vacios)} campo(s) vacío(s). Completa todos los campos "
                       "(usa 0 donde no haya término) para poder resolver.")
        return

    # Vista dinámica del sistema
    st.markdown("##### Sistema capturado")
    v1, v2 = st.columns(2)
    with v1:
        st.latex(latex_sistema(A, b))
    with v2:
        st.latex(forma_matricial_latex(A, b))

    # Validación 2: determinante
    singular, d = es_singular(A)
    if singular:
        st.error("⚠️ **El sistema no tiene solución única** (det(A) = 0). "
                 "Corrige los coeficientes para poder resolverlo.")
        return
    st.success(f"det(A) = {n6(d)} ≠ 0 → el sistema tiene solución única.")

    # ----- Selección de método -----
    st.header("🧭 Método y procedimiento")
    metodo = st.radio("Método numérico", METODOS, horizontal=True, key="metodo",
                      on_change=cb_cambio_metodo)
    cfg = leer_config_gs() if metodo == M_GS else None
    if metodo == M_GS and cfg is None:
        return

    # ----- Ejecución protegida del método -----
    try:
        if metodo == M_LU:
            res = metodo_lu(A, b)
        elif metodo == M_GS:
            res = metodo_gauss_seidel(A, b, cfg["x0"], cfg["tol"], cfg["max_iter"])
        else:
            res = metodo_krylov(A, b)
    except Exception as e:      # cualquier error inesperado se muestra sin cerrar la app
        st.error(f"❌ No se pudo completar el método: {e}. Revisa los datos e inténtalo de nuevo.")
        return

    pasos = res["pasos"]
    N = len(pasos)

    # Si cambió el sistema/método/configuración se reinicia la navegación
    firma = (metodo, tuple(tuple(f) for f in A), tuple(b),
             None if cfg is None else (tuple(cfg["x0"]), cfg["tol"], cfg["max_iter"]))
    if st.session_state["firma"] != firma:
        st.session_state["firma"] = firma
        st.session_state["modo"] = "todo"
        st.session_state["paso"] = 1
    st.session_state["paso"] = min(max(1, st.session_state["paso"]), N)
    modo, paso_act = st.session_state["modo"], st.session_state["paso"]

    # ----- Botones de navegación -----
    nb = st.columns(4)
    nb[0].button("🔁 Paso a paso", on_click=cb_nav, args=("reiniciar",),
                 disabled=(modo == "pasos" and paso_act == 1))
    nb[1].button("⬅️ Paso anterior", on_click=cb_nav, args=("anterior",),
                 disabled=(modo == "todo" or paso_act <= 1))
    nb[2].button("Siguiente paso ➡️", on_click=cb_nav, args=("siguiente",),
                 disabled=(modo == "todo" or paso_act >= N))
    nb[3].button("📖 Mostrar todo", on_click=cb_nav, args=("todo",), disabled=(modo == "todo"))

    visibles = N if modo == "todo" else paso_act
    st.progress(visibles / N, text=f"Mostrando {visibles} de {N} pasos")

    # ----- Pasos en tarjetas colapsables -----
    for idx in range(visibles):
        abierto = (modo == "todo") or (idx == visibles - 1)
        with st.expander(pasos[idx].titulo, expanded=abierto):
            render_paso(pasos[idx])

    # El resultado se muestra al terminar el recorrido
    if not (modo == "todo" or paso_act == N):
        return
    mostrar_resultados(A, b, res)


def mostrar_resultados(A, b, res):
    """Resultado final, comprobación, gráficas y exportación a PDF."""
    st.header("✅ Resultado y comprobación")
    if res["error"]:
        st.error(f"❌ {res['error']}")
        return
    sol = res["solucion"]
    m = st.columns(3)
    for i in range(3):
        m[i].metric(VARS[i], n6(sol[i]))
    st.latex(r"(x, y, z) = (" + ", ".join(n6(v) for v in sol) + ")")

    st.markdown("##### Comprobación: sustitución en las ecuaciones originales")
    df = tabla_comprobacion(A, b, sol)
    st.dataframe(df, hide_index=True)
    if (df["✓"] == "✓").all():
        st.success("✓ La solución satisface las 3 ecuaciones (tolerancia 1e-4).")
    else:
        st.warning("La solución no satisface todas las ecuaciones dentro de la tolerancia.")

    st.markdown("##### Gráfica 3D: intersección de los tres planos")
    mostrar_plotly(figura_planos(A, b, sol), "plot3d")
    if res["metodo"] == M_GS and res["extra"].get("ks"):
        st.markdown("##### Gráfica de convergencia (Gauss-Seidel)")
        mostrar_plotly(figura_convergencia(res["extra"]), "plot_conv")

    st.markdown("##### Exportar")
    try:
        pdf = generar_pdf(A, b, [res])
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in res["metodo"]).strip("_")
        st.download_button("📄 Exportar procedimiento a PDF", data=pdf,
                           file_name=f"procedimiento_{slug}.pdf", mime="application/pdf",
                           key="pdf_metodo")
    except Exception as e:
        st.error(f"No se pudo generar el PDF: {e}")


def pantalla_comparar():
    """Puntos extra: corre los 3 métodos y compara; PDF con todo."""
    st.header("⚖️ Comparación de métodos")
    st.caption("Ejecuta LU, Gauss-Seidel y Krylov sobre el sistema capturado y verifica que lleguen a la misma solución.")

    # Se lee el sistema desde el estado (los campos se dibujan en la pestaña Resolver)
    A, b, vacios = [[None] * 3 for _ in range(3)], [None] * 3, []
    for i in range(3):
        for j in range(4):
            v = st.session_state.get(f"c_{i}_{j}")
            if v is None:
                vacios.append((i, j))
            elif j < 3:
                A[i][j] = v
            else:
                b[i] = v
    if vacios:
        st.info("Primero captura un sistema completo en la pestaña **✏️ Resolver** (o carga un ejemplo).")
        return
    singular, _ = es_singular(A)
    if singular:
        st.error("⚠️ El sistema no tiene solución única (det(A) = 0).")
        return

    st.latex(latex_sistema(A, b))
    cfg = st.session_state["gs_cfg"]
    st.caption(f"Gauss-Seidel usa: x⁽⁰⁾ = ({', '.join(n6(v) for v in cfg['x0'])}), "
               f"ε = {cfg['tol']:g}, máx. {cfg['max_iter']} iteraciones.")
    if st.button("▶️ Ejecutar comparación"):
        st.session_state["comparar"] = True
    if not st.session_state["comparar"]:
        return

    resultados = ejecutar_los_tres(A, b, cfg)
    comp = tabla_comparativa(resultados)
    st.dataframe(comp, hide_index=True)
    con_sol = [r for r in resultados if r["solucion"] is not None]
    if len(con_sol) == 3 and (comp["¿Coincide?"] == "✓").all():
        st.success("✓ Los tres métodos llegan a la misma solución.")
    else:
        for r, ok in zip(resultados, comp["¿Coincide?"]):
            if r["solucion"] is None:
                st.warning(f"**{r['metodo']}**: {r['error']}")
            elif ok != "✓":
                st.warning(f"**{r['metodo']}** no coincide con el resto: {r['extra'].get('detalle', '')}.")
    try:
        pdf = generar_pdf(A, b, resultados, comparacion=comp)
        st.download_button("📄 Exportar PDF completo (3 métodos + comparación)", data=pdf,
                           file_name="procedimiento_completo_3_metodos.pdf", mime="application/pdf",
                           key="pdf_todos")
    except Exception as e:
        st.error(f"No se pudo generar el PDF: {e}")


# ==============================================================================
# 11. PROGRAMA PRINCIPAL
# ==============================================================================
def main():
    st.set_page_config(page_title="Solucionador 3x3 · Métodos Numéricos", page_icon="🧮", layout="wide")
    inyectar_css()
    inicializar_estado()
    tab_inicio, tab_resolver, tab_comparar = st.tabs(["🏠 Inicio", "✏️ Resolver", "⚖️ Comparar métodos"])
    with tab_inicio:
        pantalla_inicio()
    with tab_resolver:
        pantalla_resolver()
    with tab_comparar:
        pantalla_comparar()


if __name__ == "__main__":
    main()
