# src/equation_simplifier.py

import sympy as sp
from sympy.parsing.latex import parse_latex

def simplify_equation(latex_str: str) -> str:
    """
    Convert LaTeX to SymPy, simplify symbolic form, output simplified LaTeX.
    Only works for algebraic expressions.
    """

    try:
        expr = parse_latex(latex_str)
        simplified = sp.simplify(expr)
        simplified_latex = sp.latex(simplified)
        return simplified_latex
    except Exception:
        # If SymPy cannot understand the equation, return original
        return latex_str
