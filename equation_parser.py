import sympy as sp
import numpy as np
from sympy.parsing.latex import parse_latex
import re

def preprocess_equation(equation_str):
    # Convert user-friendly sum/product notation to sympy format
    # Example: Sum_{k=1}^{5} x^k / k!  -->  Sum(x**k / factorial(k), (k, 1, 5))
    sum_pattern = r"Sum_\{([a-zA-Z])=(\d+)\}\^\{(\d+)\} (.+)"
    prod_pattern = r"Prod_\{([a-zA-Z])=(\d+)\}\^\{(\d+)\} (.+)"
    def sum_repl(match):
        var, start, end, expr = match.groups()
        # Replace ^ with ** for powers, and ! with factorial()
        expr = expr.replace('^', '**')
        expr = re.sub(r'(\w)!', r'factorial(\1)', expr)
        return f"Sum({expr}, ({var}, {start}, {end}))"
    def prod_repl(match):
        var, start, end, expr = match.groups()
        expr = expr.replace('^', '**')
        expr = re.sub(r'(\w)!', r'factorial(\1)', expr)
        return f"Product({expr}, ({var}, {start}, {end}))"
    equation_str = re.sub(sum_pattern, sum_repl, equation_str)
    equation_str = re.sub(prod_pattern, prod_repl, equation_str)
    return equation_str

def parse_equation(equation_str, variables=('x', 'y')):
    equation_str = preprocess_equation(equation_str)
    dummy_vars = set()
    for match in re.finditer(r'(Sum|Product)\((.*?),\s*\((\w+),', equation_str):
        dummy_vars.add(match.group(3))
    try:
        if '\\' in equation_str or '{' in equation_str or '}' in equation_str or '\\frac' in equation_str:
            expr = parse_latex(equation_str)
        else:
            raise ValueError
    except Exception:
        sigmoid = lambda x: 1 / (1 + sp.exp(-x))
        floor = sp.functions.floor
        ceil = sp.functions.ceiling
        heaviside = sp.Heaviside
        DiracDelta = sp.DiracDelta
        KroneckerDelta = sp.KroneckerDelta
        all_vars = list(variables) + list(dummy_vars)
        symbols = sp.symbols(all_vars)
        local_dict = {
            'sigmoid': sigmoid,
            'floor': floor,
            'ceil': ceil,
            'Heaviside': heaviside,
            'DiracDelta': DiracDelta,
            'KroneckerDelta': KroneckerDelta,
            'pi': sp.pi,
            'e': sp.E,
            'factorial': sp.factorial,
            'Sum': sp.Sum,
            'Product': sp.Product,
        }
        for v in dummy_vars:
            local_dict[v] = sp.symbols(v)
        expr = sp.sympify(equation_str, locals=local_dict)
    mesh_vars = tuple(sp.symbols(list(variables)))
    # If the result is still a Sum or Product, try to evaluate it numerically for each input
    if expr.has(sp.Sum, sp.Product):
        def func(*args):
            subs = dict(zip(mesh_vars, args))
            return np.vectorize(lambda *vals: expr.subs(dict(zip(mesh_vars, vals))).doit().evalf())(*args)
        return func
    expr = expr.doit()
    func = sp.lambdify(mesh_vars, expr, modules=['numpy'])
    return func
