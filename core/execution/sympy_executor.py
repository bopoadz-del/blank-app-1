from sympy import symbols, Eq, solve, sympify
from core.execution.sandbox import time_limit, TimeoutError
class SafeExecutor:
    def __init__(self, expression:str): self.expression=expression
    def solve_for(self, target_symbol:str, inputs:dict):
        with time_limit(2):
            expr=sympify(self.expression)
            syms={k: (v['value'] if isinstance(v,dict) and 'value' in v else v) for k,v in inputs.items()}
            target=symbols(target_symbol)
            sol=solve(Eq(0, expr), target)
            if not sol: raise ValueError('No solution')
            return float(sol[0].subs({symbols(k): syms[k] for k in syms}))
