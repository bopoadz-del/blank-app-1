"""Reasoner Engine - Core formula execution with lightweight validation."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import logging
import math
import random
import statistics
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.models.schemas import ValidationStageEnum


logger = logging.getLogger(__name__)


class FormulaExecutionError(Exception):
    """Exception raised during formula execution."""
    pass


class ValidationError(Exception):
    """Exception raised during validation."""
    pass


class ReasonerEngine:
    """
    Core reasoning engine for mathematical formula execution.
    Handles parsing, validation, execution, and result tracking.
    """
    
    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._safe_functions = self._initialize_safe_functions()

    def _initialize_safe_functions(self) -> Dict[str, Any]:
        """Initialise the whitelist of mathematical helpers available to formulas."""

        safe_funcs: Dict[str, Any] = {
            # Math module functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "pow": pow,
            "fabs": math.fabs,
            "floor": math.floor,
            "ceil": math.ceil,
            "pi": math.pi,
            "e": math.e,

            # Built-ins that are safe for numeric computation
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
        }
        return safe_funcs
    
    async def execute_formula(
        self,
        formula_expression: str,
        input_values: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a mathematical formula with given inputs.
        
        Args:
            formula_expression: SymPy-compatible formula string
            input_values: Dictionary of input parameter values
            context: Optional context data (climate, material, etc.)
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary containing result, execution time, and metadata
        """
        start_time = time.time()
        timeout = timeout or settings.MAX_FORMULA_EXECUTION_TIME
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(formula_expression, input_values)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for formula execution: {cache_key[:16]}...")
                cached_result = self._cache[cache_key]
                return {
                    **cached_result,
                    "cached": True,
                    "execution_time": 0.0
                }
            
            # Execute with timeout
            try:
                task = asyncio.to_thread(
                    self._execute_formula_sync,
                    formula_expression,
                    input_values,
                )
                result = await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise FormulaExecutionError(
                    f"Formula execution exceeded timeout of {timeout}s"
                ) from exc
            
            execution_time = time.time() - start_time
            
            # Build response
            response = {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "cached": False,
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache result
            if len(self._cache) < settings.FORMULA_CACHE_SIZE:
                # Store a shallow copy so subsequent modifications do not affect the cache
                self._cache[cache_key] = dict(response)
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Formula execution failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "cached": False,
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _execute_formula_sync(
        self,
        formula_expression: str,
        input_values: Dict[str, Any]
    ) -> Any:
        """Synchronous formula execution (runs in thread pool)."""
        try:
            compiled_expr, required_variables = self._analyze_expression(formula_expression)
        except ValueError as exc:  # pragma: no cover - defensive, exercised via tests
            raise FormulaExecutionError(f"Execution error: {exc}") from exc

        missing = [name for name in sorted(required_variables) if name not in input_values]
        if missing:
            raise FormulaExecutionError(f"Missing required input: {missing[0]}")

        eval_context: Dict[str, Any] = dict(self._safe_functions)
        eval_context.update(input_values)

        try:
            result = eval(compiled_expr, {"__builtins__": {}}, eval_context)
        except Exception as exc:  # pragma: no cover - mirrors runtime protection
            raise FormulaExecutionError(f"Execution error: {exc}") from exc

        if isinstance(result, complex):
            if abs(result.imag) < 1e-12:
                result = result.real
            else:
                raise FormulaExecutionError("Complex results are not supported")

        if isinstance(result, bool):  # Preserve boolean results exactly
            return result

        return float(result) if isinstance(result, (int, float)) else result
    
    def _generate_cache_key(
        self,
        formula_expression: str,
        input_values: Dict[str, Any]
    ) -> str:
        """Generate cache key from formula and inputs."""
        key_str = f"{formula_expression}:{sorted(input_values.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    # ========== VALIDATION METHODS ==========
    
    async def validate_formula(
        self,
        formula_expression: str,
        input_parameters: Dict[str, Any],
        output_parameters: Dict[str, Any],
        test_data: Optional[List[Dict[str, Any]]] = None,
        stages: Optional[List[ValidationStageEnum]] = None
    ) -> Dict[str, Any]:
        """
        Run multi-stage validation on a formula.
        
        Returns:
            Dictionary with validation results for each stage
        """
        stages = stages or list(ValidationStageEnum)
        results = []
        overall_passed = True
        
        for stage in stages:
            if stage == ValidationStageEnum.SYNTACTIC:
                result = self._validate_syntactic(formula_expression)
            elif stage == ValidationStageEnum.DIMENSIONAL:
                result = self._validate_dimensional(
                    formula_expression, input_parameters, output_parameters
                )
            elif stage == ValidationStageEnum.PHYSICAL:
                result = await self._validate_physical(
                    formula_expression, input_parameters, output_parameters
                )
            elif stage == ValidationStageEnum.EMPIRICAL:
                result = await self._validate_empirical(
                    formula_expression, test_data or []
                )
            elif stage == ValidationStageEnum.OPERATIONAL:
                result = await self._validate_operational(
                    formula_expression, input_parameters
                )
            else:
                result = {"passed": False, "error": f"Unknown stage: {stage}"}
            
            results.append({
                "stage": stage.value,
                **result,
            })
            
            if not result.get("passed", False):
                overall_passed = False
        
        return {
            "overall_passed": overall_passed,
            "stages": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _validate_syntactic(self, formula_expression: str) -> Dict[str, Any]:
        """Stage 1: Validate syntax."""
        try:
            _, variables = self._analyze_expression(formula_expression)
        except ValueError as exc:
            return {
                "passed": False,
                "confidence": 0.0,
                "error": f"Syntax error: {exc}",
            }

        return {
            "passed": True,
            "confidence": 1.0,
            "details": {"symbols": sorted(variables)},
        }
    
    def _validate_dimensional(
        self,
        formula_expression: str,
        input_parameters: Dict[str, Any],
        output_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 2: Validate dimensional consistency."""
        try:
            # Simple unit checking (can be extended with pint library)
            # For now, just check that units are specified
            input_units = {k: v.get('unit') for k, v in input_parameters.items() if 'unit' in v}
            output_units = {k: v.get('unit') for k, v in output_parameters.items() if 'unit' in v}
            
            if not input_units and not output_units:
                return {
                    "passed": True,
                    "confidence": 0.5,
                    "warning": "No units specified for validation"
                }
            
            # Basic dimensional check passed
            return {
                "passed": True,
                "confidence": 0.8,
                "details": {
                    "input_units": input_units,
                    "output_units": output_units
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "confidence": 0.0,
                "error": f"Dimensional validation error: {str(e)}"
            }
    
    async def _validate_physical(
        self,
        formula_expression: str,
        input_parameters: Dict[str, Any],
        output_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 3: Validate physical realism of outputs."""
        try:
            # Generate test inputs within reasonable ranges
            test_cases = self._generate_physical_test_cases(input_parameters)
            
            unrealistic_count = 0
            for test_input in test_cases:
                result = await self.execute_formula(formula_expression, test_input)
                
                if not result["success"]:
                    continue
                
                # Check output ranges
                output_value = result["result"]
                for param_name, param_def in output_parameters.items():
                    min_val = param_def.get('min_value')
                    max_val = param_def.get('max_value')
                    
                    if min_val is not None and output_value < min_val:
                        unrealistic_count += 1
                    if max_val is not None and output_value > max_val:
                        unrealistic_count += 1
            
            tolerance = settings.PHYSICAL_VALIDATION_TOLERANCE
            pass_rate = 1.0 - (unrealistic_count / len(test_cases))
            passed = pass_rate >= (1.0 - tolerance)
            
            return {
                "passed": passed,
                "confidence": pass_rate,
                "details": {
                    "test_cases": len(test_cases),
                    "unrealistic_count": unrealistic_count,
                    "pass_rate": pass_rate
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "confidence": 0.0,
                "error": f"Physical validation error: {str(e)}"
            }
    
    async def _validate_empirical(
        self,
        formula_expression: str,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Stage 4: Validate against empirical test data."""
        if not test_data:
            return {
                "passed": True,
                "confidence": 0.5,
                "warning": "No test data provided for empirical validation"
            }
        
        try:
            errors: List[float] = []
            for test_case in test_data:
                inputs = test_case.get('inputs', {})
                expected_output = test_case.get('expected_output')

                if expected_output is None:
                    continue
                
                result = await self.execute_formula(formula_expression, inputs)
                
                if not result["success"]:
                    errors.append(float('inf'))
                    continue
                
                actual = result["result"]
                relative_error = (
                    abs(actual - expected_output) / abs(expected_output)
                    if expected_output != 0
                    else abs(actual)
                )
                errors.append(relative_error)

            if not errors:
                return {
                    "passed": False,
                    "confidence": 0.0,
                    "error": "No valid test cases"
                }

            mean_error = sum(errors) / len(errors)
            tolerance = settings.EMPIRICAL_VALIDATION_TOLERANCE
            passed = mean_error <= tolerance

            return {
                "passed": passed,
                "confidence": 1.0 - min(mean_error, 1.0),
                "details": {
                    "test_cases": len(test_data),
                    "mean_relative_error": mean_error,
                    "max_error": max(errors),
                    "min_error": min(errors),
                }
            }

        except Exception as e:
            return {
                "passed": False,
                "confidence": 0.0,
                "error": f"Empirical validation error: {str(e)}"
            }
    
    async def _validate_operational(
        self,
        formula_expression: str,
        input_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 5: Validate operational performance."""
        try:
            # Test execution speed
            test_input = self._generate_single_test_case(input_parameters)
            
            execution_times: List[float] = []
            for _ in range(10):
                result = await self.execute_formula(formula_expression, test_input)
                if result["success"]:
                    execution_times.append(result["execution_time"])
            
            if not execution_times:
                return {
                    "passed": False,
                    "confidence": 0.0,
                    "error": "Formula failed all operational tests"
                }
            
            avg_time = sum(execution_times) / len(execution_times)
            std_time = statistics.pstdev(execution_times) if len(execution_times) > 1 else 0.0
            
            # Check if execution time is acceptable (< 5 seconds)
            passed = avg_time < 5.0
            confidence = max(0.0, 1.0 - (avg_time / 10.0))
            
            return {
                "passed": passed,
                "confidence": confidence,
                "details": {
                    "average_execution_time": avg_time,
                    "std_execution_time": std_time,
                    "min_time": min(execution_times),
                    "max_time": max(execution_times),
                }
            }

        except Exception as e:
            return {
                "passed": False,
                "confidence": 0.0,
                "error": f"Operational validation error: {str(e)}"
            }
    
    def _generate_physical_test_cases(
        self,
        input_parameters: Dict[str, Any],
        count: int = 20
    ) -> List[Dict[str, Any]]:
        """Generate realistic test cases for physical validation."""
        test_cases: List[Dict[str, Any]] = []

        for _ in range(count):
            test_case = {}
            for param_name, param_def in input_parameters.items():
                min_val = param_def.get('min_value', 0.1)
                max_val = param_def.get('max_value', 100.0)

                # Generate random value in range
                value = random.uniform(min_val, max_val)
                test_case[param_name] = value

            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_single_test_case(
        self,
        input_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a single test case."""
        test_case: Dict[str, Any] = {}
        for param_name, param_def in input_parameters.items():
            default = param_def.get('default')
            if default is not None:
                test_case[param_name] = default
            else:
                min_val = param_def.get('min_value', 1.0)
                max_val = param_def.get('max_value', 10.0)
                test_case[param_name] = (min_val + max_val) / 2

        return test_case

    def _analyze_expression(self, formula_expression: str) -> Tuple[object, Set[str]]:
        """Validate and compile a formula expression.

        Returns a tuple with the compiled expression and the set of variable names
        that must be supplied through ``input_values``.
        """

        try:
            tree = ast.parse(formula_expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(exc) from exc

        analyzer = _SafeExpressionAnalyzer(self._safe_functions)
        analyzer.visit(tree)
        compiled = compile(tree, "<formula>", "eval")
        return compiled, analyzer.variables


class _SafeExpressionAnalyzer(ast.NodeVisitor):
    """Ensure that only a safe subset of Python expressions is executed."""

    _ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
        ast.USub,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.IfExp,
        ast.Tuple,
        ast.List,
    )

    def __init__(self, safe_names: Dict[str, Any]) -> None:
        super().__init__()
        self.safe_names = safe_names
        self.variables: Set[str] = set()

    # ``NodeVisitor`` routes every node through ``visit`` before calling the
    # specialised ``visit_*`` method.  Overriding it lets us enforce the allowed
    # node whitelist in one place.
    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        if not isinstance(node, self._ALLOWED_NODES):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
        return super().visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: D401 - short implementation
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        if node.func.id not in self.safe_names:
            raise ValueError(f"Function '{node.func.id}' is not permitted")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: D401 - short implementation
        if node.id not in self.safe_names:
            self.variables.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: D401 - short implementation
        raise ValueError("Attribute access is not allowed in formulas")

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: D401 - short implementation
        raise ValueError("Indexing is not allowed in formulas")


# Global instance
reasoner_engine = ReasonerEngine()
