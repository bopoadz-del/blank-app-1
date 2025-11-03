"""
Test suite for Reasoner Engine.
"""
import pytest
import asyncio
from app.services.reasoner import reasoner_engine
from app.models.schemas import ValidationStageEnum


class TestReasonerEngine:
    """Test the Reasoner Engine formula execution."""
    
    @pytest.mark.asyncio
    async def test_simple_formula_execution(self):
        """Test basic formula execution."""
        result = await reasoner_engine.execute_formula(
            formula_expression="2 * x + 3",
            input_values={"x": 5}
        )
        
        assert result["success"] is True
        assert result["result"] == 13
        assert "execution_time" in result
    
    @pytest.mark.asyncio
    async def test_formula_with_sqrt(self):
        """Test formula with square root."""
        result = await reasoner_engine.execute_formula(
            formula_expression="sqrt(x)",
            input_values={"x": 16}
        )
        
        assert result["success"] is True
        assert result["result"] == 4.0
    
    @pytest.mark.asyncio
    async def test_missing_input(self):
        """Test formula execution with missing input."""
        result = await reasoner_engine.execute_formula(
            formula_expression="x + y",
            input_values={"x": 5}
        )
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_complex_formula(self):
        """Test complex engineering formula."""
        # Beam deflection formula
        result = await reasoner_engine.execute_formula(
            formula_expression="(5 * w * L**4) / (384 * E * I)",
            input_values={
                "w": 10.0,
                "L": 5.0,
                "E": 200.0,
                "I": 0.0001
            }
        )
        
        assert result["success"] is True
        assert result["result"] > 0
    
    @pytest.mark.asyncio
    async def test_formula_caching(self):
        """Test that formulas are cached."""
        # First execution
        result1 = await reasoner_engine.execute_formula(
            formula_expression="x * 2",
            input_values={"x": 10}
        )
        
        # Second execution (should be cached)
        result2 = await reasoner_engine.execute_formula(
            formula_expression="x * 2",
            input_values={"x": 10}
        )
        
        assert result1["success"] is True
        assert result2["success"] is True
        assert result2["cached"] is True
        assert result2["execution_time"] == 0.0


class TestValidation:
    """Test the 5-stage validation pipeline."""
    
    @pytest.mark.asyncio
    async def test_syntactic_validation_pass(self):
        """Test syntactic validation with valid formula."""
        result = await reasoner_engine.validate_formula(
            formula_expression="x + y * 2",
            input_parameters={
                "x": {"type": "float", "unit": "m"},
                "y": {"type": "float", "unit": "m"}
            },
            output_parameters={
                "result": {"type": "float", "unit": "m"}
            },
            stages=[ValidationStageEnum.SYNTACTIC]
        )
        
        assert result["overall_passed"] is True
        assert result["stages"][0]["passed"] is True
    
    @pytest.mark.asyncio
    async def test_syntactic_validation_fail(self):
        """Test syntactic validation with invalid formula."""
        result = await reasoner_engine.validate_formula(
            formula_expression="x + + y",  # Invalid syntax
            input_parameters={
                "x": {"type": "float"},
                "y": {"type": "float"}
            },
            output_parameters={
                "result": {"type": "float"}
            },
            stages=[ValidationStageEnum.SYNTACTIC]
        )
        
        assert result["overall_passed"] is False
    
    @pytest.mark.asyncio
    async def test_dimensional_validation(self):
        """Test dimensional validation."""
        result = await reasoner_engine.validate_formula(
            formula_expression="x + y",
            input_parameters={
                "x": {"type": "float", "unit": "m"},
                "y": {"type": "float", "unit": "m"}
            },
            output_parameters={
                "result": {"type": "float", "unit": "m"}
            },
            stages=[ValidationStageEnum.DIMENSIONAL]
        )
        
        assert result["stages"][0]["passed"] is True
    
    @pytest.mark.asyncio
    async def test_physical_validation(self):
        """Test physical validation with realistic outputs."""
        result = await reasoner_engine.validate_formula(
            formula_expression="sqrt(x)",
            input_parameters={
                "x": {"type": "float", "min_value": 0, "max_value": 100}
            },
            output_parameters={
                "result": {"type": "float", "min_value": 0, "max_value": 10}
            },
            stages=[ValidationStageEnum.PHYSICAL]
        )
        
        assert result["stages"][0]["passed"] is True
    
    @pytest.mark.asyncio
    async def test_empirical_validation_with_test_data(self):
        """Test empirical validation with test data."""
        test_data = [
            {"inputs": {"x": 4}, "expected_output": 2},
            {"inputs": {"x": 9}, "expected_output": 3},
            {"inputs": {"x": 16}, "expected_output": 4}
        ]
        
        result = await reasoner_engine.validate_formula(
            formula_expression="sqrt(x)",
            input_parameters={"x": {"type": "float"}},
            output_parameters={"result": {"type": "float"}},
            test_data=test_data,
            stages=[ValidationStageEnum.EMPIRICAL]
        )
        
        assert result["stages"][0]["passed"] is True
        assert result["stages"][0]["confidence"] > 0.9
    
    @pytest.mark.asyncio
    async def test_operational_validation(self):
        """Test operational validation (performance)."""
        result = await reasoner_engine.validate_formula(
            formula_expression="x * 2 + y * 3",
            input_parameters={
                "x": {"type": "float", "default": 1.0},
                "y": {"type": "float", "default": 2.0}
            },
            output_parameters={"result": {"type": "float"}},
            stages=[ValidationStageEnum.OPERATIONAL]
        )
        
        assert result["stages"][0]["passed"] is True
        details = result["stages"][0]["details"]
        assert details["average_execution_time"] < 5.0  # Should be fast
    
    @pytest.mark.asyncio
    async def test_full_validation_pipeline(self):
        """Test all 5 validation stages together."""
        result = await reasoner_engine.validate_formula(
            formula_expression="(5 * w * L**4) / (384 * E * I)",
            input_parameters={
                "w": {"type": "float", "unit": "kN/m", "min_value": 0.1, "max_value": 100, "default": 10},
                "L": {"type": "float", "unit": "m", "min_value": 1, "max_value": 50, "default": 5},
                "E": {"type": "float", "unit": "GPa", "min_value": 10, "max_value": 400, "default": 200},
                "I": {"type": "float", "unit": "m^4", "min_value": 1e-6, "max_value": 1, "default": 0.0001}
            },
            output_parameters={
                "deflection": {"type": "float", "unit": "m", "min_value": 0, "max_value": 1}
            }
        )
        
        # At least syntactic and operational should pass
        assert len(result["stages"]) == 5
        syntactic_passed = any(s["stage"] == "syntactic" and s["passed"] for s in result["stages"])
        assert syntactic_passed is True


class TestFormulaExecution:
    """Test real-world formula executions."""
    
    @pytest.mark.asyncio
    async def test_concrete_strength_formula(self):
        """Test concrete compressive strength maturity formula."""
        result = await reasoner_engine.execute_formula(
            formula_expression="S_ultimate * (1 - exp(-k * maturity))",
            input_values={
                "S_ultimate": 50.0,
                "k": 0.005,
                "maturity": 2000.0
            }
        )
        
        assert result["success"] is True
        assert 0 < result["result"] < 50  # Should be less than ultimate
    
    @pytest.mark.asyncio
    async def test_npv_formula(self):
        """Test NPV calculation."""
        # Simplified NPV for single period
        result = await reasoner_engine.execute_formula(
            formula_expression="cf / (1 + r)",
            input_values={
                "cf": 1000.0,
                "r": 0.1
            }
        )
        
        assert result["success"] is True
        assert abs(result["result"] - 909.09) < 1  # NPV of 1000 at 10%
    
    @pytest.mark.asyncio
    async def test_heat_transfer_formula(self):
        """Test heat transfer by conduction."""
        result = await reasoner_engine.execute_formula(
            formula_expression="(k * A * (T_hot - T_cold)) / thickness",
            input_values={
                "k": 0.5,
                "A": 1.0,
                "T_hot": 100.0,
                "T_cold": 20.0,
                "thickness": 0.1
            }
        )
        
        assert result["success"] is True
        assert result["result"] > 0  # Heat transfer should be positive


@pytest.fixture
def sample_formula():
    """Fixture providing a sample formula definition."""
    return {
        "formula_id": "test_formula",
        "name": "Test Formula",
        "domain": "test",
        "formula_expression": "x * 2 + y",
        "input_parameters": {
            "x": {"type": "float", "unit": "m"},
            "y": {"type": "float", "unit": "m"}
        },
        "output_parameters": {
            "result": {"type": "float", "unit": "m"}
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
