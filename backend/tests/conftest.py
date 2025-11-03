"""Pytest configuration and fixtures."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base

@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def sample_formula():
    """Sample formula for testing."""
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
