"""Initialize database with initial formulas."""
import json
from pathlib import Path
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, engine
from app.models.database import Base, Formula, FormulaStatus

def init_database():
    """Initialize database tables and load initial formulas."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Load initial formulas
    formulas_path = Path(__file__).parent.parent.parent / "data" / "formulas" / "initial_library.json"
    
    if not formulas_path.exists():
        print(f"Initial formulas file not found at {formulas_path}")
        return
    
    with open(formulas_path) as f:
        initial_formulas = json.load(f)
    
    db = SessionLocal()
    
    try:
        for formula_data in initial_formulas:
            # Check if formula already exists
            existing = db.query(Formula).filter(
                Formula.formula_id == formula_data["formula_id"]
            ).first()
            
            if existing:
                print(f"Formula {formula_data['formula_id']} already exists, skipping...")
                continue
            
            # Create formula
            formula = Formula(
                formula_id=formula_data["formula_id"],
                name=formula_data["name"],
                description=formula_data.get("description"),
                domain=formula_data["domain"],
                formula_expression=formula_data["formula_expression"],
                input_parameters=formula_data["input_parameters"],
                output_parameters=formula_data["output_parameters"],
                required_context=formula_data.get("required_context", {}),
                optional_context=formula_data.get("optional_context", {}),
                source=formula_data.get("source"),
                source_reference=formula_data.get("source_reference"),
                status=FormulaStatus.APPROVED,
                confidence_score=0.7  # Initial moderate confidence
            )
            
            db.add(formula)
            print(f"Added formula: {formula_data['name']}")
        
        db.commit()
        print(f"\nSuccessfully initialized database with {len(initial_formulas)} formulas!")
        
    except Exception as e:
        db.rollback()
        print(f"Error initializing database: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
