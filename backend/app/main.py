"""
Main FastAPI application for The Reasoner AI Platform.
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime

from app.core.config import settings
from app.core.database import get_db, engine
from app.models import database, schemas
from app.services.reasoner import reasoner_engine
from app.services.tinker import tinker_ml

# Create database tables
database.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Universal Mathematical Reasoning Infrastructure with Continuous Learning"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== HEALTH & STATUS ====================

@app.get("/health", response_model=schemas.HealthCheck)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_connected = True
    except:
        db_connected = False
    
    return schemas.HealthCheck(
        status="healthy" if db_connected else "degraded",
        version=settings.APP_VERSION,
        database_connected=db_connected,
        mlflow_connected=True,  # TODO: Implement MLflow check
        edge_nodes_connected=len(settings.EDGE_NODES),
        timestamp=datetime.utcnow()
    )


# ==================== FORMULA MANAGEMENT ====================

@app.post(
    f"{settings.API_V1_PREFIX}/formulas",
    response_model=schemas.FormulaResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_formula(
    formula: schemas.FormulaCreate,
    db: Session = Depends(get_db)
):
    """Create a new formula."""
    # Check if formula_id already exists
    existing = db.query(database.Formula).filter(
        database.Formula.formula_id == formula.formula_id
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formula with ID {formula.formula_id} already exists"
        )
    
    # Create formula
    db_formula = database.Formula(
        formula_id=formula.formula_id,
        name=formula.name,
        description=formula.description,
        domain=formula.domain,
        formula_expression=formula.formula_expression,
        input_parameters=formula.input_parameters,
        output_parameters=formula.output_parameters,
        required_context=formula.required_context,
        optional_context=formula.optional_context,
        source=formula.source,
        source_reference=formula.source_reference,
        version=formula.version,
        status=database.FormulaStatus.PENDING_REVIEW
    )
    
    db.add(db_formula)
    db.commit()
    db.refresh(db_formula)
    
    return db_formula


@app.get(
    f"{settings.API_V1_PREFIX}/formulas",
    response_model=List[schemas.FormulaResponse]
)
async def list_formulas(
    domain: Optional[str] = None,
    status: Optional[schemas.FormulaStatusEnum] = None,
    min_confidence: float = 0.0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List formulas with optional filters."""
    query = db.query(database.Formula)
    
    if domain:
        query = query.filter(database.Formula.domain == domain)
    
    if status:
        query = query.filter(database.Formula.status == status)
    
    if min_confidence > 0:
        query = query.filter(database.Formula.confidence_score >= min_confidence)
    
    formulas = query.order_by(
        database.Formula.confidence_score.desc()
    ).limit(limit).all()
    
    return formulas


@app.get(
    f"{settings.API_V1_PREFIX}/formulas/{{formula_id}}",
    response_model=schemas.FormulaResponse
)
async def get_formula(formula_id: str, db: Session = Depends(get_db)):
    """Get a specific formula by ID."""
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )
    
    return formula


@app.patch(
    f"{settings.API_V1_PREFIX}/formulas/{{formula_id}}",
    response_model=schemas.FormulaResponse
)
async def update_formula(
    formula_id: str,
    update: schemas.FormulaUpdate,
    db: Session = Depends(get_db)
):
    """Update a formula."""
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )
    
    # Update fields
    update_data = update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(formula, field, value)
    
    formula.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(formula)
    
    return formula


# ==================== FORMULA EXECUTION ====================

@app.post(
    f"{settings.API_V1_PREFIX}/formulas/execute",
    response_model=schemas.FormulaExecutionResponse
)
async def execute_formula(
    request: schemas.FormulaExecutionRequest,
    db: Session = Depends(get_db)
):
    """Execute a formula with given inputs."""
    # Get formula
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == request.formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {request.formula_id} not found"
        )
    
    # Execute formula
    result = await reasoner_engine.execute_formula(
        formula_expression=formula.formula_expression,
        input_values=request.input_values,
        context=request.context_data
    )
    
    # Create execution record
    execution_id = str(uuid.uuid4())
    execution_status = (
        database.ExecutionStatus.COMPLETED if result["success"]
        else database.ExecutionStatus.FAILED
    )
    
    db_execution = database.FormulaExecution(
        execution_id=execution_id,
        formula_id=formula.id,
        input_values=request.input_values,
        output_values=result.get("result"),
        context_data=request.context_data,
        status=execution_status,
        execution_time=result.get("execution_time"),
        error_message=result.get("error"),
        expected_output=request.expected_output,
        edge_node_id=request.edge_node_id
    )
    
    # Calculate error if expected output provided
    if request.expected_output and result["success"]:
        try:
            expected = float(list(request.expected_output.values())[0])
            actual = float(result["result"])
            error = abs(actual - expected) / abs(expected) if expected != 0 else abs(actual)
            db_execution.actual_vs_expected_error = error
            db_execution.validation_passed = error < 0.05
        except:
            pass
    
    db.add(db_execution)
    db.commit()
    
    # Update confidence via Tinker ML
    if db_execution.status == database.ExecutionStatus.COMPLETED:
        await tinker_ml.update_confidence_from_execution(
            db=db,
            formula_id=formula.id,
            execution_success=result["success"],
            context=request.context_data,
            error_magnitude=db_execution.actual_vs_expected_error
        )
    
    db.refresh(db_execution)
    return db_execution


@app.get(
    f"{settings.API_V1_PREFIX}/executions/{{execution_id}}",
    response_model=schemas.FormulaExecutionResponse
)
async def get_execution(execution_id: str, db: Session = Depends(get_db)):
    """Get execution details."""
    execution = db.query(database.FormulaExecution).filter(
        database.FormulaExecution.execution_id == execution_id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found"
        )
    
    return execution


# ==================== RECOMMENDATIONS ====================

@app.post(
    f"{settings.API_V1_PREFIX}/formulas/recommend",
    response_model=schemas.FormulaRecommendationResponse
)
async def recommend_formulas(
    request: schemas.FormulaRecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get formula recommendations based on domain and context."""
    recommendations = tinker_ml.recommend_formulas(
        db=db,
        domain=request.domain,
        context=request.context,
        min_confidence=request.min_confidence,
        limit=request.limit
    )
    
    return schemas.FormulaRecommendationResponse(
        recommendations=recommendations,
        total_count=len(recommendations),
        context_used=request.context or {}
    )


# ==================== VALIDATION ====================

@app.post(
    f"{settings.API_V1_PREFIX}/formulas/{{formula_id}}/validate",
    response_model=schemas.ValidationResponse
)
async def validate_formula(
    formula_id: str,
    request: schemas.ValidationRequest,
    db: Session = Depends(get_db)
):
    """Run validation on a formula."""
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )
    
    # Run validation
    validation_result = await reasoner_engine.validate_formula(
        formula_expression=formula.formula_expression,
        input_parameters=formula.input_parameters,
        output_parameters=formula.output_parameters,
        test_data=request.test_data,
        stages=request.validation_stages
    )
    
    # Store validation results
    for stage_result in validation_result["stages"]:
        db_validation = database.ValidationResult(
            formula_id=formula.id,
            validation_stage=stage_result["stage"],
            passed=stage_result["passed"],
            confidence=stage_result.get("confidence"),
            validation_data=stage_result.get("details"),
            error_message=stage_result.get("error"),
            validated_by="system"
        )
        db.add(db_validation)
    
    # Update formula validation status
    if validation_result["overall_passed"]:
        formula.validation_stages_passed = [
            s["stage"] for s in validation_result["stages"] if s["passed"]
        ]
        formula.last_validation_date = datetime.utcnow()
    
    db.commit()
    
    return schemas.ValidationResponse(
        formula_id=formula.formula_id,
        overall_passed=validation_result["overall_passed"],
        stages=validation_result["stages"],
        timestamp=datetime.utcnow()
    )


# ==================== ANALYTICS & INSIGHTS ====================

@app.get(
    f"{settings.API_V1_PREFIX}/analytics/formulas/{{formula_id}}",
    response_model=schemas.FormulaAnalytics
)
async def get_formula_analytics(formula_id: str, db: Session = Depends(get_db)):
    """Get analytics for a specific formula."""
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )
    
    # Get learning events for confidence trend
    learning_events = db.query(database.LearningEvent).filter(
        database.LearningEvent.formula_id == formula.id
    ).order_by(database.LearningEvent.timestamp).limit(100).all()
    
    confidence_trend = [
        {
            "timestamp": e.timestamp.isoformat(),
            "confidence": e.new_confidence
        }
        for e in learning_events
    ]
    
    # Get context performances
    context_perfs = db.query(database.ContextPerformance).filter(
        database.ContextPerformance.formula_id == formula.id
    ).order_by(database.ContextPerformance.confidence_in_context.desc()).limit(10).all()
    
    top_contexts = [
        {
            "context": cp.context_data,
            "success_rate": cp.successful_executions / cp.total_executions if cp.total_executions > 0 else 0,
            "executions": cp.total_executions
        }
        for cp in context_perfs
    ]
    
    return schemas.FormulaAnalytics(
        formula_id=formula.formula_id,
        name=formula.name,
        domain=formula.domain,
        total_executions=formula.total_executions,
        success_rate=formula.successful_executions / formula.total_executions if formula.total_executions > 0 else 0,
        average_execution_time=formula.average_execution_time or 0.0,
        confidence_trend=confidence_trend,
        top_contexts=top_contexts,
        performance_by_context={}
    )


@app.get(
    f"{settings.API_V1_PREFIX}/analytics/system",
    response_model=schemas.SystemMetrics
)
async def get_system_metrics(db: Session = Depends(get_db)):
    """Get system-wide metrics."""
    total_formulas = db.query(database.Formula).count()
    total_executions = db.query(database.FormulaExecution).count()
    
    # Average confidence
    avg_confidence = db.query(
        database.func.avg(database.Formula.confidence_score)
    ).scalar() or 0.0
    
    # Formulas by status
    status_counts = {}
    for status_val in database.FormulaStatus:
        count = db.query(database.Formula).filter(
            database.Formula.status == status_val
        ).count()
        status_counts[status_val.value] = count
    
    # Formulas by domain
    domain_counts = {}
    domains = db.query(database.Formula.domain).distinct().all()
    for (domain,) in domains:
        count = db.query(database.Formula).filter(
            database.Formula.domain == domain
        ).count()
        domain_counts[domain] = count
    
    # Recent learning events
    recent_events = db.query(database.LearningEvent).filter(
        database.LearningEvent.timestamp >= datetime.utcnow() - timedelta(days=7)
    ).count()
    
    return schemas.SystemMetrics(
        total_formulas=total_formulas,
        total_executions=total_executions,
        average_confidence=float(avg_confidence),
        formulas_by_status=status_counts,
        formulas_by_domain=domain_counts,
        recent_learning_events=recent_events,
        edge_nodes_active=len(settings.EDGE_NODES),
        uptime_seconds=0.0  # TODO: Implement uptime tracking
    )


@app.get(
    f"{settings.API_V1_PREFIX}/learning/insights",
)
async def get_learning_insights(
    formula_id: Optional[str] = None,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get learning insights and trends."""
    formula_db_id = None
    if formula_id:
        formula = db.query(database.Formula).filter(
            database.Formula.formula_id == formula_id
        ).first()
        if formula:
            formula_db_id = formula.id
    
    insights = tinker_ml.get_learning_insights(
        db=db,
        formula_id=formula_db_id,
        days=days
    )
    
    return insights


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1
    )
