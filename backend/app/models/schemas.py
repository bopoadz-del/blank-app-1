"""
Pydantic schemas for API validation and serialization.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum


# Enums
class FormulaStatusEnum(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    AUTO_DEPLOYED = "auto_deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ExecutionStatusEnum(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ValidationStageEnum(str, Enum):
    SYNTACTIC = "syntactic"
    DIMENSIONAL = "dimensional"
    PHYSICAL = "physical"
    EMPIRICAL = "empirical"
    OPERATIONAL = "operational"


# Formula Schemas
class ParameterDefinition(BaseModel):
    """Parameter definition schema."""
    type: str  # "float", "int", "string", "array"
    unit: Optional[str] = None
    description: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default: Optional[Any] = None
    required: bool = True


class FormulaCreate(BaseModel):
    """Schema for creating a new formula."""
    formula_id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=200)
    description: Optional[str] = None
    domain: str = Field(..., max_length=100)
    formula_expression: str
    input_parameters: Dict[str, ParameterDefinition]
    output_parameters: Dict[str, ParameterDefinition]
    required_context: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    optional_context: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    source: Optional[str] = None
    source_reference: Optional[str] = None
    version: str = "1.0.0"


class FormulaUpdate(BaseModel):
    """Schema for updating a formula."""
    name: Optional[str] = None
    description: Optional[str] = None
    formula_expression: Optional[str] = None
    status: Optional[FormulaStatusEnum] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class FormulaResponse(BaseModel):
    """Schema for formula response."""
    id: int
    formula_id: str
    name: str
    description: Optional[str]
    domain: str
    formula_expression: str
    input_parameters: Dict[str, Any]
    output_parameters: Dict[str, Any]
    required_context: Dict[str, Any]
    optional_context: Dict[str, Any]
    source: Optional[str]
    source_reference: Optional[str]
    version: str
    status: FormulaStatusEnum
    confidence_score: float
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: Optional[float]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Execution Schemas
class FormulaExecutionRequest(BaseModel):
    """Schema for formula execution request."""
    formula_id: str
    input_values: Dict[str, Any]
    context_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None
    edge_node_id: Optional[str] = None
    
    @validator('input_values')
    def validate_inputs_not_empty(cls, v):
        if not v:
            raise ValueError("input_values cannot be empty")
        return v


class FormulaExecutionResponse(BaseModel):
    """Schema for formula execution response."""
    execution_id: str
    formula_id: int
    status: ExecutionStatusEnum
    input_values: Dict[str, Any]
    output_values: Optional[Dict[str, Any]]
    context_data: Dict[str, Any]
    execution_time: Optional[float]
    error_message: Optional[str]
    validation_passed: Optional[bool]
    actual_vs_expected_error: Optional[float]
    execution_timestamp: datetime
    mlflow_run_id: Optional[str]
    
    class Config:
        from_attributes = True


# Recommendation Schemas
class FormulaRecommendationRequest(BaseModel):
    """Schema for getting formula recommendations."""
    domain: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    limit: int = Field(10, ge=1, le=100)


class FormulaRecommendation(BaseModel):
    """Schema for a single formula recommendation."""
    formula_id: str
    name: str
    description: Optional[str]
    domain: str
    confidence_score: float
    match_score: float  # How well context matches
    total_executions: int
    success_rate: float


class FormulaRecommendationResponse(BaseModel):
    """Schema for formula recommendations response."""
    recommendations: List[FormulaRecommendation]
    total_count: int
    context_used: Dict[str, Any]


# Learning Schemas
class ConfidenceUpdate(BaseModel):
    """Schema for confidence score update."""
    formula_id: str
    execution_result: bool  # success or failure
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    evidence: Optional[Dict[str, Any]] = Field(default_factory=dict)


class LearningEventResponse(BaseModel):
    """Schema for learning event response."""
    event_id: str
    event_type: str
    formula_id: Optional[int]
    old_confidence: Optional[float]
    new_confidence: Optional[float]
    reason: Optional[str]
    timestamp: datetime
    triggered_by: Optional[str]
    
    class Config:
        from_attributes = True


# Validation Schemas
class ValidationRequest(BaseModel):
    """Schema for validation request."""
    formula_id: str
    validation_stages: List[ValidationStageEnum] = Field(
        default_factory=lambda: list(ValidationStageEnum)
    )
    test_data: Optional[List[Dict[str, Any]]] = None


class ValidationStageResult(BaseModel):
    """Schema for a single validation stage result."""
    stage: ValidationStageEnum
    passed: bool
    confidence: Optional[float]
    error_message: Optional[str]
    details: Optional[Dict[str, Any]]


class ValidationResponse(BaseModel):
    """Schema for validation response."""
    formula_id: str
    overall_passed: bool
    stages: List[ValidationStageResult]
    timestamp: datetime


# Analytics Schemas
class FormulaAnalytics(BaseModel):
    """Schema for formula analytics."""
    formula_id: str
    name: str
    domain: str
    total_executions: int
    success_rate: float
    average_execution_time: float
    confidence_trend: List[Dict[str, Any]]  # [{timestamp, confidence}]
    top_contexts: List[Dict[str, Any]]  # [{context, success_rate}]
    performance_by_context: Dict[str, Any]


class SystemMetrics(BaseModel):
    """Schema for system-wide metrics."""
    total_formulas: int
    total_executions: int
    average_confidence: float
    formulas_by_status: Dict[FormulaStatusEnum, int]
    formulas_by_domain: Dict[str, int]
    recent_learning_events: int
    edge_nodes_active: int
    uptime_seconds: float


# Data Ingestion Schemas
class DataSourceCreate(BaseModel):
    """Schema for creating a data source."""
    source_id: str
    source_type: str  # "google_drive", "api", "upload"
    source_config: Dict[str, Any]
    sync_interval: int = 3600
    is_active: bool = True


class DataSourceResponse(BaseModel):
    """Schema for data source response."""
    id: int
    source_id: str
    source_type: str
    last_sync_timestamp: Optional[datetime]
    sync_interval: int
    is_active: bool
    total_files_processed: int
    total_data_points_extracted: int
    last_error: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Health Check
class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    version: str
    database_connected: bool
    mlflow_connected: bool
    edge_nodes_connected: int
    timestamp: datetime
