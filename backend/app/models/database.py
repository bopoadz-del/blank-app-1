"""
SQLAlchemy models for The Reasoner AI Platform.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, JSON, DateTime, 
    Text, Enum, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import JSONB
import enum


Base = declarative_base()


class FormulaStatus(str, enum.Enum):
    """Formula status in the system."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    AUTO_DEPLOYED = "auto_deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ExecutionStatus(str, enum.Enum):
    """Execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ValidationStage(str, enum.Enum):
    """Validation stages."""
    SYNTACTIC = "syntactic"
    DIMENSIONAL = "dimensional"
    PHYSICAL = "physical"
    EMPIRICAL = "empirical"
    OPERATIONAL = "operational"


class Formula(Base):
    """Mathematical formula with metadata and versioning."""
    __tablename__ = "formulas"
    
    id = Column(Integer, primary_key=True, index=True)
    formula_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    domain = Column(String(100), index=True, nullable=False)
    
    # Formula definition
    formula_expression = Column(Text, nullable=False)  # SymPy expression
    input_parameters = Column(JSONB, nullable=False)  # {"param": {"type": "float", "unit": "m", ...}}
    output_parameters = Column(JSONB, nullable=False)
    
    # Context requirements
    required_context = Column(JSONB, default={})  # {"climate": ["hot_arid"], "material": [...]}
    optional_context = Column(JSONB, default={})
    
    # Metadata
    source = Column(String(100))  # "ISO_standard", "consultant_report", "AI_discovered"
    source_reference = Column(String(500))
    version = Column(String(20), default="1.0.0")
    status = Column(Enum(FormulaStatus), default=FormulaStatus.PENDING_REVIEW)
    
    # Learning metrics
    confidence_score = Column(Float, default=0.5)
    total_executions = Column(Integer, default=0)
    successful_executions = Column(Integer, default=0)
    failed_executions = Column(Integer, default=0)
    average_execution_time = Column(Float)  # seconds
    
    # Validation
    validation_stages_passed = Column(JSONB, default=[])
    last_validation_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    
    # Relationships
    executions = relationship("FormulaExecution", back_populates="formula")
    context_performances = relationship("ContextPerformance", back_populates="formula")
    
    __table_args__ = (
        Index("idx_formula_domain_status", "domain", "status"),
        Index("idx_formula_confidence", "confidence_score"),
    )


class FormulaExecution(Base):
    """Record of formula execution with inputs, outputs, and context."""
    __tablename__ = "formula_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(String(100), unique=True, index=True, nullable=False)
    formula_id = Column(Integer, ForeignKey("formulas.id"), nullable=False)
    
    # Execution data
    input_values = Column(JSONB, nullable=False)
    output_values = Column(JSONB)
    context_data = Column(JSONB, default={})
    
    # Execution metadata
    status = Column(Enum(ExecutionStatus), default=ExecutionStatus.QUEUED)
    execution_time = Column(Float)  # seconds
    error_message = Column(Text)
    
    # Result validation
    expected_output = Column(JSONB)  # if available for validation
    actual_vs_expected_error = Column(Float)  # percentage error
    validation_passed = Column(Boolean)
    
    # Environment
    edge_node_id = Column(String(100))
    executed_by = Column(String(100))
    execution_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # MLflow tracking
    mlflow_run_id = Column(String(100))
    
    # Relationships
    formula = relationship("Formula", back_populates="executions")
    
    __table_args__ = (
        Index("idx_execution_formula_status", "formula_id", "status"),
        Index("idx_execution_timestamp", "execution_timestamp"),
    )


class ContextPerformance(Base):
    """Track formula performance in specific contexts."""
    __tablename__ = "context_performances"
    
    id = Column(Integer, primary_key=True, index=True)
    formula_id = Column(Integer, ForeignKey("formulas.id"), nullable=False)
    
    # Context definition
    context_hash = Column(String(64), index=True, nullable=False)  # MD5 of sorted context
    context_data = Column(JSONB, nullable=False)  # full context dict
    
    # Performance metrics
    total_executions = Column(Integer, default=0)
    successful_executions = Column(Integer, default=0)
    average_error = Column(Float)
    confidence_in_context = Column(Float, default=0.5)
    
    # Statistics
    last_execution_date = Column(DateTime)
    first_execution_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    formula = relationship("Formula", back_populates="context_performances")
    
    __table_args__ = (
        UniqueConstraint("formula_id", "context_hash", name="uq_formula_context"),
        Index("idx_context_confidence", "confidence_in_context"),
    )


class ValidationResult(Base):
    """Store validation results for formulas."""
    __tablename__ = "validation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    formula_id = Column(Integer, ForeignKey("formulas.id"), nullable=False)
    
    # Validation details
    validation_stage = Column(Enum(ValidationStage), nullable=False)
    passed = Column(Boolean, nullable=False)
    confidence = Column(Float)
    
    # Details
    validation_data = Column(JSONB)  # test inputs, expected outputs, etc.
    error_message = Column(Text)
    notes = Column(Text)
    
    # Metadata
    validated_by = Column(String(100))  # "system" or user ID
    validated_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_validation_formula_stage", "formula_id", "validation_stage"),
    )


class DataSource(Base):
    """Track data sources for ingestion."""
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Source details
    source_type = Column(String(50), nullable=False)  # "google_drive", "api", "upload"
    source_config = Column(JSONB, nullable=False)  # credentials, endpoints, etc.
    
    # Processing
    last_sync_timestamp = Column(DateTime)
    sync_interval = Column(Integer, default=3600)  # seconds
    is_active = Column(Boolean, default=True)
    
    # Statistics
    total_files_processed = Column(Integer, default=0)
    total_data_points_extracted = Column(Integer, default=0)
    last_error = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LearningEvent(Base):
    """Track learning events for system improvement."""
    __tablename__ = "learning_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Event details
    event_type = Column(String(50), nullable=False)  # "confidence_update", "formula_discovery", etc.
    formula_id = Column(Integer, ForeignKey("formulas.id"))
    
    # Changes
    old_confidence = Column(Float)
    new_confidence = Column(Float)
    reason = Column(Text)
    evidence_data = Column(JSONB)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    triggered_by = Column(String(100))  # "system", "user", "execution_result"
    
    __table_args__ = (
        Index("idx_learning_event_type", "event_type"),
        Index("idx_learning_timestamp", "timestamp"),
    )
