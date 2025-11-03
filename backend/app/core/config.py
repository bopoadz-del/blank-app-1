"""
Core configuration for The Reasoner AI Platform.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, PostgresDsn


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    APP_NAME: str = "Reasoner AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database
    DATABASE_URL: PostgresDsn = Field(
        default="postgresql+asyncpg://reasoner:reasoner123@localhost:5432/reasoner_db"
    )
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False
    
    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "reasoner-formulas"
    
    # Edge Computing
    EDGE_NODES: List[str] = Field(default_factory=list)
    EDGE_SYNC_INTERVAL: int = 300  # seconds
    
    # Formula Execution
    MAX_FORMULA_EXECUTION_TIME: int = 30  # seconds
    FORMULA_CACHE_SIZE: int = 1000
    ENABLE_PARALLEL_EXECUTION: bool = True
    MAX_PARALLEL_WORKERS: int = 8
    
    # Learning & Confidence
    AUTO_DEPLOY_CONFIDENCE_THRESHOLD: float = 0.95
    HUMAN_REVIEW_CONFIDENCE_THRESHOLD: float = 0.70
    MIN_SAMPLES_FOR_CONFIDENCE: int = 10
    CONFIDENCE_DECAY_RATE: float = 0.01  # per failed execution
    CONFIDENCE_GROWTH_RATE: float = 0.02  # per successful execution
    
    # Validation Stages
    ENABLE_SYNTACTIC_VALIDATION: bool = True
    ENABLE_DIMENSIONAL_VALIDATION: bool = True
    ENABLE_PHYSICAL_VALIDATION: bool = True
    ENABLE_EMPIRICAL_VALIDATION: bool = True
    ENABLE_OPERATIONAL_VALIDATION: bool = True
    
    PHYSICAL_VALIDATION_TOLERANCE: float = 0.10  # 10%
    EMPIRICAL_VALIDATION_TOLERANCE: float = 0.05  # 5%
    
    # Data Ingestion
    GOOGLE_DRIVE_CREDENTIALS_PATH: Optional[str] = None
    GOOGLE_DRIVE_FOLDER_ID: Optional[str] = None
    DATA_INGESTION_INTERVAL: int = 3600  # seconds
    SUPPORTED_FILE_TYPES: List[str] = Field(
        default_factory=lambda: [".csv", ".xlsx", ".json", ".pdf", ".docx"]
    )
    
    # Security
    SECRET_KEY: str = "CHANGE-THIS-IN-PRODUCTION-USE-STRONG-SECRET"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Monitoring
    ENABLE_PROMETHEUS_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Context Detection
    ENABLE_AUTO_CONTEXT_DETECTION: bool = True
    CONTEXT_CONFIDENCE_THRESHOLD: float = 0.75
    
    # Formula Library
    INITIAL_FORMULAS_PATH: str = "data/formulas/initial_library.json"
    FORMULA_DOMAINS: List[str] = Field(
        default_factory=lambda: [
            "structural_engineering",
            "concrete_technology", 
            "thermal_analysis",
            "financial_metrics",
            "energy_systems",
            "manufacturing",
            "fluid_dynamics"
        ]
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
