"""
API endpoints for data ingestion and context detection.

Add these to main.py after the existing endpoints.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List

from app.core.database import get_db
from app.core.config import settings
from app.services.data_ingestion import GoogleDriveConnector, FileParser, DataExtractor
from app.services.context_detection import ContextDetector, ContextEnricher
from app.models import schemas

router = APIRouter()


# ==================== DATA INGESTION ====================

@router.post("/data-sources/google-drive/sync")
async def sync_google_drive(db: Session = Depends(get_db)):
    """Sync files from Google Drive."""
    if not settings.GOOGLE_DRIVE_CREDENTIALS_PATH:
        raise HTTPException(
            status_code=400,
            detail="Google Drive not configured. Set GOOGLE_DRIVE_CREDENTIALS_PATH in .env"
        )
    
    try:
        connector = GoogleDriveConnector(
            credentials_path=settings.GOOGLE_DRIVE_CREDENTIALS_PATH,
            folder_id=settings.GOOGLE_DRIVE_FOLDER_ID
        )
        
        synced_files = connector.sync_folder(
            local_cache_dir="./data/drive_cache",
            file_types=[
                'application/pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'text/csv'
            ]
        )
        
        return {
            "synced_count": len(synced_files),
            "files": [
                {
                    "name": f['name'],
                    "type": f['mimeType'],
                    "modified": f['modifiedTime'],
                    "size": f.get('size', 0)
                }
                for f in synced_files
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-sources/files/parse")
async def parse_uploaded_file(file: UploadFile = File(...)):
    """Parse uploaded file and extract data."""
    try:
        # Save temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Parse file
        parsed_data = FileParser.parse_file(temp_path)
        
        # Extract numerical data
        numerical_data = DataExtractor.extract_numerical_data(parsed_data)
        
        # Extract context hints
        context_hints = DataExtractor.extract_context_hints(parsed_data)
        
        return {
            "filename": file.filename,
            "file_type": parsed_data['type'],
            "parsed_data": parsed_data,
            "numerical_data": numerical_data,
            "context_hints": context_hints
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-sources/drive-cache")
async def list_drive_cache():
    """List files in Google Drive cache."""
    import os
    from pathlib import Path
    
    cache_dir = Path("./data/drive_cache")
    
    if not cache_dir.exists():
        return {"files": [], "count": 0}
    
    files = []
    for file_path in cache_dir.glob("*"):
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            })
    
    return {
        "files": files,
        "count": len(files),
        "cache_directory": str(cache_dir)
    }


# ==================== CONTEXT DETECTION ====================

@router.post("/context/detect-from-text")
async def detect_context_from_text(request: dict):
    """Detect context from text content."""
    text = request.get('text', '')
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    detector = ContextDetector()
    context = detector.detect_from_text(text)
    
    # Enrich context
    enricher = ContextEnricher()
    enriched = enricher.enrich_with_standards(context)
    enriched = enricher.enrich_with_constraints(enriched)
    
    return {
        "detected_context": context,
        "enriched_context": enriched,
        "confidence": context.get('confidence', 0.0)
    }


@router.post("/context/detect-from-sensors")
async def detect_context_from_sensors(sensor_data: dict):
    """Detect context from sensor readings."""
    detector = ContextDetector()
    context = detector.detect_from_sensor_data(sensor_data)
    
    return {
        "sensor_data": sensor_data,
        "detected_context": context
    }


@router.post("/context/detect-from-location")
async def detect_context_from_location(location: dict):
    """Detect context from geographic location."""
    detector = ContextDetector()
    context = detector.detect_from_location(location)
    
    return {
        "location": location,
        "detected_context": context
    }


@router.post("/context/detect-comprehensive")
async def detect_comprehensive_context(request: dict):
    """Detect context from multiple sources."""
    text = request.get('text')
    sensor_data = request.get('sensor_data')
    location = request.get('location')
    input_values = request.get('input_values')
    
    detector = ContextDetector()
    context = detector.detect_comprehensive(
        text=text,
        sensor_data=sensor_data,
        location=location,
        input_values=input_values
    )
    
    # Enrich
    enricher = ContextEnricher()
    enriched = enricher.enrich_with_standards(context)
    enriched = enricher.enrich_with_constraints(enriched)
    
    return {
        "detected_context": context,
        "enriched_context": enriched,
        "confidence": context.get('confidence', 0.0),
        "sources_used": {
            "text": text is not None,
            "sensors": sensor_data is not None,
            "location": location is not None,
            "inputs": input_values is not None
        }
    }


# ==================== FORMULA EXECUTION WITH AUTO CONTEXT ====================

@router.post("/formulas/execute-with-auto-context")
async def execute_formula_with_auto_context(
    request: dict,
    db: Session = Depends(get_db)
):
    """
    Execute formula with automatic context detection.
    Combines formula execution with intelligent context detection.
    """
    from app.services.reasoner import reasoner_engine
    from app.services.tinker import tinker_ml
    from app.models.database import Formula
    import uuid
    
    formula_id = request.get('formula_id')
    input_values = request.get('input_values', {})
    
    # Optional: provide hints for better detection
    text_hint = request.get('text_hint')
    sensor_data = request.get('sensor_data')
    location = request.get('location')
    
    # Get formula
    formula = db.query(Formula).filter(
        Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(status_code=404, detail="Formula not found")
    
    # Auto-detect context
    detector = ContextDetector()
    detected_context = detector.detect_comprehensive(
        text=text_hint,
        sensor_data=sensor_data,
        location=location,
        input_values=input_values
    )
    
    logger.info(f"Auto-detected context: {detected_context}")
    
    # Execute formula with detected context
    result = await reasoner_engine.execute_formula(
        formula_expression=formula.formula_expression,
        input_values=input_values,
        context=detected_context
    )
    
    # Store execution and update confidence
    if result["success"]:
        await tinker_ml.update_confidence_from_execution(
            db=db,
            formula_id=formula.id,
            execution_success=True,
            context=detected_context
        )
    
    return {
        "execution_result": result,
        "auto_detected_context": detected_context,
        "context_confidence": detected_context.get('confidence', 0.0)
    }


# Add to main.py:
# app.include_router(router, prefix=f"{settings.API_V1_PREFIX}", tags=["data-ingestion-context"])
