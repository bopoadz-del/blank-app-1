# Data Ingestion & Context Detection - Complete Guide

## Overview

The Reasoner AI Platform now includes **complete implementations** of:

1. **Google Drive Connector** - Sync files from Google Drive
2. **File Parsers** - Extract data from PDF, DOCX, Excel, CSV, JSON
3. **Context Detection** - Automatically detect context from multiple sources
4. **Context Classification** - Classify climate, materials, site conditions, project types
5. **Context Enrichment** - Add standards and constraints

---

## 1. Google Drive Integration

### Setup

```bash
# 1. Get Google Drive API credentials
# Go to https://console.cloud.google.com/
# Create service account and download JSON credentials

# 2. Configure in .env
GOOGLE_DRIVE_CREDENTIALS_PATH=/path/to/credentials.json
GOOGLE_DRIVE_FOLDER_ID=your-folder-id
```

### Usage

```python
from app.services.data_ingestion import GoogleDriveConnector

# Initialize connector
connector = GoogleDriveConnector(
    credentials_path="/path/to/credentials.json",
    folder_id="your-folder-id"
)

# List files
files = connector.list_files(
    file_types=['application/pdf'],
    modified_after=datetime(2024, 1, 1),
    limit=100
)

# Download specific file
content = connector.download_file(file_id="abc123")

# Sync entire folder
synced = connector.sync_folder(
    local_cache_dir="./drive_cache",
    file_types=['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
)
```

### API Endpoints

```bash
# Sync Google Drive
curl -X POST http://localhost:8000/api/v1/data-sources/google-drive/sync

# List cached files
curl http://localhost:8000/api/v1/data-sources/drive-cache
```

---

## 2. File Parsing

### Supported Formats

- **PDF** - Text and table extraction
- **DOCX** - Paragraphs and tables
- **Excel** (XLSX, XLS) - All sheets with data
- **CSV** - Tabular data
- **JSON** - Structured data

### Usage

```python
from app.services.data_ingestion import FileParser, DataExtractor

# Parse file (auto-detects type)
parsed = FileParser.parse_file("/path/to/document.pdf")

# Extract numerical data
numerical = DataExtractor.extract_numerical_data(parsed)
# Returns: {"length": [{"value": 5.0, "unit": "m"}], ...}

# Extract context hints
context_hints = DataExtractor.extract_context_hints(parsed)
# Returns: {"climate": "hot_arid", "material": "concrete", ...}
```

### API Example

```bash
# Upload and parse file
curl -X POST http://localhost:8000/api/v1/data-sources/files/parse \
  -F "file=@document.pdf"

# Returns:
# {
#   "filename": "document.pdf",
#   "file_type": "pdf",
#   "numerical_data": {...},
#   "context_hints": {...}
# }
```

---

## 3. Context Detection

### Detection Sources

The system can detect context from:

1. **Text Content** - Documents, descriptions, specifications
2. **Sensor Data** - Temperature, humidity, pressure, wind
3. **Location Data** - GPS coordinates, elevation, distance to coast
4. **Input Values** - Infer from formula parameters

### Text-Based Detection

```python
from app.services.context_detection import ContextDetector

detector = ContextDetector()

text = """
This concrete building in coastal Dubai will be exposed
to severe marine environment with hot and humid climate.
Using Type V cement for sulfate resistance.
"""

context = detector.detect_from_text(text)
# Returns:
# {
#   "climate": "hot_humid",
#   "material": "concrete",
#   "site_condition": "coastal",
#   "cement_type": "Type_V",
#   "exposure_class": "severe",
#   "confidence": 1.0
# }
```

### Sensor-Based Detection

```python
sensor_data = {
    "temperature": 38,  # Celsius
    "humidity": 75,     # %
    "pressure": 101.2,  # kPa
    "wind_speed": 25    # m/s
}

context = detector.detect_from_sensor_data(sensor_data)
# Returns:
# {
#   "climate": "hot_humid",
#   "site_condition": "coastal"
# }
```

### Location-Based Detection

```python
location = {
    "latitude": 25.2048,
    "longitude": 55.2708,
    "elevation": 5,
    "distance_to_coast": 2,
    "country": "AE"
}

context = detector.detect_from_location(location)
# Returns:
# {
#   "climate_zone": "tropical",
#   "site_condition": "coastal",
#   "elevation_category": "low_altitude",
#   "building_code": "UAE"
# }
```

### Comprehensive Detection

```python
# Combine all sources for best accuracy
context = detector.detect_comprehensive(
    text="Coastal concrete building project in Dubai",
    sensor_data={"temperature": 38, "humidity": 75},
    location={"latitude": 25.2, "country": "AE"},
    input_values={"f_c": 50, "E": 30}
)

# Returns merged context from all sources with confidence score
```

### API Examples

```bash
# Detect from text
curl -X POST http://localhost:8000/api/v1/context/detect-from-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Coastal concrete structure in hot humid climate"}'

# Detect from sensors
curl -X POST http://localhost:8000/api/v1/context/detect-from-sensors \
  -H "Content-Type: application/json" \
  -d '{"temperature": 38, "humidity": 75, "pressure": 101.2}'

# Detect from location
curl -X POST http://localhost:8000/api/v1/context/detect-from-location \
  -H "Content-Type: application/json" \
  -d '{"latitude": 25.2, "longitude": 55.3, "country": "AE"}'

# Comprehensive detection
curl -X POST http://localhost:8000/api/v1/context/detect-comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Dubai coastal project",
    "sensor_data": {"temperature": 38},
    "location": {"latitude": 25.2},
    "input_values": {"f_c": 50}
  }'
```

---

## 4. Context Classification

### Built-in Classifiers

#### Climate Classifier
Detects: `hot_arid`, `hot_humid`, `temperate`, `cold`

**Patterns:**
- hot_arid: "desert", "arid", "low humidity", "Saudi", "Dubai"
- hot_humid: "tropical", "humid", "monsoon", "Singapore", "Mumbai"
- temperate: "moderate", "mild", "London", "Paris"
- cold: "arctic", "freezing", "sub-zero", "Moscow", "Alaska"

#### Material Classifier
Detects: `concrete`, `steel`, `aluminum`, `wood`, `masonry`, `composite`

#### Site Condition Classifier
Detects: `coastal`, `mountain`, `urban`, `industrial`, `marine`

#### Project Type Classifier
Detects: `building`, `bridge`, `road`, `tunnel`, `dam`, `pipeline`, `foundation`

### Custom Classifiers

```python
class CustomClassifier:
    def __init__(self):
        self.patterns = {
            'your_category': r'\b(pattern1|pattern2)\b'
        }
    
    def classify(self, text: str) -> Optional[str]:
        for category, pattern in self.patterns.items():
            if re.search(pattern, text):
                return category
        return None
```

---

## 5. Context Enrichment

### Standards Enrichment

```python
from app.services.context_detection import ContextEnricher

enricher = ContextEnricher()

context = {"material": "concrete", "climate": "hot_arid"}

enriched = enricher.enrich_with_standards(context)
# Adds: {"applicable_standards": ["ACI_318", "ASTM_C39"]}

enriched = enricher.enrich_with_constraints(context)
# Adds: {"constraints": ["thermal_expansion", "curing_requirements"]}
```

### Standard Mappings

**Materials:**
- concrete → ACI_318, ASTM_C39
- steel → AISC, ASTM_A36
- aluminum → AA, ASTM_B209

**Locations:**
- US → ACI, AISC
- UK → BS (British Standards)
- EU → EN (Eurocode)
- Saudi Arabia → SASO
- UAE → UAE Building Code

---

## 6. Formula Execution with Auto-Context

### Smart Execution

The system can automatically detect context and execute formulas:

```python
# Traditional execution (manual context)
result = await execute_formula(
    formula_id="concrete_strength",
    input_values={"S_ultimate": 50, "maturity": 2000},
    context={"climate": "hot_arid", "material": "concrete"}
)

# Smart execution (auto-detect context)
result = await execute_formula_with_auto_context(
    formula_id="concrete_strength",
    input_values={"S_ultimate": 50, "maturity": 2000},
    text_hint="Project in Dubai coastal area",
    sensor_data={"temperature": 38},
    location={"latitude": 25.2}
)
# System automatically detects: climate, material, site conditions
```

### API Example

```bash
curl -X POST http://localhost:8000/api/v1/formulas/execute-with-auto-context \
  -H "Content-Type: application/json" \
  -d '{
    "formula_id": "concrete_compressive_strength_maturity",
    "input_values": {
      "S_ultimate": 50,
      "k": 0.005,
      "maturity": 2000
    },
    "text_hint": "Coastal project in Dubai with severe marine exposure",
    "sensor_data": {
      "temperature": 38,
      "humidity": 75
    },
    "location": {
      "latitude": 25.2048,
      "country": "AE"
    }
  }'

# Returns:
# {
#   "execution_result": {...},
#   "auto_detected_context": {
#     "climate": "hot_humid",
#     "site_condition": "coastal",
#     "material": "concrete",
#     "building_code": "UAE",
#     "confidence": 0.95
#   }
# }
```

---

## 7. Integration Examples

### Example 1: Google Drive → Parse → Execute

```python
# 1. Sync from Google Drive
connector = GoogleDriveConnector(credentials_path="creds.json")
files = connector.sync_folder("./cache")

# 2. Parse files
for file_info in files:
    parsed = FileParser.parse_file(file_info['local_path'])
    
    # 3. Extract data
    numerical = DataExtractor.extract_numerical_data(parsed)
    context = DataExtractor.extract_context_hints(parsed)
    
    # 4. Execute formulas with extracted data
    for formula_id in relevant_formulas:
        result = await execute_formula(
            formula_id=formula_id,
            input_values=numerical,
            context=context
        )
```

### Example 2: Sensor Stream → Context → Formula

```python
# Continuous sensor monitoring
async def process_sensor_stream():
    while True:
        sensor_data = await read_sensors()
        
        # Detect context
        context = detector.detect_from_sensor_data(sensor_data)
        
        # Execute relevant formulas
        if context['climate'] == 'hot_arid':
            # Use heat-adjusted formulas
            await execute_formula(
                formula_id="thermal_expansion_adjusted",
                input_values=sensor_data,
                context=context
            )
```

### Example 3: Document Analysis → Formula Selection

```python
# Analyze construction specification document
spec_doc = FileParser.parse_file("specifications.pdf")

# Extract context
context = DataExtractor.extract_context_hints(spec_doc)
numerical = DataExtractor.extract_numerical_data(spec_doc)

# Get recommended formulas for this context
recommendations = tinker_ml.recommend_formulas(
    db=db,
    domain="structural_engineering",
    context=context,
    min_confidence=0.8
)

# Execute recommended formulas
for rec in recommendations:
    result = await execute_formula(
        formula_id=rec['formula_id'],
        input_values=numerical,
        context=context
    )
```

---

## 8. Configuration

### Environment Variables

```bash
# Google Drive
GOOGLE_DRIVE_CREDENTIALS_PATH=/path/to/credentials.json
GOOGLE_DRIVE_FOLDER_ID=your-folder-id
DATA_INGESTION_INTERVAL=3600  # seconds

# File types to sync
SUPPORTED_FILE_TYPES=.csv,.xlsx,.json,.pdf,.docx

# Context detection
ENABLE_AUTO_CONTEXT_DETECTION=true
CONTEXT_CONFIDENCE_THRESHOLD=0.75
```

### Customization

```python
# Custom file parser
class CustomParser:
    @staticmethod
    def parse_custom_format(file_path: str) -> Dict:
        # Your parsing logic
        return {"data": [...]}

# Register custom parser
FileParser.parsers['.custom'] = CustomParser.parse_custom_format

# Custom context detector
class CustomContextDetector:
    def detect_special_conditions(self, data):
        # Your detection logic
        return {"special_context": "value"}
```

---

## 9. Testing

### Unit Tests

```python
# Test context detection
def test_climate_detection():
    detector = ContextDetector()
    context = detector.detect_from_text("hot desert climate")
    assert context['climate'] == 'hot_arid'

# Test file parsing
def test_pdf_parsing():
    parsed = FileParser.parse_pdf("test.pdf")
    assert parsed['type'] == 'pdf'
    assert len(parsed['text']) > 0
```

### Integration Tests

```bash
# Test Google Drive sync
python -m pytest tests/integration/test_google_drive.py

# Test context detection pipeline
python -m pytest tests/integration/test_context_detection.py
```

---

## 10. Production Deployment

### Setup Checklist

- [ ] Configure Google Drive API credentials
- [ ] Set up file cache directory (with sufficient space)
- [ ] Configure sync intervals
- [ ] Test context detection accuracy
- [ ] Set up monitoring for sync failures
- [ ] Configure backup for cached files

### Monitoring

```python
# Monitor sync health
@app.get("/health/data-ingestion")
async def check_ingestion_health():
    return {
        "google_drive_connected": check_drive_connection(),
        "last_sync": get_last_sync_time(),
        "cached_files": count_cached_files(),
        "context_detection_enabled": settings.ENABLE_AUTO_CONTEXT_DETECTION
    }
```

---

## Conclusion

The Reasoner AI Platform now has **complete, production-ready implementations** for:

✅ Google Drive integration  
✅ Multi-format file parsing (PDF, DOCX, Excel, CSV, JSON)  
✅ Comprehensive context detection (text, sensors, location, inputs)  
✅ Context classification (climate, materials, site, project)  
✅ Context enrichment (standards, constraints)  
✅ Smart formula execution with auto-context

**No stubs. No placeholders. Fully functional.**
