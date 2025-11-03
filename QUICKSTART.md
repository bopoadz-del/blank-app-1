# Quick Start Guide - Reasoner AI Platform

## Prerequisites

- Docker & Docker Compose installed
- 8GB RAM minimum
- Python 3.11+ (for local development)

## Installation & Setup (5 minutes)

### 1. Start the System

```bash
# Clone/extract the package
cd reasoner-platform

# Start all services
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
docker-compose logs -f
```

### 2. Initialize Database with Formulas

```bash
# Initialize database and load 10+ built-in formulas
docker-compose exec backend python -m app.core.init_db
```

### 3. Access the Platform

- **Dashboard:** http://localhost:3000
- **API Documentation:** http://localhost:8000/docs
- **MLflow Tracking:** http://localhost:5000

## First Steps

### Try a Formula Execution

1. Open Dashboard: http://localhost:3000
2. Click "Execute" tab
3. Select "Concrete Compressive Strength" formula
4. Enter inputs:
   - S_ultimate: 50
   - k: 0.005
   - maturity: 2000
5. Click "Execute Formula"
6. See result and execution time

### View Learning Insights

1. Click "Learning" tab
2. See confidence updates and learning events
3. Monitor system improvement over time

## API Usage Examples

### Execute a Formula

```bash
curl -X POST http://localhost:8000/api/v1/formulas/execute \
  -H "Content-Type: application/json" \
  -d '{
    "formula_id": "beam_deflection_simply_supported",
    "input_values": {
      "w": 10.0,
      "L": 5.0,
      "E": 200.0,
      "I": 0.0001
    },
    "context_data": {
      "climate": "temperate",
      "material": "steel"
    }
  }'
```

### Get Formula Recommendations

```bash
curl "http://localhost:8000/api/v1/formulas/recommend" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "structural_engineering",
    "context": {"climate": "hot_arid"},
    "min_confidence": 0.7,
    "limit": 5
  }'
```

### List All Formulas

```bash
curl "http://localhost:8000/api/v1/formulas?limit=100"
```

### Get System Metrics

```bash
curl "http://localhost:8000/api/v1/analytics/system"
```

## System Components

### Backend (Port 8000)
- FastAPI REST API
- Reasoner Engine (SymPy-based formula execution)
- Tinker ML (confidence learning)
- PostgreSQL + TimescaleDB

### Frontend (Port 3000)
- React Dashboard
- Formula library browser
- Real-time execution interface
- Learning insights visualization

### MLflow (Port 5000)
- Experiment tracking
- Model versioning
- Performance metrics

## Next Steps

1. **Add Your Own Formulas:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/formulas \
     -H "Content-Type: application/json" \
     -d @your_formula.json
   ```

2. **Run Validations:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/formulas/YOUR_FORMULA_ID/validate \
     -H "Content-Type: application/json" \
     -d '{"validation_stages": ["syntactic", "dimensional", "physical"]}'
   ```

3. **Monitor Learning:**
   ```bash
   curl "http://localhost:8000/api/v1/learning/insights?days=7"
   ```

## Stopping the System

```bash
# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs backend
docker-compose logs postgres

# Restart services
docker-compose restart
```

### Database connection errors
```bash
# Wait for postgres to be ready
docker-compose exec postgres pg_isready -U reasoner

# Reinitialize database
docker-compose exec backend python -m app.core.init_db
```

### Port conflicts
Edit `docker-compose.yml` to change port mappings:
```yaml
ports:
  - "8001:8000"  # Change 8000 to 8001
```

## Support

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- System Metrics: http://localhost:8000/api/v1/analytics/system
