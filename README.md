# The Reasoner AI Platform

**Universal Mathematical Reasoning Infrastructure with Continuous Learning**

A production-ready system that combines symbolic reasoning, machine learning, and edge computing to provide context-aware mathematical solutions across multiple domains (construction, energy, finance, manufacturing).

## Core Innovation

Instead of fixed formulas in traditional engineering software, The Reasoner:
- **Learns which formulas work best** in specific contexts
- **Tracks success/failure rates** and adjusts confidence scores
- **Auto-deploys trusted formulas** vs. requiring human approval for new ones
- **Processes at the edge** for offline-first operation

## Architecture (4 Layers)

```
┌─────────────────────────────────────────┐
│  PLANNER AI (Strategic Orchestration)   │
│  - Task routing, resource allocation    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  TINKER ML (Continuous Learning)        │
│  - Success tracking, confidence updates │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  REASONER ENGINE (Formula Execution)    │
│  - SymPy/SciPy integration              │
│  - 5-stage validation pipeline          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  EDGE NODES (Data Processing)           │
│  - Local computation, offline-first     │
└─────────────────────────────────────────┘
```

## Tech Stack

**Backend:** FastAPI, Python 3.11+, SQLAlchemy, Pydantic
**Database:** PostgreSQL + TimescaleDB
**ML/Math:** SymPy, NumPy, SciPy, PyTorch, scikit-learn
**Tracking:** MLflow
**Edge:** Jetson Orin Nano (optional), TensorRT
**Frontend:** React + Tailwind CSS
**DevOps:** Docker, docker-compose, nginx

## Quick Start

```bash
# 1. Clone and setup
git clone <repo>
cd reasoner-platform

# 2. Start services
docker-compose up -d

# 3. Initialize database
docker-compose exec backend python -m app.core.init_db

# 4. Access dashboard
open http://localhost:3000

# 5. API documentation
open http://localhost:8000/docs
```

## 4-Week Implementation Timeline

**Weeks 1-2: Core System**
- ✅ Formula registry and execution engine
- ✅ Data ingestion (Google Drive, APIs)
- ✅ Context detection and formula selection
- ✅ Database schema and API endpoints

**Weeks 3-4: Intelligence Layer**
- ✅ Tinker ML (success tracking, confidence scoring)
- ✅ 5-stage validation pipeline
- ✅ Credibility-based autonomy framework
- ✅ Dashboard with formula library and results

## Key Features

### 1. Formula Registry (20+ Built-in Formulas)
- Structural engineering (beam deflection, column capacity)
- Concrete properties (strength, curing, thermal)
- Financial metrics (NPV, IRR, Sharpe ratio)
- Energy systems (heat transfer, efficiency)

### 2. Context-Aware Selection
```python
context = {
    "climate": "hot_humid",
    "material": "concrete_grade_50",
    "contractor_history": 0.85,
    "site_conditions": "coastal"
}
# System automatically selects best formula based on historical success
```

### 3. Continuous Learning
- Tracks every formula execution result
- Updates confidence scores based on accuracy
- Learns which formulas work in which contexts
- Auto-deploys trusted formulas (>95% confidence)

### 4. 5-Stage Validation
1. **Syntactic:** Valid Python/SymPy syntax
2. **Dimensional:** Unit consistency check
3. **Physical:** Realistic output ranges
4. **Empirical:** Matches test data within tolerance
5. **Operational:** Performs in production

## Project Structure

```
reasoner-platform/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes
│   │   ├── core/         # Config, database, security
│   │   ├── models/       # SQLAlchemy models
│   │   └── services/     # Business logic
│   ├── tests/            # Unit and integration tests
│   └── requirements.txt
├── frontend/
│   ├── src/              # React components
│   └── public/
├── edge/
│   ├── edge_processor.py # Jetson edge computing
│   └── sensors/          # Data collection scripts
├── devops/
│   ├── docker-compose.yml
│   └── nginx/
├── data/
│   ├── formulas/         # Initial formula library (JSON)
│   └── datasets/         # Test datasets
└── docs/
    ├── API.md            # API documentation
    ├── ARCHITECTURE.md   # System design
    └── DEPLOYMENT.md     # Production deployment guide
```

## API Examples

### Execute Formula
```bash
curl -X POST http://localhost:8000/api/v1/formulas/execute \
  -H "Content-Type: application/json" \
  -d '{
    "formula_id": "concrete_strength_maturity",
    "inputs": {
      "time_hours": 168,
      "temperature_celsius": 35,
      "cement_type": "Type_I"
    },
    "context": {
      "climate": "hot_arid",
      "site_id": "project_alpha"
    }
  }'
```

### Get Formula Recommendations
```bash
curl http://localhost:8000/api/v1/formulas/recommend \
  -G \
  -d "domain=structural" \
  -d "context_climate=temperate" \
  -d "min_confidence=0.8"
```

## Configuration

Edit `backend/app/core/config.py` or use environment variables:

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/reasoner
MLFLOW_TRACKING_URI=http://localhost:5000
EDGE_NODES=node1:192.168.1.10,node2:192.168.1.11
AUTO_DEPLOY_THRESHOLD=0.95
HUMAN_REVIEW_THRESHOLD=0.70
```

## Testing

```bash
# Run all tests
docker-compose exec backend pytest

# With coverage
docker-compose exec backend pytest --cov=app --cov-report=html

# Integration tests only
docker-compose exec backend pytest tests/integration/
```

## Deployment

See `docs/DEPLOYMENT.md` for:
- Production configuration
- Scaling guidelines
- Monitoring setup
- Backup strategies
- Security hardening

## License

MIT License

## Contact

For technical questions or collaboration opportunities, contact the development team.
