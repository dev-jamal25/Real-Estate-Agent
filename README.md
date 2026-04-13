# AI Real Estate Agent

A two-stage AI system that extracts property features from natural language, predicts house prices using ML, and explains the valuation through an LLM interpretation.

**Architecture**: LLM extraction → validation → ML prediction → LLM interpretation → FastAPI + Streamlit

---

## Quick Start

### Prerequisites
- Python 3.12
- Virtual environment (`.venv`)

### Setup

1. **Create and activate virtual environment**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install runtime dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install development dependencies** (optional, for testing/linting)
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your LLM API key and model paths
   ```

5. **Start API server**
   ```bash
   python -m app.main
   ```
   Server runs at `http://localhost:8000`

6. **Start Streamlit UI** (in another terminal, activate .venv first)
   ```bash
   streamlit run ui/streamlit_app.py
   ```
   UI opens at `http://localhost:8501`

---

## Project Structure

```
real-estate-agent/
├── app/                       # FastAPI backend
│   ├── main.py               # App entry point
│   ├── config.py             # Configuration from .env
│   ├── api/
│   │   ├── routes.py         # POST /predict, GET /health
│   │   └── dependencies.py   # Dependency injection
│   ├── llm/
│   │   ├── client.py         # LLM API wrapper
│   │   ├── extractor.py      # Stage 1: feature extraction
│   │   ├── interpreter.py    # Stage 2: interpretation
│   │   └── prompt_loader.py  # Prompt management
│   ├── ml/
│   │   ├── loader.py         # Model/preprocessor loading
│   │   ├── predictor.py      # ML prediction
│   │   └── preprocessing.py  # Feature preprocessing
│   ├── models/
│   │   ├── request_models.py   # Request schemas
│   │   ├── response_models.py  # Response schemas
│   │   └── feature_models.py   # Feature schemas
│   └── core/
│       ├── exceptions.py      # Custom exceptions
│       ├── logging_config.py  # Logging setup
│       ├── validators.py      # Feature validation
│       └── constants.py       # Constants
├── ui/
│   └── streamlit_app.py      # Streamlit UI
├── scripts/
│   ├── train_model.py        # ML training
│   ├── evaluate_models.py    # Model comparison
│   ├── run_prompt_eval.py    # Prompt testing
│   └── export_artifacts.py   # Artifact serialization
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   ├── test_api.py           # API tests
│   ├── test_extractor.py     # Extraction tests
│   ├── test_predictor.py     # Prediction tests
│   └── test_validation.py    # Validation tests
├── notebooks/
│   └── real_estate_agent.ipynb  # EDA & training experiments
├── artifacts/                # Serialized models and stats
├── prompts/                  # LLM prompts
│   ├── extract_v1.txt
│   ├── extract_v2.txt
│   └── interpret_v1.txt
├── logs/                     # Application logs
├── docs/
│   ├── architecture_notes.md
│   └── decisions.md
├── .env.example             # Environment template
├── Dockerfile               # Container image
├── .dockerignore
├── .gitignore
├── pyproject.toml           # Tool configuration
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Dev dependencies
└── README.md
```

---

## API Endpoints

### POST /predict
Endpoint for extraction, validation, and prediction.

**Request:**
```json
{
  "query": "3-bedroom ranch with garage in a good neighborhood",
  "feature_overrides": {"bedrooms": 3}
}
```

**Response (incomplete):**
```json
{
  "query": "...",
  "extraction": {
    "features": {"bedrooms": 3},
    "missing_fields": ["bathrooms", "square_feet"],
    "confidence": 0.65
  },
  "status": "incomplete",
  "prediction": null,
  "interpretation": null
}
```

**Response (complete):**
```json
{
  "query": "...",
  "extraction": {...},
  "status": "complete",
  "prediction": {
    "predicted_price": 285000.0,
    "model_name": "RandomForestRegressor",
    "confidence_interval": {"lower": 245000, "upper": 325000}
  },
  "interpretation": {
    "narrative": "...",
    "key_drivers": [...],
    "market_context": "..."
  }
}
```

### GET /health
Health check endpoint. Returns `{"status": "ok"}`.

---

## Development

### Run Tests
```bash
pytest tests/ -v --cov=app
```

### Format and Lint
```bash
black app/ tests/
isort app/ tests/
flake8 app/ tests/
mypy app/
```

---

## Next Steps

Implementation is scaffolded but not yet complete. Features to implement:
- [ ] LLM Stage 1 feature extraction
- [ ] Feature validation and completeness checking
- [ ] ML model training and serialization
- [ ] LLM Stage 2 prediction interpretation
- [ ] Streamlit UI integration with API
- [ ] Full test suite
- [ ] Docker containerization
- [ ] Deployment

---

## License

MIT
