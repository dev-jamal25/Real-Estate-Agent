# AI Real Estate Agent - Stage 1 Extraction

**This slice**: LLM-powered property feature extraction with prompt versioning and evaluation.

A conversational chatbot that extracts structured property features from natural language input. This is the first stage of a two-stage AI real estate valuation system.

**Architecture**: User input → Extraction LLM → Structured features (+ validation) → Service response

---

## Quick Start

### Prerequisites
- Python 3.12
- Virtual environment (`.venv`)
- OpenRouter API key (for real LLM calls; mock mode available for testing)

### Setup

1. **Create and activate virtual environment**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install dev dependencies** (for testing/linting)
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenRouter API key
   ```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_extraction_service.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Run Prompt Evaluation

```bash
python -c "
from app.extraction.evaluator import ExtractionEvaluator
evaluator = ExtractionEvaluator('evals/extraction_eval_cases.yaml')
results = evaluator.evaluate(versions=['v1', 'v2'], mock_responses={
    'v1': {'extracted_features': {}, 'missing_features': [], 'reply': 'test'},
    'v2': {'extracted_features': {}, 'missing_features': [], 'reply': 'test'}
})
evaluator.save_results()
print('Evaluation complete. Results saved to logs/extraction_prompt_eval.jsonl')
"
```

---

## Stage 1 Architecture

### Contract-First Design
- Single source of truth: `app/contracts/property_features.py`
  - 12 required features
  - Feature groups (numeric, ordinal, nominal)
  - Ordinal mappings (user-facing → model tokens)

### Request-Response Schema
```python
# Input: latest_message + accumulated_state
# Output: Stage1Response
{
  "extracted_features": {...},    # Partially or fully normalized
  "missing_features": [...],       # What's still needed
  "is_complete": False,            # All 12 features present?
  "reply": "..."                   # LLM-generated follow-up
}
```

### Prompt Versioning
- **extract_v1.txt**: Concise, direct extraction strategy
- **extract_v2.txt**: Conservative, constrained approach
- Stored in `prompts/`
- Loaded and cached by `app/llm/prompt_loader.py`
- Evaluated on test cases in `evals/extraction_eval_cases.yaml`

### Modules

| Module | Purpose |
|--------|---------|
| `app/contracts/property_features.py` | Feature contract (12 features, mappings, groups) |
| `app/core/config.py` | Environment settings (OpenRouter API) |
| `app/core/exceptions.py` | Custom exceptions |
| `app/core/logging_config.py` | Logging setup with file rotation |
| `app/llm/openrouter_client.py` | HTTP client to OpenRouter API |
| `app/llm/prompt_loader.py` | Prompt loading and caching |
| `app/extraction/schemas.py` | Pydantic models (ExtractedFeatures, Stage1Response) |
| `app/extraction/normalizer.py` | Ordinal/numeric value normalization |
| `app/extraction/service.py` | Main orchestration pipeline |
| `app/extraction/evaluator.py` | Prompt variant evaluation on test cases |

---

## Configuration

### Environment Variables
- `OPENROUTER_API_KEY` - Your OpenRouter API key (required for real calls)
- `OPENROUTER_BASE_URL` - API endpoint (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_MODEL` - Model (default: `openrouter/elephant-alpha`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `LOG_FILE` - Log file path (default: `logs/app.log`)

### Feature Names (12 Required)
```
OverallQual, GrLivArea, TotalBsmtSF, GarageCars, Neighborhood,
ExterQual, YearBuilt, FullBath, KitchenQual, BsmtQual,
TotRmsAbvGrd, LotArea
```

### Ordinal Mappings
User input → Model token:
- `None` → `None`
- `Poor` → `Po`
- `Fair` → `Fa`
- `Typical/Average` → `TA`
- `Good` → `Gd`
- `Excellent` → `Ex`

---

## Testing Strategy

All tests mock OpenRouter; no real API key required for tests.

- `tests/test_extraction_service.py`: Service orchestration, merging state, validation
- `tests/test_extraction_normalizer.py`: Value normalization, ordinal mapping
- `tests/test_extraction_evaluator.py`: Prompt evaluation, JSONL logging

---

## Logging

- **Console**: INFO level output to terminal
- **File**: `logs/app.log` with 10MB rotation (5 backups)
- **Evaluation**: JSONL format at `logs/extraction_prompt_eval.jsonl`

---

## Key Design Decisions

See `docs/decisions.md` for rationale on:
- OpenRouter-only (no Gemini, no fallback in this slice)
- JSON-structured LLM output
- Stateless service architecture
- Mock-first testing
- Prompt versioning with evaluation

---

## What This Slice Does NOT Include

- FastAPI routes / REST endpoints
- ML prediction / artifact loading
- Interpretation stage (Stage 2)
- Streamlit UI
- Fallback providers
- Database / session storage


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
