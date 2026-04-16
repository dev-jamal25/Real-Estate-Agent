# Architecture and Design - Stage 1 Extraction

## Goal

Extract property features from conversational user input using an LLM, validate them, and return what's missing. This is the first stage of a two-stage real estate valuation system.

## High-Level Flow (Stage 1 Only)

```
User message + accumulated state
         ↓
   Prompt loading (v1 or v2)
         ↓
   OpenRouter API call (elephant-alpha)
         ↓
   JSON parsing + validation
         ↓
   Feature normalization
         ↓
   Completeness check
         ↓
   Stage1Response (features + missing list + reply)
```

## Main Components

### 1. Contracts Layer
- **`app/contracts/property_features.py`**
  - Single source of truth: 12 required features
  - Feature groups (numeric: 7, ordinal: 3, nominal: 2)
  - Ordinal user→model token mappings
  - User-friendly descriptions
  - Invariant: Every required feature must be in exactly one group

### 2. LLM Layer
- **`app/llm/openrouter_client.py`**
  - Direct HTTP to OpenRouter API
  - Deterministic JSON extraction (temperature = 0)
  - No SDK; minimal dependencies
  - Mock-friendly for testing
  
- **`app/llm/prompt_loader.py`**
  - Loads extract_v1.txt and extract_v2.txt
  - Caches prompts to avoid repeated disk I/O
  - Raises PromptLoadException if file missing

### 3. Extraction Pipeline
- **`app/extraction/schemas.py`**
  - `ExtractedFeatures`: Partial feature dict (all optional, all nullable)
  - `Stage1Response`: The canonical output contract
  - `LLMExtractionOutput`: Raw LLM response before normalization

- **`app/extraction/normalizer.py`**
  - Converts ordinal user values → model tokens
  - Coerces numeric strings → int/float
  - Strict: raises NormalizationException on invalid values
  - Never guesses; null values stay null

- **`app/extraction/service.py`**
  - Main orchestrator
  - Takes latest message + accumulated state
  - Calls LLM → parses → normalizes → validates
  - Returns Stage1Response

### 4. Evaluation
- **`app/extraction/evaluator.py`**
  - Loads test cases from YAML
  - Runs both v1 and v2 on each case
  - Logs results to JSONL (version, input, output, validation_passed)
  - Supports mock responses for testing

### 5. Core Infrastructure
- **`app/core/config.py`**: Environment settings (OpenRouter API key)
- **`app/core/constants.py`**: MAX_MISSING_FIELDS_PER_TURN, temperatures, paths
- **`app/core/exceptions.py`**: ExtractionException, LLMException, ValidationException, etc.
- **`app/core/logging_config.py`**: File rotation, console output

## Data Contracts

### Input
```python
latest_message: str  # Current user input
accumulated_state: Dict[str, Any]  # Previously extracted features (may include None)
```

### Output: Stage1Response
```python
{
  "extracted_features": ExtractedFeatures,  # Normalized values
  "missing_features": List[str],             # Feature names still needed
  "is_complete": bool,                       # All 12 present?
  "reply": str                               # LLM-generated follow-up
}
```

## The 12 Required Features

| Numeric | Ordinal | Nominal |
|---------|---------|---------|
| GrLivArea | ExterQual | Neighborhood |
| TotalBsmtSF | KitchenQual | |
| GarageCars | BsmtQual | |
| YearBuilt | | |
| FullBath | | |
| TotRmsAbvGrd | | |
| LotArea | | |
| OverallQual | | |

## Ordinal Mappings

| User Input | Model Token |
|------------|-------------|
| None | None |
| Poor | Po |
| Fair | Fa |
| Typical, Average, Typical/Average | TA |
| Good | Gd |
| Excellent | Ex |

## Prompt Versioning

### extract_v1.txt
- **Strategy**: Concise, direct extraction
- **Focus**: Fast extraction of clearly stated values
- **Tone**: Straightforward

### extract_v2.txt
- **Strategy**: Conservative, constrained
- **Focus**: Strict validation; asks for clarification on ambiguity
- **Tone**: Disciplined, explicit

Both prompts:
- Return JSON with `extracted_features`, `missing_features`, `reply`
- Temperature = 0 (deterministic)
- Max 4 fields per follow-up question
- Never invent missing values

## Stateless Architecture

The service is stateless by design. All state is passed in and returned:
```python
# Call 1
state1 = {"GrLivArea": 3000}
result1 = extraction_service.extract("2 full baths", state1)
# result1.extracted_features now has {GrLivArea: 3000, FullBath: 2, ...}

# Call 2
state2 = {k: v for k, v in result1.extracted_features.dict().items() if v is not None}
result2 = extraction_service.extract("built in 2005", state2)
# result2 has all previous + new values
```

This allows:
- Session storage at any level (in-memory, Redis, database)
- Horizontal scaling (no sticky sessions)
- Easy testing (no setup/teardown)

## Artifacts and Logs

```
prompts/
  extract_v1.txt      # Extraction prompt variant 1
  extract_v2.txt      # Extraction prompt variant 2

evals/
  extraction_eval_cases.yaml   # Test cases for prompt evaluation

logs/
  app.log                      # Application logs (rotated)
  extraction_prompt_eval.jsonl # Evaluation results (append-only)
```

## Testing Strategy

- **Unit tests**: Mock OpenRouter, test normalizer, validate schemas
- **Integration tests**: Mock responses, test full pipeline
- **Evaluation**: Run both prompts on YAML test cases, compare results
- **No API integration tests in this slice** (real API testing is manual or CI-only)

## Design Principles

1. **Contract-first**: Features defined centrally; all modules import from contracts
2. **Strict validation**: Reject invalid input early; never silently invent values
3. **Stateless service**: Caller manages state accumulation
4. **Mock-first testing**: No real API calls during test runs
5. **Deterministic extraction**: Temperature = 0; same input = same output
6. **Logged not printed**: Structured logging with file rotation
7. **Prompt versioning**: Evaluate and compare extraction strategies

## What's NOT in This Slice

- FastAPI: No REST API yet (next slice)
- ML prediction: No artifacts, no model loading
- Interpretation stage: No Stage 2 LLM
- Streamlit UI: No web interface
- Database: No persistence layer
- Fallback providers: No multi-provider support
- Streaming responses: Blocking call-response only

## Testing Coverage

- Normalization: Numeric, ordinal, nominal values
- Extraction: Basic, with accumulated state, with errors
- Completeness: Missing features detection, is_complete flag
- Schemas: Pydantic validation of Stage1Response
- Evaluation: Test case loading, JSONL logging, error handling

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| OpenRouter API outage | Manual retry; fallback in next slice |
| Malformed LLM JSON | Validation + exception; logged |
| Prompt drift (v1 vs v2) | Evaluation on fixed test cases |
| Normalization errors | Graceful null; logged as warning |
| State mutation bugs | Immutable, explicit accumulation |

- data leakage
- prompt reliability
- missing values
- deployment