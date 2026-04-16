# Architecture Decision Records (ADRs) - Stage 1 Extraction

## ADR 1: OpenRouter Only (No Multi-Provider, No Gemini)

**Context**: Initial implementation must be focused and testable.

**Decision**: Use OpenRouter + elephant-alpha model exclusively for this slice. No Gemini SDK, no fallback logic, no provider abstraction.

**Rationale**:
- Simplicity: Direct HTTP client is easier to mock for testing
- Reduced scope: No need for multi-provider abstraction yet
- Mock-first: Tests don't need real API keys
- Clarity: Single responsibility for the LLM layer

**Consequences**:
- Future multi-provider support will require refactoring `openrouter_client.py`
- No graceful degradation if OpenRouter is unavailable
- Acceptable for this slice; add fallback in next iteration if needed

---

## ADR 2: Structured JSON Output from LLM

**Context**: We need deterministic extraction of property features.

**Decision**: Require the LLM to return JSON with `extracted_features`, `missing_features`, and `reply` fields.

**Rationale**:
- Parseable: Easy to extract and validate structured output
- Deterministic: Temperature = 0 for consistency
- Validatable: Pydantic models enforce schema
- Separates concerns: LLM focuses on content; code focuses on validation/normalization

**Consequences**:
- Stricter LLM prompts required
- JSON parsing errors will fail the request (by design)
- Easier to test and debug

---

## ADR 3: Stateless Service, Accumulation at Caller

**Context**: The extraction service will be called repeatedly in conversation flow.

**Decision**: Keep the service stateless. Callers accumulate feature state and pass it on each call.

**Rationale**:
- Composability: Works with any state management (session, database, in-memory)
- Testability: No session setup needed in tests
- Clarity: Feature flow is explicit at the API boundary
- Flexibility: Each turn can use different logic to manage state

**Consequences**:
- Caller must manage state merging (done in FastAPI later)
- No implicit session storage in the extraction layer
- Easier to reason about and test

---

## ADR 4: Prompt Versioning with Explicit Evaluation

**Context**: We want to test different extraction strategies before committing to one.

**Decision**: Keep v1 and v2 as distinct prompt files. Evaluate both on test cases. Log results to JSONL.

**Rationale**:
- Comparison: Can measure which variant extracts better
- Discoverability: Test cases live in `evals/` as YAML
- Traceability: JSONL logs version, input, output, success
- Iteration: Easy to add v3, v4 later

**Consequences**:
- Must maintain two prompt variants
- Evaluation is manual/on-demand (not automated on each run)
- JSONL logs are append-only; old runs remain

---

## ADR 5: Ordinal Mapping with User-Friendly Tokens

**Context**: Quality fields (ExterQual, KitchenQual, BsmtQual) use ordinal values.

**Decision**: Users see natural language (Poor, Fair, Good, Excellent). Model sees short tokens (Po, Fa, Gd, Ex).

**Rationale**:
- UX: Users speak naturally
- Compact: Model features are concise
- Mappings: Centralized in `property_features.py`
- Clarity: Schema is explicit about what maps to what

**Consequences**:
- Normalizer must handle case-insensitive matching
- Tests must verify mapping is bidirectional-aware
- Future: If mappings change, update in one place

---

## ADR 6: Contract-First Feature Definition

**Context**: Features are scattered and hard to track.

**Decision**: Single source of truth: `app/contracts/property_features.py`. Defines:
- The exact 12 required features
- Feature groups (numeric, ordinal, nominal)
- Ordinal mappings
- User-friendly descriptions

**Rationale**:
- DRY: One place to change feature specs
- Discoverability: Contracts live at the package boundary
- Testing: Easy to import and assert against
- Documentation: Self-documenting code

**Consequences**:
- Changes to features require updating the contract
- All modules import from contracts
- Easier to add validation rules later

---

## ADR 7: Mock-First Testing

**Context**: Tests must run without OpenRouter API access.

**Decision**: All tests use mock LLM responses. Real API testing is manual or CI-only (if API key provided).

**Rationale**:
- Speed: Tests run in milliseconds
- Isolation: No network dependencies
- Repeatability: Same mock response every time
- Cost: No API charges for test runs

**Consequences**:
- Must manually test against real API occasionally
- Mock responses must be realistic
- Tests don't catch API format changes (mitigated by integration tests in next slice)

---

## ADR 8: Logging Instead of Print

**Context**: Observability and debugging.

**Decision**: Use Python `logging` module exclusively. No `print()` for operational behavior.

**Rationale**:
- Levels: Can adjust verbosity without code changes
- Handlers: Can route logs to file, console, external services
- Tracing: Structured context (module, level, timestamp)
- Testing: Log assertions work cleanly

**Consequences**:
- Tests must configure logging if they need to capture logs
- Slightly more verbose than print()
- Evaluation results are JSONL (structured logs, not text logs)

---

## ADR 9: Pydantic for Schema Validation

**Context**: Must validate LLM output and extracted features.

**Decision**: Use Pydantic BaseModel for all data classes (ExtractedFeatures, Stage1Response, LLMExtractionOutput).

**Rationale**:
- Validation: Catches type errors early
- Docs: Schemas are self-documenting
- Serialization: Easy JSON conversions
- IDE support: Type hints enable autocomplete

**Consequences**:
- Adds pydantic dependency
- Strict mode (reject extra fields) must be configured per schema
- Easier to add validators later

---

## What We're NOT Doing (Yet)

- FastAPI: No HTTP routes in this slice
- ML prediction: No artifact loading
- Interpretation (Stage 2): Separate LLM stage
- Database: No session storage
- Fallback: No multi-provider failover
- Streaming: Blocking request-response
- Caching: No LLM response caching (simplicity first)

These are all valid for future iterations. This slice is MVP-scoped.
