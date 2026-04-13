# Copilot Instructions — AI Real Estate Agent

## Project Goal
Build an AI real estate agent that:
1. extracts structured property features from natural language using an LLM,
2. validates them with Pydantic,
3. predicts house price using a trained ML pipeline,
4. interprets the prediction with a second LLM stage,
5. exposes the workflow through FastAPI,
6. supports user review/filling of missing values in Streamlit.

## Core Architecture Rules
- Keep Stage 1 extraction and Stage 2 interpretation separate.
- Do not let the LLM silently invent missing values.
- Missing required model features must be surfaced explicitly.
- Prediction must run only when all required features are present.
- API/business logic/ML/LLM concerns must remain separated.
- Keep route handlers thin.

## API Rules
- Main business endpoint: POST /predict
- Support endpoint: GET /health
- POST /predict should:
  - accept free-text query
  - run extraction
  - validate extracted fields
  - return missing fields if incomplete
  - run prediction only if complete
  - return interpretation only after prediction
- Always return typed Pydantic responses.

## ML Rules
- Use Ames Housing dataset.
- Target is SalePrice.
- Use train/validation/test split.
- No data leakage.
- Fit imputers, encoders, and scalers on train only.
- Compare multiple model candidates.
- Evaluate best model on test exactly once.
- Serialized best model must be loadable by FastAPI at startup.

## Prompt Rules
- Store prompts in text files under /prompts.
- Keep 2 extraction prompt variants.
- Keep 1 interpretation prompt initially.
- Extraction prompt must be deterministic and strict.
- Prefer low temperature for extraction.
- Log prompt version, input, output, and validation result.

## Code Quality Rules
- Follow PEP 8.
- Use type hints for all function signatures.
- Keep files focused and not overloaded.
- Use descriptive snake_case names.
- Avoid unnecessary abstractions.
- Add docstrings for public functions and classes.
- Prefer clarity over cleverness.

## Error Handling Rules
- Never use bare except.
- Never swallow exceptions silently.
- Validate input at boundaries.
- Log full internal errors server-side.
- Return sanitized error messages to clients.

## Testing Rules
- Add tests for API behavior, validation logic, extraction parsing, and prediction flow.
- Test incomplete input cases and error paths.
- Test user correction loop behavior.
- Use pytest naming conventions.

## Security Rules
- Never hardcode API keys or secrets.
- Use environment variables.
- Keep .env out of version control.

## Collaboration Rules
- Keep PRs focused.
- Do not commit directly to main.
- Use conventional commit messages.