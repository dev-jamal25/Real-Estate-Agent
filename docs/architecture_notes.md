"""Architecture and design decisions."""
# TODO: Document high-level architecture, data flow, design choices

## Goal
One-paragraph project goal.

## High-Level Flow
User query -> LLM Stage 1 -> validation/completeness -> ML prediction -> LLM Stage 2 -> API response -> Streamlit UI

## Main Components
- FastAPI
- Streamlit
- LLM extraction
- ML pipeline
- LLM interpretation

## API Design
- POST /predict
- GET /health

## State and Loop Logic
How incomplete inputs are returned to the user and resubmitted until all required features are present.

## Data Contracts
Main request/response Pydantic models.

## Artifacts
Where model, stats, prompts, and logs live.

## Risks
- data leakage
- prompt reliability
- missing values
- deployment