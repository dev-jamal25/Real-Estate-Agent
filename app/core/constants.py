"""Application constants for Stage 1 extraction."""

# LLM Configuration
LLM_TEMPERATURE = 0  # Deterministic extraction

# Maximum missing fields to ask for in one turn
MAX_MISSING_FIELDS_PER_TURN = 4

# Prompt evaluation settings
EVAL_LOG_FILE = "logs/extraction_prompt_eval.jsonl"

