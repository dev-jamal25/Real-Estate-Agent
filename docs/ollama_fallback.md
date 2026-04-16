# Ollama Fallback Implementation

## Overview

The LLM client now supports automatic fallback to local Ollama when Gemini credits are exhausted or when other provider errors occur. This ensures the system can continue operating even when Gemini API access is unavailable.

## Architecture

### Error Detection Flow

```
Gemini API Call
    ↓
Exception Caught
    ↓
_handle_gemini_error() - Maps to ProviderError types
    ↓
Error Classification
    ├─ ProviderQuotaError (429, quota exhausted, credits depleted)
    ├─ ProviderTimeoutError (deadline exceeded, timeouts)
    ├─ ProviderServerError (5xx, service unavailable)
    └─ ProviderUnavailableError (connection, SSL, etc.)
    ↓
Check Fallback Config
    ├─ If fallback enabled for error type
    │   └─ Call _try_ollama_fallback()
    └─ Else
        └─ Raise error
```

### Key Components

**1. GeminiClient.generate_stage1_response()**
- Primary entry point for Stage 1 extraction
- Attempts Gemini API call first
- Catches ALL exceptions and maps them to provider errors
- Routes to Ollama if appropriate fallback conditions are met

**2. GeminiClient._handle_gemini_error()**
- Maps google-genai SDK exceptions to custom ProviderError types
- Handles both typed exceptions (`google.api_core.exceptions.*`) and string pattern matching
- Detects credit exhaustion via:
  - `google.api_core.exceptions.ResourceExhausted`
  - "429" HTTP status code
  - "quota", "rate_limit", "resource exhausted" keywords

**3. GeminiClient._try_ollama_fallback()**
- Ensures Ollama client is configured
- Calls OllamaClient with same prompt
- Logs fallback attempt and result
- Re-raises if Ollama also fails

**4. OllamaClient.generate_stage1_response()**
- Makes HTTP POST request to Ollama /api/generate endpoint
- Returns LLM response as-is (should be valid JSON)
- Handles connection, timeout, and server errors

## Configuration

### Enable Fallback

In `app/config.py`:

```python
# Fallback behavior
llm_enable_fallback: bool = True              # Master switch for fallback
llm_fallback_on_quota: bool = True            # Fallback on 429/quota errors
llm_fallback_on_timeout: bool = True          # Fallback on deadline exceeded
llm_fallback_on_5xx: bool = True              # Fallback on 5xx server errors

# Ollama configuration
ollama_enabled: bool = True                   # Enable Ollama as fallback target
ollama_base_url: str = "http://localhost:11434"
ollama_model: str = "llama3.2:1b"             # or other installed model
ollama_timeout: int = 30                      # Request timeout in seconds
```

### Environment Setup

To use Ollama fallback:

1. **Install Ollama**: Download from https://ollama.ai
2. **Start Ollama**: Run `ollama serve` (default port 11434)
3. **Pull a model**: `ollama pull llama3.2:1b` (or preferred model)
4. **Configure API**: Set `OLLAMA_ENABLED=true` or use config defaults

## Error Detection Patterns

### Quota/Credits Exhausted (HTTP 429)
Detected by:
- `google.api_core.exceptions.ResourceExhausted`
- Error message contains "429"
- Error message contains "quota" or "rate_limit"
- Error message contains "resource exhausted"

**Response**: Attempt Ollama fallback (if `fallback_on_quota=True`)

### Timeout Errors
Detected by:
- `google.api_core.exceptions.DeadlineExceeded`
- Error message contains "deadline" or "timeout"

**Response**: Attempt Ollama fallback (if `fallback_on_timeout=True`)

### Server Errors (5xx)
Detected by:
- `google.api_core.exceptions.ServiceUnavailable`
- `google.api_core.exceptions.InternalServerError`
- Error message contains "503", "500", or "service unavailable"

**Response**: Attempt Ollama fallback (if `fallback_on_5xx=True`)

### Connection/Availability Issues
Detected by:
- Error message contains "connection", "unable to connect"
- Error message contains "ssl", "certificate"

**Response**: Attempt Ollama fallback (no config flag required)

## Logging

Fallback events are logged with appropriate levels:

```python
# Fallback triggered
logger.warning("Gemini quota/credits exhausted; attempting Ollama fallback: ...")

# Fallback attempt
logger.info("Attempting Stage 1 extraction with Ollama fallback (llama3.2:1b) at http://localhost:11434")

# Success
logger.info("Stage 1 response generated via Ollama fallback")

# Failure
logger.error("Ollama fallback failed: ...")
```

## Testing

Run fallback tests:

```bash
pytest tests/test_ollama_fallback.py -v
```

### Test Coverage

- ✅ Quota error triggers Ollama fallback
- ✅ google.api_core.exceptions.ResourceExhausted mapping
- ✅ Quota error raised if fallback disabled
- ✅ Timeout error triggers Ollama fallback
- ✅ Server error triggers Ollama fallback
- ✅ Error if both Gemini and Ollama fail
- ✅ Error if Ollama not configured but fallback requested
- ✅ Error detection for quota keywords
- ✅ Error detection for timeout keywords
- ✅ Error detection for server error keywords
- ✅ Error detection for connection keywords

## Usage Example

### Scenario: Gemini Credits Exhausted

```python
from app.llm.client import GeminiClient
from app.config import settings

# Initialize client with fallback enabled
client = GeminiClient(
    use_mock=False,
    api_key="sk-...",
    model_name="gemini-1.5-flash",
    enable_fallback=True,
    fallback_on_quota=True,
    fallback_on_timeout=False,
    fallback_on_5xx=False,
    ollama_enabled=True,
    ollama_base_url="http://localhost:11434",
    ollama_model="llama3.2:1b",
    ollama_timeout=30,
)

# When Gemini credits run out:
try:
    response = client.generate_stage1_response("Extract property features...")
    # Response is from Ollama fallback
    print(f"Got response via fallback: {response}")
except Exception as e:
    # Fallback also failed or was disabled
    print(f"Both providers failed: {e}")
```

### Scenario: Disable Fallback

```python
# For production with reliable Gemini access
client = GeminiClient(
    ...,
    enable_fallback=False,  # Disable fallback completely
    ...,
)

# Errors will not trigger fallback, will be raised immediately
```

## Monitoring

Track fallback usage by checking logs:

```bash
# See all fallback attempts
grep "attempting Ollama fallback" logs/*.log

# See fallback failures
grep "Ollama fallback failed" logs/*.log

# See Ollama configuration
grep "Ollama fallback configured" logs/*.log
```

## Considerations

### Performance
- Ollama runs locally, typically faster than network calls to Gemini
- May have lower quality responses depending on model chosen
- Use smaller, faster models (llama3.2:1b) for low latency

### Quality
- Response quality varies by Ollama model
- Test with your target models before relying on fallback
- Monitor for degraded extraction quality during fallback periods

### Cost Savings
- Fallback to free local Ollama saves Gemini API credits
- Useful for development/testing with limited API quotas
- Enables graceful degradation when credits exhausted

### Production Use
- Ensure Ollama is running and healthy
- Monitor system resources (local LLM is CPU/memory intensive)
- Configure appropriate timeouts for your infrastructure
- Consider running Ollama on separate machine for scale

## Future Enhancements

1. **Multiple Fallbacks**: Support multiple fallback providers in sequence
2. **Provider Switching**: Switch to cheaper provider automatically
3. **Response Caching**: Cache extractions to reduce provider calls
4. **Circuit Breaker**: Temporarily bypass failing providers
5. **Provider Health Checks**: Proactive health monitoring
6. **Metrics Dashboard**: Track provider usage and success rates
