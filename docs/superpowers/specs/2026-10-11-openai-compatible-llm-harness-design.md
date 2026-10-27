# OpenAI-Compatible LLM Harness Design

## Goal

Make BCNLT's model-context evaluation harness reliably usable with both the local Bifrost runtime and hosted OpenAI models through an OpenAI-compatible chat-completions contract.

The milestone is complete when the harness can validate a configured endpoint with a real completion, evaluate deterministic probes across a Python project, preserve partial results when individual requests fail, and produce a sanitized report with enough provenance to reproduce the run.

## Scope

This milestone includes:

- A shared OpenAI-compatible provider implementation for Bifrost and OpenAI.
- Provider readiness checks using a small real completion request.
- Optional authentication and environment-backed custom headers.
- Deterministic directory evaluation with file and probe limits.
- Per-request latency, status, response parsing, and error reporting.
- Sanitized run metadata and corpus provenance in JSON reports.
- Automated wire-contract, failure-handling, CLI, and project-evaluation tests.
- An opt-in live Bifrost acceptance test.

This milestone does not include:

- Streaming responses.
- Tool or function calling.
- Automatic model discovery or model downloading.
- Provider-specific SDK dependencies.
- Retry or routing policy that competes with Bifrost's own routing behavior.
- Claims that BCNLT improves model accuracy unless a live comparative run demonstrates it.

## Architecture

### Provider configuration

Introduce an immutable provider configuration containing:

- provider profile: `bifrost` or `openai`
- base URL
- model ID
- timeout
- maximum output tokens
- optional bearer token resolved from a named environment variable
- optional custom headers whose values are resolved from named environment variables

Configuration objects may expose sanitized metadata, but must never expose resolved secret values. Reports record environment variable names only when useful and omit authorization header values.

### Shared OpenAI-compatible model

Use one `OpenAICompatibleChatModel` for both Bifrost and OpenAI. It sends non-streaming requests to `{base_url}/chat/completions` using the standard messages shape.

The implementation must:

- omit `Authorization` when no bearer token is configured
- include environment-backed custom headers when configured
- set deterministic generation parameters where supported
- constrain output length
- validate HTTP status, JSON decoding, response shape, and textual content
- classify failures into stable categories suitable for reports
- measure request latency

The profile controls validation rules rather than creating duplicate transports. The OpenAI profile requires an API key. The Bifrost profile permits unauthenticated local use and optional custom authentication headers.

### Readiness check

The readiness check sends a minimal deterministic prompt through the same chat-completions path used by evaluation. It succeeds only when the endpoint returns valid assistant text matching the expected acknowledgement.

Connection success alone is insufficient. An unreachable endpoint, authentication failure, unknown model, malformed response, or unusable completion fails readiness with a concise diagnostic before a full run starts.

### Project evaluation

Directory inputs are discovered recursively using sorted `*.py` paths. Generated directories and common environment/cache directories are excluded. Optional file and probe limits bound cost and runtime.

For each selected definition probe, the harness evaluates the same question twice:

1. with standard source context
2. with BCNLT semantic context

Each request produces a structured observation containing answer text, parsed line number, correctness, elapsed time, and any failure category. A request failure does not discard completed observations or stop unrelated probes.

The run stops before probe evaluation only when the explicit readiness check fails.

### Answer contract

The prompt requests a compact JSON answer containing a definition line or an unknown result. Parsing accepts the requested JSON form first and a strict `line N` fallback second. Numbers embedded in unrelated prose do not count as correct answers.

Malformed or ambiguous answers are recorded as parse failures rather than silently graded as incorrect model knowledge.

## CLI Design

The CLI supports:

- `--model-eval-provider oracle|bifrost|openai`
- `--model-eval-base-url URL`
- `--model-eval-model MODEL`
- `--model-eval-api-key-env ENV_NAME`
- repeatable `--model-eval-header-env HEADER=ENV_NAME`
- `--model-eval-timeout SECONDS`
- `--model-eval-max-output-tokens COUNT`
- `--model-eval-smoke`
- `--model-eval-max-files COUNT`
- `--model-eval-max-probes COUNT`

The existing oracle remains the deterministic offline default. Bifrost and OpenAI require an explicit model ID. OpenAI defaults to the hosted OpenAI base URL; Bifrost requires an explicit base URL so the harness never guesses which local service to call.

`--model-eval-smoke` performs readiness only. A non-oracle model evaluation performs readiness before starting unless an explicit future option disables it; this milestone does not add such a bypass.

Directory input is accepted for model evaluation. Output paths remain explicit and parent directories must already exist.

## Report Schema

Reports include:

- schema version
- start time and elapsed duration
- input root or file
- selected files
- provider profile, sanitized base URL, model ID, and timeout
- repository commit when available
- dirty-worktree boolean when available
- readiness result and latency
- probe counts, request counts, success counts, failure counts, and failure categories
- standard-context and BCNLT-context accuracy over successfully parsed answers
- per-probe answers, parsed values, correctness, latency, and errors

Accuracy denominators are explicit. The report separately exposes total probes, successful model responses, parseable answers, and correct answers so infrastructure failures cannot masquerade as poor model quality.

## Error Handling

Stable error categories include:

- `connection_error`
- `timeout`
- `authentication_error`
- `http_error`
- `invalid_json`
- `invalid_response_shape`
- `empty_response`
- `answer_parse_error`

Diagnostics may include a bounded, sanitized message. Authorization values, API keys, custom secret header values, and complete server error bodies are never persisted.

## Testing Strategy

Implementation follows red-green-refactor cycles.

Automated tests use a local HTTP server to verify the real wire contract rather than only injecting a fake transport. Tests cover:

- Bifrost requests without authentication
- OpenAI bearer authentication
- environment-backed custom headers
- endpoint and payload construction
- readiness success and each readiness failure class
- valid JSON answers and strict `line N` fallback
- malformed, empty, and ambiguous answers
- HTTP errors, timeouts, and invalid response bodies
- continuation after individual probe failures
- deterministic project file discovery and limits
- sanitized report metadata
- CLI smoke mode and directory evaluation

An opt-in live Bifrost test is enabled only when the required URL and model environment variables are present. Its absence skips the live test without weakening the deterministic suite.

## Acceptance Criteria

The milestone is accepted when:

1. All deterministic tests pass.
2. A local HTTP acceptance server proves the complete CLI-to-HTTP-to-report path.
3. The smoke command gives actionable failures for unreachable, unauthorized, unknown-model, and malformed-response cases.
4. A project evaluation preserves partial results when one request fails.
5. Generated reports contain no configured secret values.
6. An opt-in live Bifrost smoke test command is documented.
7. If a Bifrost endpoint and model are available, the live smoke test is executed and its actual result is reported without implying unmeasured quality gains.

## Follow-On Work

After this milestone, BCNLT can run the documented real-project comparison with either Bifrost or OpenAI. Later work may add concurrency, retries, streaming, richer question types, token-usage accounting, or additional OpenAI-compatible endpoints, but those are outside this design.
