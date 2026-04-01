# Spec: SEA LLM Client & Explicit Provider Registry

**Date**: 2026-04-01  
**Status**: Draft  
**Topic**: Refactoring LLM client orchestration in `Self-Evolving-Agent` to support explicit providers and role-based configuration via JSON.

## 1. Overview
Current LLM client implementation is tied to a specific OpenAI class and lacks a clean way to swap backends (e.g., NVIDIA NIM, vLLM) based on agent roles (Coder vs. Summarizer). This spec introduces a `ProviderRegistry` and `UnifiedLLMClient` that use explicit JSON configuration.

## 2. Requirements & Constraints
- **Explicit Provider Mapping**: No auto-parsing of model names. Roles must point to a specific provider ID.
- **JSON-Driven Config**: Configuration of providers and roles is handled via JSON strings in environment variables.
- **Compatibility**: Use standard `response_format={"type": "json_schema", ...}` for broad support across Open-Source and NVIDIA endpoints (avoiding strict `beta...parse`).
- **Resilience**: Implement dynamic retry arguments with sensible defaults.
- **Environment Focus**: Only `Self-Evolving-Agent/.env.example` will be modified; active `.env` files are left to the user.

## 3. Architecture

### 3.1 Configuration Keys
- `SEA_PROVIDERS`: A JSON array defining backends.
  ```json
  [
    {"id": "nvidia", "name": "NVIDIA NIM", "base_url": "...", "api_key_env": "NVIDIA_API_KEY", "max_retries": 3},
    {"id": "openai", "name": "OpenAI Cloud", "base_url": "...", "api_key_env": "OPENAI_API_KEY"}
  ]
  ```
- `SEA_ROLE_CODER`: JSON object mapping the coder role to a provider/model.
  ```json
  {"provider": "nvidia", "model": "nvdev/nvidia/llama-3.1-nemotron-ultra-253b-v1"}
  ```
- `SEA_ROLE_SUMMARIZER`: JSON object mapping the summarizer role.

### 3.2 Core Components

#### `LLMProviderRegistry`
- Loads `SEA_PROVIDERS` and resolves `api_key_env` to actual environment variables.
- Factory method to produce an authenticated `OpenAI` client for a given provider ID.

#### `UnifiedLLMClient`
- Implemented as a cleaner replacement/wrapper for role-specific logic.
- Takes a `role` name on initialization (or method call).
- Looks up the role config -> looks up the provider -> executes the call.

## 4. Error Handling & Logging
- **Validation**: Error if `provider_id` in a role doesn't exist in the registry.
- **Dynamic Retries**: `max_retries` pulled from provider config, defaulting to `2` if not specified.
- **Streaming (Future Phase)**: Design allows for an optional `stream` flag in provider config to log reasoning traces.

## 5. Success Criteria
1. `uv sync` resolves correctly (handled in main KernelBench repo).
2. `test_provider_registry.py` verifies correct lookup and environment variable resolution.
3. Smoke-test script demonstrates switching from OpenAI to NVIDIA by changing role JSONs.
