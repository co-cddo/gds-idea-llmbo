# AGENTS.md

Guidance for AI coding agents working in the `llmbo-bedrock` repository.
This is a Python library for AWS Bedrock batch inference with an adapter pattern
for multiple model providers (Claude, Mistral). Uses `uv` as the package manager.

## Build / Lint / Test Commands

```bash
# Setup
uv sync --all-extras --dev

# Run all tests (with coverage)
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_batch_inferer.py

# Run a single test function
uv run pytest tests/unit/test_batch_inferer.py::test_function_name

# Run only integration tests (require AWS access)
uv run pytest tests/integration/

# Lint (ruff with default config -- no custom rules configured)
ruff check .
ruff check --fix .

# Type check (mypy with default config -- no custom rules configured)
uv run mypy src/
```

CI runs pytest and ruff across Python 3.10-3.13. All PRs must pass both.
Coverage is enabled by default via `--cov=llmbo` in pyproject.toml addopts.

## Code Style

### Imports

Organize in three groups separated by blank lines, following PEP 8 / isort order:

1. Standard library (`import json`, `import os`, `from typing import Any`)
2. Third-party (`import boto3`, `from pydantic import BaseModel`)
3. Local project (`from .models import ModelInput`)

Within the package (`src/llmbo/`), always use **relative imports**:
```python
from .models import ModelInput
from ..models import Manifest
from .batch_inferer import BatchInferer
```

In tests, use **absolute imports**:
```python
from llmbo import BatchInferer, ModelInput
from llmbo.adapters import AnthropicAdapter
```

Prefer `from X import Y` over `import X`. Wrap multi-name imports in parens:
```python
from .models import (
    Manifest,
    ModelInput,
    ToolChoice,
)
```

### Type Annotations

Python >=3.10 is required. Use **modern syntax everywhere** -- no `typing` generics:

```python
# Correct
def foo(items: list[str], config: dict[str, Any]) -> int | None: ...
session: boto3.Session | None = None
output_model: type[BaseModel]

# Wrong -- do not use
from typing import Optional, List, Dict, Type
def foo(items: List[str]) -> Optional[int]: ...
```

Only import `Any` and `Literal` from `typing`. Use `X | None` (not `Optional[X]`),
`list[str]` (not `List[str]`), `dict[str, Any]` (not `Dict`), `type[Foo]` (not `Type`).

Use string-quoted return types for self-referential classmethods: `-> "BatchInferer"`.

Annotate all public method signatures (params + return). Private helpers may omit
return types if trivial.

### Naming Conventions

| Element             | Convention        | Examples                                    |
|---------------------|-------------------|---------------------------------------------|
| Classes             | `PascalCase`      | `BatchInferer`, `AnthropicAdapter`          |
| Functions / methods | `snake_case`      | `prepare_requests`, `push_requests_to_s3`   |
| Private methods     | `_snake_case`     | `_check_bucket`, `_write_requests_locally`  |
| Constants           | `UPPER_SNAKE`     | `VALID_FINISHED_STATUSES`, `MIN_BATCH_SIZE` |
| Instance attributes | `snake_case`      | `model_name`, `job_arn`, `file_name`        |
| Test functions      | `test_snake_case` | `test_init`, `test_validate_result_valid`   |

Exception: `Manifest` dataclass fields use `camelCase` (`totalRecordCount`) to
match the AWS API response directly for `Manifest(**json_data)` deserialization.

### Docstrings

Google-style with `Args:`, `Returns:`, `Raises:`, `Note:`, `Example:` sections.
Types go in parentheses after the parameter name:

```python
def create_job(self, inputs: dict[str, ModelInput]) -> dict[str, Any]:
    """Create a batch inference job.

    Args:
        inputs (dict[str, ModelInput]): Mapping of request ID to model input.

    Returns:
        dict[str, Any]: AWS API response.

    Raises:
        ValueError: If inputs has fewer than 100 items.
    """
```

Public methods get full docstrings. Private methods get a short one-liner.
Test functions and fixtures get one-line docstrings.

### Error Handling

Follow the **log-then-raise** pattern. Always log at `error` level before raising:

```python
self.logger.error(f"Bucket {bucket_name} is not accessible: {e}")
raise ValueError(f"Bucket {bucket_name} is not accessible") from e
```

Always chain exceptions with `raise ... from e` when re-raising.
Convert boto3 `ClientError` into domain exceptions (`ValueError`, `RuntimeError`).

In adapter `validate_result` methods, return `None` for soft failures instead of
raising, and log at `debug` level for diagnostics.

Common exception types used:
- `ValueError` -- validation failures (bad ARN, bucket, inputs, batch size)
- `RuntimeError` -- operational failures (job creation, AWS API errors)
- `FileNotFoundError` / `FileExistsError` -- missing expected files
- `NotImplementedError` -- abstract base class methods

### Logging

Use the standard library `logging` module. Each class defines a class-level logger:

```python
class MyClass:
    logger = logging.getLogger(f"{__name__}.MyClass")
```

Use f-strings in all log messages (not `%`-style). Log levels:
- `info` -- operational milestones (init, job creation, progress updates)
- `error` -- immediately before raising an exception
- `debug` -- detailed diagnostics in adapters
- `warning` -- degraded conditions (fallback adapter, duplicate registration)

Do not configure handlers/formatters in library code -- leave that to consumers.

### Formatting

No explicit line-length configured in ruff or any linter. Keep lines under ~100
characters where practical. Long f-strings can be split with implicit concatenation:

```python
self.logger.error(
    f"Bucket {bucket_name} is not in region [{region}] "
    f"as the model [{bucket_region}]"
)
```

## Architecture Overview

Source lives in `src/llmbo/`. Key classes:
- `BatchInferer` (`batch_inferer.py`) -- main orchestrator: prepare, upload, create, poll, download, load
- `StructuredBatchInferer` (`structured_batch_inferer.py`) -- extends BatchInferer with Pydantic validation
- `ModelInput` / `Manifest` (`models.py`) -- dataclasses for inputs and job manifests
- `ModelAdapterRegistry` (`registry.py`) -- regex-based adapter lookup
- `ModelProviderAdapter` (`adapters/base.py`) -- abstract base for model-specific behavior
- Concrete adapters: `AnthropicAdapter`, `MistralAdapter`, `MistralFunctionCallingAdapter`

New model adapters: create in `src/llmbo/adapters/`, inherit `ModelProviderAdapter`,
register in `src/llmbo/__init__.py` with `ModelAdapterRegistry.register(r"pattern", Adapter)`.

## Testing Patterns

Tests use **pytest** with plain functions (no classes). Shared fixtures in `tests/conftest.py`:
- `mock_boto3_session` -- patches `boto3.Session`, routes `.client()` to per-service mocks
- `mock_bedrock_client` / `mock_s3_client` / `mock_iam_client` -- pre-configured `MagicMock`s
- `sample_inputs` -- 100 `ModelInput` instances (minimum batch size)
- `batch_inferer` / `structured_batch_inferer` -- fully wired instances with mocked AWS
- `ExampleOutput` -- test Pydantic `BaseModel`

Use `unittest.mock.MagicMock` and `patch` for mocking. Assert with plain `assert`
and `pytest.raises(ExceptionType, match=r"regex")`. Verify logs with `caplog`.

## Pending Improvements

See `llmbo.md` for a planned `output_dir` parameter (local file path control) and a
`Manifest` dataclass bug where new Bedrock API fields cause `TypeError`.
See `IMPROVEMENTS.md` for a prioritized roadmap of 21 issues including removing the
hard `AWS_PROFILE` requirement, fixing input mutation, and converting `ModelInput` to Pydantic.
