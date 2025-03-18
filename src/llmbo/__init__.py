from .adapters import AnthropicAdapter
from .llmbo import (
    BatchInferer,
    StructuredBatchInferer,
)
from .models import (
    Manifest,
    ModelInput,
    ToolChoice,
)
from .registry import ModelAdapterRegistry

# Register the model adapters
ModelAdapterRegistry.register(r"(anthropic|claude)", AnthropicAdapter)

__all__ = [
    "Manifest",
    "ToolChoice",
    "ModelInput",
    "BatchInferer",
    "StructuredBatchInferer",
]
