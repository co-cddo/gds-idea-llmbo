from .adapters import AnthropicAdapter, MistralFunctionAdapter
from .batch_inferer import BatchInferer
from .models import (
    Manifest,
    ModelInput,
    ToolChoice,
)
from .registry import ModelAdapterRegistry
from .structured_batch_inferer import StructuredBatchInferer

# Register the model adapters
ModelAdapterRegistry.register(r"(anthropic|claude)", AnthropicAdapter)
ModelAdapterRegistry.register(r"(mistral|mixtral)", MistralFunctionAdapter)

__all__ = [
    "Manifest",
    "ToolChoice",
    "ModelInput",
    "BatchInferer",
    "StructuredBatchInferer",
    "ModelAdapterRegistry",
]
