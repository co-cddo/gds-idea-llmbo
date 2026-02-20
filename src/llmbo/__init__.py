from .adapters import AnthropicAdapter, MistralAdapter
from .batch_inferer import BatchInferer
from .models import (
    Manifest,
    ModelInput,
    ToolChoice,
)
from .registry import ModelAdapterRegistry
from .structured_batch_inferer import StructuredBatchInferer

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

# Register the model adapters
ModelAdapterRegistry.register(r"(anthropic|claude)", AnthropicAdapter)
ModelAdapterRegistry.register(r"(mistral|mixtral)", MistralAdapter)

__all__ = [
    "__version__",
    "Manifest",
    "ToolChoice",
    "ModelInput",
    "BatchInferer",
    "StructuredBatchInferer",
    "ModelAdapterRegistry",
]
