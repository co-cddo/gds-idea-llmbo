import logging
import re
from re import Pattern

from .adapters.base import DefaultAdapter, ModelProviderAdapter


class ModelAdapterRegistry:
    """Registry for model provider adapters.

    This registry maps model name patterns to their corresponding adapter classes.
    Users can register custom adapters for new model providers or to override
    existing implementations.

    Example:
        >>> # Register a custom adapter for a new model
        >>> ModelAdapterRegistry.register("my-custom-model", MyCustomAdapter)
    """

    _adapters: list[tuple[Pattern, type[ModelProviderAdapter]]] = []
    logger = logging.getLogger(__name__)

    @classmethod
    def register(cls, pattern: str, adapter_class: type[ModelProviderAdapter]) -> None:
        """Register an adapter class for a specific model pattern.

        Args:
            pattern: Regex pattern to match against model names
            adapter_class: The adapter class to use for matching models

        Raises:
            TypeError: If adapter_class is not a subclass of ModelProviderAdapter
        """
        # Add type validation to ensure adapter_class is a proper ModelProviderAdapter
        if not issubclass(adapter_class, ModelProviderAdapter):
            raise TypeError(
                f"Adapter class must be a subclass of ModelProviderAdapter, "
                f"got {adapter_class.__name__}"
            )

        compiled_pattern = re.compile(pattern)

        # Check for duplicate pattern and log a warning
        for i, (existing_pattern, _) in enumerate(cls._adapters):
            if existing_pattern.pattern == compiled_pattern.pattern:
                cls.logger.warning(
                    f"Adapter for pattern '{pattern}' is being replaced with {adapter_class.__name__}"
                )
                # Remove the existing adapter with the same pattern
                cls._adapters.pop(i)
                break

        # Add new adapter to the beginning of the list for higher precedence
        cls._adapters.insert(0, (compiled_pattern, adapter_class))
        cls.logger.info(
            f"Registered adapter {adapter_class.__name__} for pattern '{pattern}'"
        )

    @classmethod
    def get_adapter(cls, model_name: str) -> type[ModelProviderAdapter]:
        """Get the appropriate adapter for a model name.

        Args:
            model_name: The model name/ID to find an adapter for

        Returns:
            An adapter class for the given model, or the default adapter if no pattern
            is found
        """
        for pattern, adapter in cls._adapters:
            if pattern.search(model_name):
                return adapter

        cls.logger.warning(
            f"No pattern found for {model_name}, returning default ModelAdapter. ",
            "This model is unsupported it may not work as expected.",
        )
        return DefaultAdapter
