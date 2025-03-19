import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from ..models import ModelInput

# TODO add better error messages for the notimplemented functions linking to
# the documentation when its written.


class ModelProviderAdapter(ABC):
    """Abstract interface for model-specific adapters.

    Base interface for all model adapters, providing model specific methods to prepare
    requests and handle responses according to provider-specific formats.

    If a model does not support tool use, you need only provide a prepare_model_input
    """

    logger = logging.getLogger(f"{__name__}.ModelProviderAdapter")

    @classmethod
    @abstractmethod
    def prepare_model_input(
        cls, model_input: ModelInput, output_model: type[BaseModel] | None = None
    ) -> ModelInput:
        """Prepare model input with provider-specific configurations.

        Args:
            model_input: Base model input configuration
            output_model: Optional Pydantic model for structured output

        Returns:
            ModelInput: Prepared with provider-specific configurations
        """
        pass

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        """Build a tool definition in the provider's specific format.

        Args:
            output_model: Pydantic model to convert to a tool definition

        Returns:
            Dict: Tool definition in provider-specific format

        Raises:
            NotImplementedError: If the provider doesn't support tools
        """
        cls.logger.error(f"{cls.__name__} does not support tool definitions")
        raise NotImplementedError(f"{cls.__name__} does not support tool definitions.")

    @classmethod
    def validate_result(
        cls, result: dict[str, Any], output_model: type[BaseModel]
    ) -> BaseModel | None:
        """Parse and validate model output.

        Args:
            result: Raw model output to process
            output_model: Optional Pydantic model for validation
                        (None if structured output not required)

        Returns:
            Validated instance of the BaseModel or None if invalid

        Raises:
            NotImplementedError: If structured output validation is not supported
                                but output_model is provided
        """
        cls.logger.error(
            f"{cls.__name__} does not support structured output validation"
        )
        raise NotImplementedError(
            f"{cls.__name__} does not support structured output validation."
        )


class DefaultAdapter(ModelProviderAdapter):
    """This adapter exists as a fall back for model registry.

    It performs no actions.
    """

    # Class-level logger for the default adapter
    logger = logging.getLogger(f"{__name__}.DefaultAdapter")
    _logged_default_warning = False

    @classmethod
    def prepare_model_input(
        cls, model_input: ModelInput, output_model: type[BaseModel] | None = None
    ) -> ModelInput:
        """Default implementation just returns the input unchanged.

        Args:
            model_input: Base model input configuration
            output_model: Optional Pydantic model for structured output

        Returns:
            ModelInput: Unchanged input model
        """
        if not cls._logged_default_warning:
            cls.logger.warning(
                "Using DefaultAdapter for model input preparation. "
                "This is likely not what you want for production use. "
                "This issue will only be logged once."
            )
            cls._logged_default_warning = True
        return model_input
