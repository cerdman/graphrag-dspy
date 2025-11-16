# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""DSPy-based Chat Model implementation for GraphRAG."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.response.base import ModelResponse as MR

logger = logging.getLogger(__name__)


class DSPyModelOutput(BaseModel):
    """Output from DSPy language model."""

    content: str = Field(description="Generated text content")
    full_response: None = Field(
        default=None, description="Full response from model, if available"
    )


class DSPyModelResponse(BaseModel):
    """Response from DSPy language model."""

    output: DSPyModelOutput = Field(description="Output from the model")
    parsed_response: BaseModel | None = Field(
        default=None, description="Parsed response from the model"
    )
    history: list = Field(
        default_factory=list,
        description="Conversation history including prompt and response",
    )


class DSPyChatModel:
    """DSPy-based Chat Model for GraphRAG.

    This model wraps DSPy's LM interface to provide a ChatModel-compatible
    interface for GraphRAG. It supports multiple providers including Claude.
    """

    def __init__(
        self,
        name: str,
        config: LanguageModelConfig,
        **kwargs: Any,
    ):
        """Initialize DSPy Chat Model.

        Args:
            name: Name of the model instance
            config: Language model configuration
            **kwargs: Additional keyword arguments
        """
        self.name = name
        self.config = config
        self._lm = None
        self._setup_dspy_model()

    def _setup_dspy_model(self) -> None:
        """Set up DSPy language model based on configuration."""
        import dspy

        model_provider = self.config.model_provider.lower()
        model = self.config.deployment_name or self.config.model

        # Build kwargs for DSPy LM (unified API in DSPy 3.0+)
        kwargs: dict[str, Any] = {
            "max_tokens": self.config.max_tokens or 4096,
            "temperature": self.config.temperature or 0.0,
        }

        # Add API key if provided
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        # Build model string for DSPy 3.0 unified LM API
        # Format: "provider/model_name"
        if model_provider == "anthropic" or model.startswith("claude"):
            # Claude via Anthropic
            model_str = f"anthropic/{model}" if "/" not in model else model
        elif model_provider == "openai":
            # OpenAI
            model_str = f"openai/{model}" if "/" not in model else model
        elif model_provider == "azure":
            # Azure OpenAI - requires additional config
            model_str = f"azure/{self.config.deployment_name or model}"
            if self.config.api_base:
                kwargs["api_base"] = self.config.api_base
            if self.config.api_version:
                kwargs["api_version"] = self.config.api_version
        else:
            # Generic provider
            model_str = f"{model_provider}/{model}"
            logger.warning(
                f"Using generic DSPy LM with provider '{model_provider}'"
            )

        # Create unified LM instance (DSPy 3.0+ API)
        self._lm = dspy.LM(model=model_str, **kwargs)

        # Configure DSPy to use this LM
        dspy.configure(lm=self._lm)

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> MR:
        """Generate async response for prompt.

        Args:
            prompt: The prompt to generate a response for
            history: Optional conversation history
            **kwargs: Additional keyword arguments

        Returns:
            ModelResponse: The generated response
        """
        # DSPy doesn't have native async support, so we run sync in thread
        import asyncio

        return await asyncio.to_thread(self.chat, prompt, history, **kwargs)

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate async streaming response.

        Args:
            prompt: The prompt to generate a response for
            history: Optional conversation history
            **kwargs: Additional keyword arguments

        Yields:
            Chunks of the response
        """
        # DSPy doesn't support streaming, so we return full response
        response = await self.achat(prompt, history, **kwargs)
        yield response.output.content

    def chat(self, prompt: str, history: list | None = None, **kwargs: Any) -> MR:
        """Generate response for prompt.

        Args:
            prompt: The prompt to generate a response for
            history: Optional conversation history
            **kwargs: Additional keyword arguments

        Returns:
            ModelResponse: The generated response
        """
        import dspy

        messages: list[dict[str, str]] = history or []
        messages.append({"role": "user", "content": prompt})

        # Build context from history if present
        if len(messages) > 1:
            # Extract previous messages for context
            context_parts = []
            for msg in messages[:-1]:  # All except current prompt
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content}")

            full_prompt = "\n".join(context_parts) + f"\nuser: {prompt}"
        else:
            full_prompt = prompt

        # Call DSPy LM directly
        try:
            response_text = self._lm(full_prompt)
        except Exception as e:
            logger.exception("DSPy LM call failed")
            raise RuntimeError(f"DSPy LM call failed: {e}") from e

        # Add response to history
        messages.append({"role": "assistant", "content": response_text})

        return DSPyModelResponse(
            output=DSPyModelOutput(content=response_text),
            parsed_response=None,
            history=messages,
        )

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None]:
        """Generate streaming response.

        Args:
            prompt: The prompt to generate a response for
            history: Optional conversation history
            **kwargs: Additional keyword arguments

        Yields:
            Chunks of the response
        """
        # DSPy doesn't support streaming, so we return full response
        response = self.chat(prompt, history, **kwargs)
        yield response.output.content
