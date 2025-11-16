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

        # Build kwargs for DSPy model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.config.max_tokens or 4096,
            "temperature": self.config.temperature or 0.0,
        }

        # Add API key if provided
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        # Configure based on provider
        if model_provider == "anthropic" or model.startswith("claude"):
            # Claude via Anthropic
            self._lm = dspy.Claude(
                model=model if "/" not in model else model.split("/")[-1],
                api_key=kwargs.get("api_key"),
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.0),
            )
        elif model_provider == "openai":
            # OpenAI
            self._lm = dspy.OpenAI(
                model=model,
                api_key=kwargs.get("api_key"),
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.0),
            )
        elif model_provider == "azure":
            # Azure OpenAI
            self._lm = dspy.AzureOpenAI(
                deployment_id=self.config.deployment_name or model,
                api_key=kwargs.get("api_key"),
                api_base=self.config.api_base,
                api_version=self.config.api_version,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.0),
            )
        else:
            # Generic LM (falls back to DSPy's default LM initialization)
            logger.warning(
                f"Unknown provider '{model_provider}', using DSPy default LM"
            )
            self._lm = dspy.LM(
                model=f"{model_provider}/{model}",
                **kwargs,
            )

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
