# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""DSPy LM adapter for GraphRAG ChatModel."""

import asyncio
from typing import Any

import dspy

from graphrag.language_model.protocol.base import ChatModel


class GraphRAGDSpyLM(dspy.LM):
    """
    DSPy Language Model adapter that wraps a GraphRAG ChatModel.

    This adapter allows GraphRAG's ChatModel implementations to be used
    as DSPy language models, enabling DSPy's prompt optimization and
    module composition features while maintaining compatibility with
    GraphRAG's existing LM infrastructure.
    """

    def __init__(
        self,
        chat_model: ChatModel,
        model: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the DSPy LM adapter.

        Args:
            chat_model: The GraphRAG ChatModel instance to wrap
            model: Optional model identifier (defaults to chat_model.config.model)
            **kwargs: Additional arguments passed to dspy.LM
        """
        self._chat_model = chat_model
        model_name = model or getattr(chat_model.config, "model", "graphrag-model")

        # Initialize DSPy LM with the model name
        super().__init__(model=model_name, **kwargs)

    def __call__(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Invoke the language model (synchronous wrapper for async chat).

        Args:
            prompt: Text prompt to send to the model
            messages: Chat messages in DSPy format
            **kwargs: Additional parameters for the model

        Returns:
            Dictionary with 'choices' key containing the response
        """
        # Run the async method in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to handle this differently
            # Create a new event loop in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self._async_call(prompt, messages, **kwargs)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                self._async_call(prompt, messages, **kwargs)
            )

    async def _async_call(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Async implementation of the LM call.

        Args:
            prompt: Text prompt to send to the model
            messages: Chat messages in DSPy format
            **kwargs: Additional parameters for the model

        Returns:
            Dictionary with 'choices' key containing the response
        """
        # Prepare the history if messages are provided
        history = None
        actual_prompt = prompt

        if messages:
            # Convert DSPy messages format to GraphRAG history format
            # DSPy messages: [{"role": "user/system/assistant", "content": "..."}]
            # GraphRAG expects the last user message as prompt, rest as history
            if len(messages) > 0:
                # Extract all but the last message as history
                if len(messages) > 1:
                    history = messages[:-1]

                # Use the last message as the prompt
                last_msg = messages[-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    actual_prompt = last_msg["content"]

        # Extract GraphRAG-specific parameters
        json_mode = kwargs.pop("json", False)
        json_model = kwargs.pop("json_model", None)
        name = kwargs.pop("name", None)

        # Call the underlying GraphRAG ChatModel
        response = await self._chat_model.achat(
            prompt=actual_prompt or "",
            history=history,
            json=json_mode,
            json_model=json_model,
            name=name,
            **kwargs,
        )

        # Convert GraphRAG ModelResponse to DSPy-compatible format
        # DSPy expects: {"choices": [{"message": {"content": "..."}}]}
        content = response.output.content or ""

        dspy_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": getattr(response, "metrics", {}),
            "_graphrag_response": response,  # Store original for debugging
        }

        # Store parsed response if available (for structured outputs)
        if response.parsed_response is not None:
            dspy_response["_parsed_response"] = response.parsed_response

        return dspy_response

    def inspect_history(self, n: int = 1) -> list[dict[str, Any]]:
        """
        Inspect the last n interactions with the model.

        Args:
            n: Number of recent interactions to return

        Returns:
            List of interaction dictionaries
        """
        # DSPy's LM base class handles history tracking
        # We can extend this if needed
        return super().inspect_history(n)
