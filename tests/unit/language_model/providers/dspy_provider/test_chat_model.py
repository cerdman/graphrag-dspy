# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for DSPy chat model provider."""

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestDSPyChatModel:
    """Tests for DSPyChatModel."""

    def test_import_dspy_chat_model(self):
        """Test that DSPyChatModel can be imported."""
        from graphrag.language_model.providers.dspy.chat_model import (
            DSPyChatModel,
        )

        assert DSPyChatModel is not None

    def test_dspy_chat_model_initialization(self):
        """Test DSPyChatModel initialization."""
        from graphrag.language_model.providers.dspy.chat_model import (
            DSPyChatModel,
        )
        from graphrag.config.models.language_model_config import (
            LanguageModelConfig,
        )

        config = LanguageModelConfig(
            type="dspy_chat",
            model="gpt-4",  # Use real model name for tiktoken
            model_provider="openai",
            api_key="test-key",
        )

        with patch("dspy.LM"):
            with patch("dspy.configure"):
                model = DSPyChatModel(name="test", config=config)
                assert model.name == "test"
                assert model.config == config

    def test_dspy_chat_model_has_chat_methods(self):
        """Test that DSPyChatModel has required ChatModel methods."""
        from graphrag.language_model.providers.dspy.chat_model import (
            DSPyChatModel,
        )

        # Check methods exist
        assert hasattr(DSPyChatModel, "achat")
        assert hasattr(DSPyChatModel, "chat")
        assert hasattr(DSPyChatModel, "achat_stream")
        assert hasattr(DSPyChatModel, "chat_stream")


class TestDSPyChatModelProviders:
    """Tests for different provider configurations."""

    @patch("dspy.LM")
    @patch("dspy.configure")
    def test_claude_provider_setup(self, mock_configure, mock_lm):
        """Test Claude provider initialization."""
        from graphrag.language_model.providers.dspy.chat_model import (
            DSPyChatModel,
        )
        from graphrag.config.models.language_model_config import (
            LanguageModelConfig,
        )

        config = LanguageModelConfig(
            type="dspy_chat",
            model="claude-sonnet-4",
            model_provider="anthropic",
            api_key="test-key",
            max_tokens=4096,
            temperature=0.0,
            encoding_model="cl100k_base",  # Explicit encoding for Claude
        )

        model = DSPyChatModel(name="claude_test", config=config)

        # Verify LM was called with anthropic provider
        mock_lm.assert_called_once()
        call_kwargs = mock_lm.call_args[1]
        # Model string should be "anthropic/claude-sonnet-4"
        assert "claude-sonnet-4" in call_kwargs.get("model", "")
        assert "anthropic" in call_kwargs.get("model", "")
        assert call_kwargs["api_key"] == "test-key"

    @patch("dspy.LM")
    @patch("dspy.configure")
    def test_openai_provider_setup(self, mock_configure, mock_lm):
        """Test OpenAI provider initialization."""
        from graphrag.language_model.providers.dspy.chat_model import (
            DSPyChatModel,
        )
        from graphrag.config.models.language_model_config import (
            LanguageModelConfig,
        )

        config = LanguageModelConfig(
            type="dspy_chat",
            model="gpt-4",
            model_provider="openai",
            api_key="test-key",
        )

        model = DSPyChatModel(name="openai_test", config=config)

        # Verify LM was called with openai provider
        mock_lm.assert_called_once()

    @patch("dspy.LM")
    @patch("dspy.configure")
    def test_azure_provider_setup(self, mock_configure, mock_lm):
        """Test Azure OpenAI provider initialization."""
        from graphrag.language_model.providers.dspy.chat_model import (
            DSPyChatModel,
        )
        from graphrag.config.models.language_model_config import (
            LanguageModelConfig,
        )

        config = LanguageModelConfig(
            type="dspy_chat",
            model="gpt-4",
            model_provider="azure",
            deployment_name="my-deployment",
            api_base="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            api_key="test-key",
        )

        model = DSPyChatModel(name="azure_test", config=config)

        # Verify LM was called with azure provider
        mock_lm.assert_called_once()


class TestDSPyModelResponse:
    """Tests for DSPy model response structures."""

    def test_dspy_model_output_structure(self):
        """Test DSPyModelOutput structure."""
        from graphrag.language_model.providers.dspy.chat_model import (
            DSPyModelOutput,
        )

        output = DSPyModelOutput(content="Test response")
        assert output.content == "Test response"
        assert output.full_response is None

    def test_dspy_model_response_structure(self):
        """Test DSPyModelResponse structure."""
        from graphrag.language_model.providers.dspy.chat_model import (
            DSPyModelResponse,
            DSPyModelOutput,
        )

        response = DSPyModelResponse(
            output=DSPyModelOutput(content="Test"),
            parsed_response=None,
            history=[{"role": "user", "content": "Hello"}],
        )

        assert response.output.content == "Test"
        assert len(response.history) == 1


class TestModelFactoryIntegration:
    """Tests for ModelFactory integration."""

    def test_dspy_chat_registered_in_factory(self):
        """Test that DSPyChat is registered in ModelFactory."""
        from graphrag.language_model.factory import ModelFactory
        from graphrag.config.enums import ModelType

        assert ModelFactory.is_supported_chat_model(ModelType.DSPyChat.value)
        assert ModelType.DSPyChat.value in ModelFactory.get_chat_models()
