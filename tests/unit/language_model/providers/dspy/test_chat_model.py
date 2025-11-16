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
            model="test-model",
            model_provider="openai",
            api_key="test-key",
        )

        with patch("dspy.OpenAI"):
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

    @patch("dspy.Claude")
    @patch("dspy.configure")
    def test_claude_provider_setup(self, mock_configure, mock_claude):
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
        )

        model = DSPyChatModel(name="claude_test", config=config)

        # Verify Claude was called
        mock_claude.assert_called_once()
        call_kwargs = mock_claude.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4"
        assert call_kwargs["api_key"] == "test-key"

    @patch("dspy.OpenAI")
    @patch("dspy.configure")
    def test_openai_provider_setup(self, mock_configure, mock_openai):
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

        # Verify OpenAI was called
        mock_openai.assert_called_once()

    @patch("dspy.AzureOpenAI")
    @patch("dspy.configure")
    def test_azure_provider_setup(self, mock_configure, mock_azure):
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

        # Verify AzureOpenAI was called
        mock_azure.assert_called_once()


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
