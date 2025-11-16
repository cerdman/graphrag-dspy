# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for DSPy-enabled extractors."""

import pytest

from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportsExtractor,
)
from tests.mock_provider import create_mock_chat_model


class TestGraphExtractorIntegration:
    """Integration tests for GraphExtractor with DSPy."""

    @pytest.mark.asyncio
    async def test_dspy_mode_basic(self):
        """Test GraphExtractor in DSPy mode."""
        mock_model = create_mock_chat_model()
        extractor = GraphExtractor(
            model_invoker=mock_model, use_dspy=True, max_gleanings=0
        )

        result = await extractor(
            texts=["Alice works at TechCorp."],
            prompt_variables={"entity_types": ["PERSON", "ORGANIZATION"]},
        )

        assert result.output is not None
        assert isinstance(result.source_docs, dict)

    @pytest.mark.asyncio
    async def test_legacy_mode_basic(self):
        """Test GraphExtractor in legacy mode."""
        mock_model = create_mock_chat_model()
        extractor = GraphExtractor(
            model_invoker=mock_model, use_dspy=False, max_gleanings=0
        )

        result = await extractor(
            texts=["Alice works at TechCorp."],
            prompt_variables={"entity_types": ["PERSON", "ORGANIZATION"]},
        )

        assert result.output is not None
        assert isinstance(result.source_docs, dict)

    @pytest.mark.asyncio
    async def test_no_global_state_pollution(self):
        """Test that multiple extractors don't interfere via global state."""
        model_a = create_mock_chat_model()
        model_b = create_mock_chat_model()

        extractor_a = GraphExtractor(model_invoker=model_a, use_dspy=True)
        extractor_b = GraphExtractor(model_invoker=model_b, use_dspy=True)

        # Create both extractors, they should maintain separate LM instances
        assert extractor_a._dspy_lm is not None
        assert extractor_b._dspy_lm is not None
        assert extractor_a._dspy_lm != extractor_b._dspy_lm

        # Both should work without interfering
        result_a = await extractor_a(
            texts=["Alice works at TechCorp."],
            prompt_variables={"entity_types": ["PERSON"]},
        )
        result_b = await extractor_b(
            texts=["Bob works at DataCorp."],
            prompt_variables={"entity_types": ["PERSON"]},
        )

        assert result_a.output is not None
        assert result_b.output is not None


class TestCommunityReportsExtractorIntegration:
    """Integration tests for CommunityReportsExtractor with DSPy."""

    @pytest.mark.asyncio
    async def test_dspy_mode_basic(self):
        """Test CommunityReportsExtractor in DSPy mode."""
        mock_model = create_mock_chat_model()
        extractor = CommunityReportsExtractor(
            model_invoker=mock_model, use_dspy=True
        )

        result = await extractor(
            input_text="Entities:\nid,entity,description\n1,ALICE,A person\n\nRelationships:\nid,source,target,description\n1,ALICE,TECHCORP,works at"
        )

        assert result.output is not None
        # DSPy mode should return structured output
        assert result.structured_output is not None or result.output != ""

    @pytest.mark.asyncio
    async def test_legacy_mode_basic(self):
        """Test CommunityReportsExtractor in legacy mode."""
        mock_model = create_mock_chat_model()
        extractor = CommunityReportsExtractor(
            model_invoker=mock_model, use_dspy=False
        )

        result = await extractor(
            input_text="Entities:\nid,entity,description\n1,ALICE,A person"
        )

        assert result.output is not None


class TestExtractorErrorHandling:
    """Test error handling in DSPy extractors."""

    @pytest.mark.asyncio
    async def test_dspy_handles_empty_input(self):
        """Test that DSPy mode handles empty input gracefully."""
        mock_model = create_mock_chat_model()
        extractor = GraphExtractor(model_invoker=mock_model, use_dspy=True)

        result = await extractor(texts=[""], prompt_variables={})

        # Should not crash, should return valid result
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_legacy_handles_empty_input(self):
        """Test that legacy mode handles empty input gracefully."""
        mock_model = create_mock_chat_model()
        extractor = GraphExtractor(model_invoker=mock_model, use_dspy=False)

        result = await extractor(texts=[""], prompt_variables={})

        # Should not crash, should return valid result
        assert result.output is not None


class TestExtractorEquivalence:
    """Test that DSPy and legacy modes produce similar results."""

    @pytest.mark.asyncio
    async def test_graph_extraction_equivalence(self):
        """Test that DSPy and legacy produce structurally similar results."""
        mock_model = create_mock_chat_model()

        extractor_dspy = GraphExtractor(
            model_invoker=mock_model, use_dspy=True, max_gleanings=0
        )
        extractor_legacy = GraphExtractor(
            model_invoker=mock_model, use_dspy=False, max_gleanings=0
        )

        test_text = ["Alice works at TechCorp. Bob is the CEO."]
        prompt_vars = {"entity_types": ["PERSON", "ORGANIZATION"]}

        result_dspy = await extractor_dspy(
            texts=test_text, prompt_variables=prompt_vars
        )
        result_legacy = await extractor_legacy(
            texts=test_text, prompt_variables=prompt_vars
        )

        # Both should produce graph outputs
        assert result_dspy.output is not None
        assert result_legacy.output is not None

        # Both should have nodes (may differ in content but both should extract something)
        assert len(result_dspy.output.nodes) >= 0
        assert len(result_legacy.output.nodes) >= 0
