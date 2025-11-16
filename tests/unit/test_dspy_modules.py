# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for DSPy modules."""

import dspy
import pytest

from graphrag.dspy_modules.index import (
    ClaimExtractionModule,
    CommunityReportModule,
    DescriptionSummaryModule,
    GraphExtractionModule,
)
from graphrag.dspy_modules.query import (
    DriftSearchModule,
    GlobalSearchMapModule,
    GlobalSearchReduceModule,
    LocalSearchModule,
    QuestionGenModule,
)


class DummyLM(dspy.LM):
    """Mock LM for testing that returns predictable outputs."""

    def __init__(self, responses: dict[str, str] | None = None):
        """Initialize with canned responses."""
        super().__init__(model="test-model")
        self.responses = responses or {}
        self.call_count = 0

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Return canned response based on prompt content."""
        self.call_count += 1

        # Simple pattern matching for test responses
        if prompt and "entity" in prompt.lower():
            content = 'ALICE<|>PERSON<|>A person\n##\nTECHCORP<|>ORG<|>A company'
        elif prompt and "relationship" in prompt.lower():
            content = "ALICE<|>TECHCORP<|>works at<|>8"
        elif prompt and "community" in prompt.lower():
            content = '{"title": "Test Community", "summary": "Test", "rating": 5.0, "rating_explanation": "Test", "findings": []}'
        elif prompt and "claim" in prompt.lower():
            content = "Claim 1: Test claim"
        elif prompt and "question" in prompt.lower():
            content = "What is this?\nWho is that?"
        else:
            content = "Test response"

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }


@pytest.fixture
def mock_lm():
    """Provide a mock LM for testing."""
    return DummyLM()


@pytest.fixture
def dspy_context(mock_lm):
    """Configure DSPy with mock LM."""
    with dspy.context(lm=mock_lm):
        yield mock_lm


class TestGraphExtractionModule:
    """Tests for GraphExtractionModule."""

    def test_basic_extraction(self, dspy_context):
        """Test basic entity and relationship extraction."""
        module = GraphExtractionModule(max_gleanings=0)

        result = module.forward(
            entity_types="PERSON,ORG",
            input_text="Alice works at TechCorp.",
            tuple_delimiter="<|>",
            record_delimiter="##",
            completion_delimiter="<|COMPLETE|>",
        )

        assert hasattr(result, "extracted_data")
        assert result.extracted_data is not None
        assert dspy_context.call_count > 0

    def test_with_gleanings(self, dspy_context):
        """Test extraction with iterative refinement."""
        module = GraphExtractionModule(max_gleanings=1)

        result = module.forward(
            entity_types="PERSON,ORG",
            input_text="Alice works at TechCorp.",
        )

        # Should make multiple calls due to refinement
        assert dspy_context.call_count >= 3  # Entity + Relationship + Refinement


class TestCommunityReportModule:
    """Tests for CommunityReportModule."""

    def test_report_generation(self, dspy_context):
        """Test community report generation."""
        module = CommunityReportModule()

        result = module.forward(
            input_text="Entity: ALICE, TECHCORP\nRelationship: ALICE works at TECHCORP",
            max_report_length=1500,
        )

        assert hasattr(result, "report")
        # Should call 3 analysts + synthesizer = 4 calls
        assert dspy_context.call_count >= 3


class TestClaimExtractionModule:
    """Tests for ClaimExtractionModule."""

    def test_claim_extraction(self, dspy_context):
        """Test claim extraction."""
        module = ClaimExtractionModule(max_iterations=1)

        result = module.forward(
            input_text="Alice is the CEO of TechCorp.",
            entity_specs="PERSON,ORG",
        )

        assert hasattr(result, "extracted_claims")
        # Identifier + Validator
        assert dspy_context.call_count >= 2


class TestDescriptionSummaryModule:
    """Tests for DescriptionSummaryModule."""

    def test_summary_generation(self, dspy_context):
        """Test dual-path summarization."""
        module = DescriptionSummaryModule()

        result = module.forward(
            descriptions="Alice is a CEO.\nAlice leads TechCorp.\nAlice founded the company."
        )

        assert hasattr(result, "summary")
        # Extractive + Abstractive + Fusion = 3 calls
        assert dspy_context.call_count >= 3


class TestLocalSearchModule:
    """Tests for LocalSearchModule."""

    def test_local_search(self, dspy_context):
        """Test ensemble local search."""
        module = LocalSearchModule()

        result = module.forward(
            context_data="Entities: ALICE, TECHCORP\nRelationships: ...",
            response_type="Multiple paragraphs",
            question="Who is Alice?",
        )

        assert hasattr(result, "answer")
        # Context + Direct + Reasoned + Synthesizer = 4 calls
        assert dspy_context.call_count >= 4


class TestGlobalSearchModules:
    """Tests for Global Search Map/Reduce."""

    def test_map_with_relevance_filtering(self, dspy_context):
        """Test map operation with filtering."""
        module = GlobalSearchMapModule(relevance_threshold=0.5)

        result = module.forward(
            community_report="Community about AI companies",
            question="Tell me about AI companies",
            response_type="Paragraphs",
        )

        assert hasattr(result, "analysis")
        # Relevance checker + analyzer = 2 calls
        assert dspy_context.call_count >= 1

    def test_reduce_with_refinement(self, dspy_context):
        """Test reduce operation with self-refinement."""
        module = GlobalSearchReduceModule()

        result = module.forward(
            analyses="Analysis 1: ...\nAnalysis 2: ...",
            question="Summarize the analyses",
            response_type="Paragraphs",
        )

        assert hasattr(result, "answer")
        # Initial synthesizer + refiner = 2 calls
        assert dspy_context.call_count >= 2


class TestDriftSearchModule:
    """Tests for DriftSearchModule."""

    def test_drift_detection(self, dspy_context):
        """Test conversation drift handling."""
        module = DriftSearchModule()

        result = module.forward(
            conversation_history="User: Tell me about AI\nAssistant: AI is...",
            current_question="What about blockchain?",
            context_data="Context data...",
            response_type="Paragraphs",
        )

        assert hasattr(result, "answer")
        # Memory + Drift detector + Answerer = 3 calls
        assert dspy_context.call_count >= 3


class TestQuestionGenModule:
    """Tests for QuestionGenModule."""

    def test_question_generation(self, dspy_context):
        """Test creative question generation with validation."""
        module = QuestionGenModule()

        result = module.forward(context="Alice is the CEO of TechCorp.", num_questions=5)

        assert hasattr(result, "questions")
        # Creative generator + validator = 2 calls
        assert dspy_context.call_count >= 2


class TestModuleComposition:
    """Tests for composing DSPy modules."""

    def test_pipeline_composition(self, dspy_context):
        """Test that modules can be composed together."""

        class SimplePipeline(dspy.Module):
            def __init__(self):
                super().__init__()
                self.extractor = GraphExtractionModule(max_gleanings=0)
                self.summarizer = DescriptionSummaryModule()

            def forward(self, text):
                # Extract then summarize
                extraction = self.extractor(
                    entity_types="PERSON,ORG", input_text=text
                )
                summary = self.summarizer(descriptions=extraction.extracted_data)
                return dspy.Prediction(
                    extraction=extraction.extracted_data, summary=summary.summary
                )

        pipeline = SimplePipeline()
        result = pipeline.forward("Alice works at TechCorp.")

        assert hasattr(result, "extraction")
        assert hasattr(result, "summary")
