#!/usr/bin/env python
"""Standalone test script for DSPy modules that avoids importing broken test infrastructure."""

import sys
import dspy
from graphrag.dspy_modules.index import (
    GraphExtractionModule,
    CommunityReportModule,
    ClaimExtractionModule,
    DescriptionSummaryModule,
)


class DummyLM(dspy.LM):
    """Mock LM for testing."""

    def __init__(self, responses=None):
        super().__init__(model="test-model")
        self.responses = responses or {}
        self.call_count = 0

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.call_count += 1

        # Pattern matching for different module types
        prompt_text = str(prompt).lower() if prompt else str(messages).lower() if messages else ""

        # DSPy expects JSON responses matching signature fields
        import json

        # More specific pattern matching using signature field names
        if "refined_entities" in prompt_text or "refined_relationships" in prompt_text:
            # ExtractionRefinementSignature (ChainOfThought)
            response = {
                "reasoning": "Checking extraction quality",
                "refinement_rationale": "Extractions look good",
                "needs_refinement": False,
                "refined_entities": 'ALICE<|>PERSON<|>A person',
                "refined_relationships": 'ALICE<|>TECHCORP<|>works at<|>1.0'
            }
        elif "relationships" in prompt_text and "entity" not in prompt_text:
            # RelationshipIdentificationSignature: reasoning (from ChainOfThought), rationale, relationships
            response = {
                "reasoning": "Let me identify relationships between entities",
                "rationale": "Identified relationships between entities",
                "relationships": 'ALICE<|>TECHCORP<|>works at<|>1.0'
            }
        elif "entities" in prompt_text and "relationship" not in prompt_text:
            # EntityIdentificationSignature: reasoning (from ChainOfThought), rationale, entities
            response = {
                "reasoning": "Let me identify entities in the text",
                "rationale": "Identified entities from the text",
                "entities": 'ALICE<|>PERSON<|>A person##TECHCORP<|>ORGANIZATION<|>A company'
            }
        elif "refinement" in prompt_text or "needs_refinement" in prompt_text:
            # ExtractionRefinementSignature (ChainOfThought)
            response = {
                "reasoning": "Checking extraction quality",
                "refinement_rationale": "Extractions look good",
                "needs_refinement": False,
                "refined_entities": 'ALICE<|>PERSON<|>A person',
                "refined_relationships": 'ALICE<|>TECHCORP<|>works at<|>1.0'
            }
        elif "format" in prompt_text or "tuple_delimiter" in prompt_text:
            # ExtractionFormattingSignature (Predict - no reasoning needed)
            response = {
                "formatted_output": '("entity"<|>ALICE<|>PERSON<|>A person)##("entity"<|>TECHCORP<|>ORG<|>A company)##<|COMPLETE|>'
            }
        elif "structure" in prompt_text:
            # CommunityStructureAnalysisSignature (ChainOfThought)
            response = {
                "reasoning": "Analyzing community structure",
                "structure_analysis": "ALICE is connected to TECHCORP through employment",
                "key_entities": "ALICE, TECHCORP"
            }
        elif "impact" in prompt_text or "severity" in prompt_text:
            # CommunityImpactAnalysisSignature (ChainOfThought)
            response = {
                "reasoning": "Assessing community impact",
                "impact_analysis": "High impact community with strong organizational ties",
                "severity_rating": 8.5,
                "rating_rationale": "Important employment relationship"
            }
        elif "findings" in prompt_text:
            # CommunityFindingsSignature (ChainOfThought)
            response = {
                "reasoning": "Extracting key findings",
                "findings_list": "1. Alice works at TechCorp\\n2. Strong organizational connection"
            }
        elif "report" in prompt_text or "synthesize" in prompt_text:
            # ReportSynthesisSignature (ChainOfThought) - return the full report structure
            response = {
                "reasoning": "Synthesizing all analyses into final report",
                "report": json.dumps({
                    "title": "Test Community Report",
                    "summary": "A test community with employment connections",
                    "findings": [
                        {"summary": "Employment Relationship", "explanation": "Alice works at TechCorp"}
                    ],
                    "rating": 7.5,
                    "rating_explanation": "Strong organizational ties"
                })
            }
        elif "claim" in prompt_text and "validation" in prompt_text:
            # ClaimValidationSignature (ChainOfThought)
            response = {
                "reasoning": "Validating extracted claims",
                "validation_thought": "Claims appear valid",
                "validated_claims": "Alice is an employee of TechCorp",
                "needs_more_extraction": False
            }
        elif "claim" in prompt_text:
            # ClaimIdentificationSignature (ChainOfThought)
            response = {
                "reasoning": "Identifying claims in text",
                "thought": "Analyzing claims in the text",
                "initial_claims": "Alice claims to be CEO of TechCorp"
            }
        elif "extractive" in prompt_text or "key_points" in prompt_text:
            # ExtractiveSummarySignature (ChainOfThought)
            response = {
                "reasoning": "Extracting key points",
                "key_points": "Alice, TechCorp, employment"
            }
        elif "abstractive" in prompt_text or "abstract_summary" in prompt_text:
            # AbstractiveSummarySignature (ChainOfThought)
            response = {
                "reasoning": "Creating abstract summary",
                "abstract_summary": "Alice works at TechCorp"
            }
        elif "fusion" in prompt_text or "fused_summary" in prompt_text:
            # SummaryFusionSignature (ChainOfThought)
            response = {
                "reasoning": "Fusing extractive and abstractive summaries",
                "fused_summary": "Alice is a person who works at TechCorp"
            }
        else:
            response = {"response": "Test response"}

        return [json.dumps(response)]


def test_graph_extraction():
    """Test GraphExtractionModule."""
    print("Testing GraphExtractionModule...")
    lm = DummyLM()

    with dspy.context(lm=lm):
        module = GraphExtractionModule(max_gleanings=1)
        result = module.forward(
            entity_types="PERSON,ORG",
            input_text="Alice works at TechCorp.",
            tuple_delimiter="<|>",
            record_delimiter="##",
            completion_delimiter="<|COMPLETE|>",
        )

    assert hasattr(result, 'extracted_data'), "Result should have extracted_data"
    assert "ALICE" in result.extracted_data or "Alice" in result.extracted_data, f"Expected ALICE in result, got: {result.extracted_data}"
    print("✓ GraphExtractionModule test passed")


def test_community_report():
    """Test CommunityReportModule."""
    print("Testing CommunityReportModule...")
    lm = DummyLM()

    with dspy.context(lm=lm):
        module = CommunityReportModule()
        result = module.forward(
            input_text="Alice works at TechCorp. TechCorp is a technology company.",
            max_report_length=1500,
        )

    assert hasattr(result, 'report'), "Result should have report field"
    assert result.report is not None, "Report should not be None"
    print("✓ CommunityReportModule test passed")


def test_claim_extraction():
    """Test ClaimExtractionModule."""
    print("Testing ClaimExtractionModule...")
    lm = DummyLM()

    with dspy.context(lm=lm):
        module = ClaimExtractionModule(max_iterations=2)
        result = module.forward(
            input_text="Alice claims to be CEO of TechCorp.",
            entity_specs="PERSON: Alice",
        )

    assert hasattr(result, 'extracted_claims'), "Result should have extracted_claims"
    assert result.extracted_claims is not None, "Claims should not be None"
    print("✓ ClaimExtractionModule test passed")


def test_description_summary():
    """Test DescriptionSummaryModule."""
    print("Testing DescriptionSummaryModule...")
    lm = DummyLM()

    with dspy.context(lm=lm):
        module = DescriptionSummaryModule()
        result = module.forward(
            descriptions="Alice is a person.\\nAlice works at TechCorp.\\nTechCorp is a company.",
        )

    assert hasattr(result, 'summary'), "Result should have summary field"
    assert result.summary is not None, "Summary should not be None"
    print("✓ DescriptionSummaryModule test passed")


def test_no_global_state_pollution():
    """Test that multiple modules don't interfere via global state."""
    print("Testing no global state pollution...")
    lm1 = DummyLM()
    lm2 = DummyLM()

    # Create two modules with different LMs
    module1 = GraphExtractionModule()
    module2 = GraphExtractionModule()

    # Run with separate contexts - they should not interfere
    with dspy.context(lm=lm1):
        result1 = module1.forward(
            entity_types="PERSON",
            input_text="Test 1",
        )

    with dspy.context(lm=lm2):
        result2 = module2.forward(
            entity_types="ORG",
            input_text="Test 2",
        )

    # Both should succeed without interference
    assert result1 is not None, "First result should not be None"
    assert result2 is not None, "Second result should not be None"
    print("✓ No global state pollution test passed")


def main():
    """Run all tests."""
    tests = [
        test_graph_extraction,
        test_community_report,
        test_claim_extraction,
        test_description_summary,
        test_no_global_state_pollution,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
