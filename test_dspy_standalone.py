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

        # Check for specific output field names (most specific first)
        # DSPy prompts include "output_fields" or field names in the prompt

        # ExtractionFormattingSignature - look for formatted_output field
        if "formatted_output" in prompt_text:
            response = {
                "formatted_output": '("entity"<|>ALICE<|>PERSON<|>A person)##("entity"<|>TECHCORP<|>ORG<|>A company)##<|COMPLETE|>'
            }
        # ExtractionRefinementSignature - look for refinement-specific fields
        elif ("refined_entities" in prompt_text and "refined_relationships" in prompt_text) or "refinement_rationale" in prompt_text:
            response = {
                "reasoning": "Checking extraction quality",
                "refinement_rationale": "Extractions look good",
                "needs_refinement": False,
                "refined_entities": 'ALICE<|>PERSON<|>A person',
                "refined_relationships": 'ALICE<|>TECHCORP<|>works at<|>1.0'
            }
        # RelationshipIdentificationSignature - look for rationale + relationships
        elif "rationale" in prompt_text and "relationships" in prompt_text:
            response = {
                "reasoning": "Let me identify relationships between entities",
                "rationale": "Identified relationships between entities",
                "relationships": 'ALICE<|>TECHCORP<|>works at<|>1.0'
            }
        # EntityIdentificationSignature - look for rationale + entities
        elif "rationale" in prompt_text and "entities" in prompt_text:
            response = {
                "reasoning": "Let me identify entities in the text",
                "rationale": "Identified entities from the text",
                "entities": 'ALICE<|>PERSON<|>A person##TECHCORP<|>ORGANIZATION<|>A company'
            }
        # ReportSynthesisSignature - look for report field
        elif "report" in prompt_text and ("structure_analysis" in prompt_text or "max_length" in prompt_text):
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
        # CommunityStructureAnalysisSignature
        elif "structure_analysis" in prompt_text and "key_entities" in prompt_text:
            response = {
                "reasoning": "Analyzing community structure",
                "structure_analysis": "ALICE is connected to TECHCORP through employment",
                "key_entities": "ALICE, TECHCORP"
            }
        # CommunityImpactAnalysisSignature
        elif "impact_analysis" in prompt_text or ("severity_rating" in prompt_text and "rating_rationale" in prompt_text):
            response = {
                "reasoning": "Assessing community impact",
                "impact_analysis": "High impact community with strong organizational ties",
                "severity_rating": 8.5,
                "rating_rationale": "Important employment relationship"
            }
        # CommunityFindingsSignature
        elif "findings_list" in prompt_text:
            response = {
                "reasoning": "Extracting key findings",
                "findings_list": "1. Alice works at TechCorp\\n2. Strong organizational connection"
            }
        # ClaimValidationSignature
        elif "validated_claims" in prompt_text or "needs_more_extraction" in prompt_text:
            response = {
                "reasoning": "Validating extracted claims",
                "validation_thought": "Claims appear valid",
                "validated_claims": "Alice is an employee of TechCorp",
                "needs_more_extraction": False
            }
        # ClaimIdentificationSignature
        elif "initial_claims" in prompt_text or ("thought" in prompt_text and "claim" in prompt_text):
            response = {
                "reasoning": "Identifying claims in text",
                "thought": "Analyzing claims in the text",
                "initial_claims": "Alice claims to be CEO of TechCorp"
            }
        # SummaryFusionSignature - check BEFORE extractive/abstractive (it contains both as inputs)
        elif "fused_summary" in prompt_text or ("key_points" in prompt_text and "abstract_summary" in prompt_text):
            response = {
                "reasoning": "Fusing extractive and abstractive summaries",
                "fused_summary": "Alice is a person who works at TechCorp"
            }
        # ExtractiveSummarySignature
        elif "key_points" in prompt_text:
            response = {
                "reasoning": "Extracting key points",
                "key_points": "Alice, TechCorp, employment"
            }
        # AbstractiveSummarySignature
        elif "abstract_summary" in prompt_text:
            response = {
                "reasoning": "Creating abstract summary",
                "abstract_summary": "Alice works at TechCorp"
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
