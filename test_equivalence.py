#!/usr/bin/env python
"""Test output equivalence between DSPy and legacy modes."""

import asyncio
from tests.mock_provider import create_mock_chat_model


async def test_graph_extractor_equivalence():
    """Compare GraphExtractor outputs in both modes."""
    from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor

    test_text = "Alice is a software engineer at TechCorp. TechCorp is a technology company based in Seattle. Bob also works at TechCorp as a data scientist."

    test_input = {
        "input_text": test_text,
    }

    test_prompt_vars = {
        "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "ROLE"]
    }

    # Legacy mode
    mock_llm_legacy = create_mock_chat_model([
        '("entity"<|>ALICE<|>PERSON<|>A software engineer)##("entity"<|>TECHCORP<|>ORGANIZATION<|>A technology company)##("entity"<|>SEATTLE<|>LOCATION<|>City)##("entity"<|>BOB<|>PERSON<|>A data scientist)##("relationship"<|>ALICE<|>TECHCORP<|>works at<|>9<|>Alice is a software engineer at TechCorp)##("relationship"<|>TECHCORP<|>SEATTLE<|>based in<|>8<|>TechCorp is based in Seattle)##("relationship"<|>BOB<|>TECHCORP<|>works at<|>9<|>Bob works at TechCorp)##<|COMPLETE|>',
        '("entity"<|>SOFTWARE_ENGINEERING<|>ROLE<|>Job function)##<|COMPLETE|>',
        'Y'
    ])

    extractor_legacy = GraphExtractor(
        model_invoker=mock_llm_legacy,
        max_gleanings=1,
        use_dspy=False
    )

    result_legacy = await extractor_legacy(test_input, test_prompt_vars)

    # DSPy mode
    mock_llm_dspy = create_mock_chat_model([
        '("entity"<|>ALICE<|>PERSON<|>A software engineer)##("entity"<|>TECHCORP<|>ORGANIZATION<|>A technology company)##("entity"<|>SEATTLE<|>LOCATION<|>City)##("entity"<|>BOB<|>PERSON<|>A data scientist)##("relationship"<|>ALICE<|>TECHCORP<|>works at<|>9<|>Alice is a software engineer at TechCorp)##("relationship"<|>TECHCORP<|>SEATTLE<|>based in<|>8<|>TechCorp is based in Seattle)##("relationship"<|>BOB<|>TECHCORP<|>works at<|>9<|>Bob works at TechCorp)##<|COMPLETE|>',
        '("entity"<|>SOFTWARE_ENGINEERING<|>ROLE<|>Job function)##<|COMPLETE|>',
        'Y'
    ])

    extractor_dspy = GraphExtractor(
        model_invoker=mock_llm_dspy,
        max_gleanings=1,
        use_dspy=True
    )

    result_dspy = await extractor_dspy(test_input, test_prompt_vars)

    # Compare outputs
    print("=" * 80)
    print("GraphExtractor Equivalence Test")
    print("=" * 80)
    print(f"\nLegacy output length: {len(result_legacy.output)}")
    print(f"DSPy output length:   {len(result_dspy.output)}")
    print(f"\nLegacy output:\n{result_legacy.output[:500]}...")
    print(f"\nDSPy output:\n{result_dspy.output[:500]}...")

    # Check if outputs are similar
    if result_legacy.output == result_dspy.output:
        print("\n✅ EXACT MATCH - Outputs are identical")
        return True
    else:
        print("\n⚠️  MISMATCH - Outputs differ")
        print(f"Similarity: {len(set(result_legacy.output) & set(result_dspy.output)) / max(len(result_legacy.output), len(result_dspy.output)) * 100:.1f}%")
        return False


async def test_community_reports_equivalence():
    """Compare CommunityReportsExtractor outputs in both modes."""
    from graphrag.index.operations.summarize_communities.community_reports_extractor import CommunityReportsExtractor

    test_input = "Alice works at TechCorp. TechCorp is based in Seattle. Bob also works at TechCorp."

    # Legacy mode
    mock_llm_legacy = create_mock_chat_model([
        '{"title": "TechCorp Employment Network", "summary": "A professional network centered around TechCorp with employees Alice and Bob", "rating": 8.0, "rating_explanation": "Strong organizational ties with multiple employees", "findings": [{"summary": "Employment Hub", "explanation": "TechCorp serves as central employer for Alice and Bob"}, {"summary": "Geographic Base", "explanation": "Company is based in Seattle"}]}'
    ])

    extractor_legacy = CommunityReportsExtractor(
        model_invoker=mock_llm_legacy,
        max_report_length=1500,
        use_dspy=False
    )

    result_legacy = await extractor_legacy(input_text=test_input)

    # DSPy mode
    mock_llm_dspy = create_mock_chat_model([
        '{"title": "TechCorp Employment Network", "summary": "A professional network centered around TechCorp with employees Alice and Bob", "rating": 8.0, "rating_explanation": "Strong organizational ties with multiple employees", "findings": [{"summary": "Employment Hub", "explanation": "TechCorp serves as central employer for Alice and Bob"}, {"summary": "Geographic Base", "explanation": "Company is based in Seattle"}]}'
    ])

    extractor_dspy = CommunityReportsExtractor(
        model_invoker=mock_llm_dspy,
        max_report_length=1500,
        use_dspy=True
    )

    result_dspy = await extractor_dspy(input_text=test_input)

    # Compare outputs
    print("\n" + "=" * 80)
    print("CommunityReportsExtractor Equivalence Test")
    print("=" * 80)
    print(f"\nLegacy output length: {len(result_legacy.output)}")
    print(f"DSPy output length:   {len(result_dspy.output)}")
    print(f"\nLegacy output:\n{result_legacy.output[:500]}...")
    print(f"\nDSPy output:\n{result_dspy.output[:500]}...")

    if result_legacy.output == result_dspy.output:
        print("\n✅ EXACT MATCH - Outputs are identical")
        return True
    else:
        print("\n⚠️  MISMATCH - Outputs differ")
        return False


async def main():
    """Run all equivalence tests."""
    print("Testing output equivalence between DSPy and legacy modes...")
    print("This will help identify if DSPy produces equivalent quality outputs.\n")

    results = []

    # Test each extractor
    results.append(("GraphExtractor", await test_graph_extractor_equivalence()))
    results.append(("CommunityReports", await test_community_reports_equivalence()))

    # Summary
    print("\n" + "=" * 80)
    print("EQUIVALENCE TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASS" if passed else "⚠️  FAIL"
        print(f"{name:<30} {status}")

    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests showing exact equivalence")

    if passed < total:
        print("\n⚠️  WARNING: Not all outputs are equivalent!")
        print("This suggests DSPy may need prompt tuning to match legacy quality.")
        print("Consider:")
        print("1. Adjusting DSPy signatures to better match legacy prompts")
        print("2. Adding few-shot examples to DSPy modules")
        print("3. Using DSPy optimizers to tune prompts")


if __name__ == "__main__":
    asyncio.run(main())
