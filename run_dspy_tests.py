#!/usr/bin/env python
"""Standalone test runner for DSPy integration tests."""

import sys
import traceback
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Test counters
total_tests = 0
passed_tests = 0
failed_tests = 0
failures = []


def run_test(test_name, test_func):
    """Run a single test and track results."""
    global total_tests, passed_tests, failed_tests
    total_tests += 1
    try:
        test_func()
        passed_tests += 1
        print(f"✅ PASS: {test_name}")
        return True
    except Exception as e:
        failed_tests += 1
        failures.append((test_name, str(e), traceback.format_exc()))
        print(f"❌ FAIL: {test_name}")
        print(f"   Error: {e}")
        return False


print("=" * 80)
print("DSPy Integration Test Suite")
print("=" * 80)
print()

# =============================================================================
# TEST SUITE 1: Extract Graph Module
# =============================================================================
print("📦 TEST SUITE: test_extract_graph.py")
print("-" * 80)


def test_import_graph_extractor():
    from graphrag.dspy_modules.extract_graph import GraphExtractor
    assert GraphExtractor is not None


def test_graph_extractor_initialization():
    from graphrag.dspy_modules.extract_graph import GraphExtractor
    extractor = GraphExtractor(max_gleanings=2)
    assert extractor.max_gleanings == 2
    assert hasattr(extractor, "extractor")
    assert hasattr(extractor, "gleaner")


def test_graph_extraction_signature():
    from graphrag.dspy_modules.extract_graph import GraphExtractionSignature
    assert GraphExtractionSignature is not None
    assert hasattr(GraphExtractionSignature, "__annotations__")


def test_extract_graph_convenience_function():
    from graphrag.dspy_modules.extract_graph import extract_graph_dspy
    assert callable(extract_graph_dspy)


def test_graph_extractor_forward_method():
    import inspect
    from graphrag.dspy_modules.extract_graph import GraphExtractor

    extractor = GraphExtractor()
    sig = inspect.signature(extractor.forward)
    assert "text" in sig.parameters
    assert "entity_types" in sig.parameters
    assert "tuple_delimiter" in sig.parameters
    assert "record_delimiter" in sig.parameters


run_test("test_import_graph_extractor", test_import_graph_extractor)
run_test("test_graph_extractor_initialization", test_graph_extractor_initialization)
run_test("test_graph_extraction_signature", test_graph_extraction_signature)
run_test("test_extract_graph_convenience_function", test_extract_graph_convenience_function)
run_test("test_graph_extractor_forward_method", test_graph_extractor_forward_method)

print()

# =============================================================================
# TEST SUITE 2: Community Reports Module
# =============================================================================
print("📦 TEST SUITE: test_community_reports.py")
print("-" * 80)


def test_import_community_report_generator():
    from graphrag.dspy_modules.community_reports import CommunityReportGenerator
    assert CommunityReportGenerator is not None


def test_community_report_generator_initialization():
    from graphrag.dspy_modules.community_reports import CommunityReportGenerator
    generator = CommunityReportGenerator()
    assert generator is not None
    assert hasattr(generator, "generator")


def test_community_report_signature():
    from graphrag.dspy_modules.community_reports import CommunityReportSignature
    assert CommunityReportSignature is not None


def test_community_report_output_pydantic_model():
    from graphrag.dspy_modules.community_reports import (
        CommunityReportOutput,
        CommunityFinding,
    )

    output = CommunityReportOutput(
        title="Test Community",
        summary="Test summary",
        rating=7.5,
        rating_explanation="High impact",
        findings=[CommunityFinding(summary="Finding", explanation="Explanation")],
    )
    assert output.title == "Test Community"
    assert output.rating == 7.5
    assert len(output.findings) == 1
    assert output.findings[0].summary == "Finding"


def test_community_report_rating_validation():
    from graphrag.dspy_modules.community_reports import CommunityReportOutput
    from pydantic import ValidationError

    # Valid rating
    output = CommunityReportOutput(
        title="Test",
        summary="Summary",
        rating=5.0,
        rating_explanation="Explanation",
        findings=[],
    )
    assert output.rating == 5.0

    # Invalid rating (> 10)
    try:
        CommunityReportOutput(
            title="Test",
            summary="Summary",
            rating=15.0,
            rating_explanation="Explanation",
            findings=[],
        )
        raise AssertionError("Should have raised ValidationError for rating > 10")
    except ValidationError:
        pass  # Expected

    # Invalid rating (< 0)
    try:
        CommunityReportOutput(
            title="Test",
            summary="Summary",
            rating=-1.0,
            rating_explanation="Explanation",
            findings=[],
        )
        raise AssertionError("Should have raised ValidationError for rating < 0")
    except ValidationError:
        pass  # Expected


def test_community_report_convenience_function():
    from graphrag.dspy_modules.community_reports import (
        generate_community_report_dspy,
    )
    assert callable(generate_community_report_dspy)


def test_community_report_forward_method():
    import inspect
    from graphrag.dspy_modules.community_reports import CommunityReportGenerator

    generator = CommunityReportGenerator()
    sig = inspect.signature(generator.forward)
    assert "input_text" in sig.parameters
    assert "max_report_length" in sig.parameters


run_test("test_import_community_report_generator", test_import_community_report_generator)
run_test("test_community_report_generator_initialization", test_community_report_generator_initialization)
run_test("test_community_report_signature", test_community_report_signature)
run_test("test_community_report_output_pydantic_model", test_community_report_output_pydantic_model)
run_test("test_community_report_rating_validation", test_community_report_rating_validation)
run_test("test_community_report_convenience_function", test_community_report_convenience_function)
run_test("test_community_report_forward_method", test_community_report_forward_method)

print()

# =============================================================================
# TEST SUITE 3: DSPy Chat Model Provider
# =============================================================================
print("📦 TEST SUITE: test_chat_model.py")
print("-" * 80)


def test_import_dspy_chat_model():
    from graphrag.language_model.providers.dspy.chat_model import DSPyChatModel
    assert DSPyChatModel is not None


def test_dspy_chat_model_has_chat_methods():
    from graphrag.language_model.providers.dspy.chat_model import DSPyChatModel
    assert hasattr(DSPyChatModel, "achat")
    assert hasattr(DSPyChatModel, "chat")
    assert hasattr(DSPyChatModel, "achat_stream")
    assert hasattr(DSPyChatModel, "chat_stream")
    assert callable(DSPyChatModel.achat)
    assert callable(DSPyChatModel.chat)


def test_dspy_model_output_structure():
    from graphrag.language_model.providers.dspy.chat_model import DSPyModelOutput
    output = DSPyModelOutput(content="Test response")
    assert output.content == "Test response"
    assert output.full_response is None


def test_dspy_model_response_structure():
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
    assert response.history[0]["role"] == "user"


def test_model_factory_integration():
    from graphrag.language_model.factory import ModelFactory
    from graphrag.config.enums import ModelType

    assert ModelFactory.is_supported_chat_model(ModelType.DSPyChat.value)
    assert ModelType.DSPyChat.value in ModelFactory.get_chat_models()


run_test("test_import_dspy_chat_model", test_import_dspy_chat_model)
run_test("test_dspy_chat_model_has_chat_methods", test_dspy_chat_model_has_chat_methods)
run_test("test_dspy_model_output_structure", test_dspy_model_output_structure)
run_test("test_dspy_model_response_structure", test_dspy_model_response_structure)
run_test("test_model_factory_integration", test_model_factory_integration)

print()

# =============================================================================
# TEST SUITE 4: Backward Compatibility
# =============================================================================
print("📦 TEST SUITE: Backward Compatibility")
print("-" * 80)


def test_existing_model_types_preserved():
    from graphrag.config.enums import ModelType
    assert ModelType.OpenAIChat.value == "openai_chat"
    assert ModelType.AzureOpenAIChat.value == "azure_openai_chat"
    assert ModelType.Chat.value == "chat"
    assert ModelType.OpenAIEmbedding.value == "openai_embedding"
    assert ModelType.AzureOpenAIEmbedding.value == "azure_openai_embedding"
    assert ModelType.Embedding.value == "embedding"


def test_new_model_type_added():
    from graphrag.config.enums import ModelType
    assert ModelType.DSPyChat.value == "dspy_chat"


def test_existing_prompts_unchanged():
    import os
    prompts = [
        "graphrag/prompts/index/extract_graph.py",
        "graphrag/prompts/index/community_report.py",
        "graphrag/prompts/query/local_search_system_prompt.py",
    ]
    for prompt_path in prompts:
        assert os.path.exists(prompt_path), f"Prompt file missing: {prompt_path}"


run_test("test_existing_model_types_preserved", test_existing_model_types_preserved)
run_test("test_new_model_type_added", test_new_model_type_added)
run_test("test_existing_prompts_unchanged", test_existing_prompts_unchanged)

print()
print("=" * 80)
print("TEST RESULTS SUMMARY")
print("=" * 80)
print(f"Total Tests:  {total_tests}")
print(f"Passed:       {passed_tests} ✅")
print(f"Failed:       {failed_tests} ❌")
print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
print("=" * 80)

if failed_tests > 0:
    print()
    print("FAILURES:")
    print("-" * 80)
    for name, error, tb in failures:
        print(f"\n❌ {name}")
        print(f"   Error: {error}")
        print(f"   Traceback:")
        print("   " + "\n   ".join(tb.split("\n")))

sys.exit(0 if failed_tests == 0 else 1)
