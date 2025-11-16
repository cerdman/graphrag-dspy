#!/usr/bin/env python
"""Performance benchmark comparing DSPy vs legacy modes for GraphRAG extractors."""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import dspy

from tests.mock_provider import MockChatLLM
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportsExtractor,
)
from graphrag.index.operations.extract_covariates.claim_extractor import ClaimExtractor
from graphrag.index.operations.summarize_descriptions.description_summary_extractor import (
    SummarizeExtractor,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    extractor_name: str
    mode: str  # "dspy" or "legacy"
    llm_calls: int
    total_tokens: int
    execution_time_ms: float
    output_size: int
    success: bool
    error: str | None = None


class MockCountingLLM(MockChatLLM):
    """Mock LLM that counts calls and tokens."""

    def __init__(self, responses: list[str] | None = None):
        super().__init__(responses=responses or self._default_responses())
        self.call_count = 0
        self.total_tokens = 0

    def _default_responses(self) -> list[str]:
        """Default responses for different extractors."""
        return [
            # Graph extraction
            '("entity"<|>ALICE<|>PERSON<|>A software engineer)##("entity"<|>TECHCORP<|>ORGANIZATION<|>A technology company)##("relationship"<|>ALICE<|>TECHCORP<|>works at<|>9<|>Alice is employed by TechCorp)##<|COMPLETE|>',
            # Additional gleanings
            '("entity"<|>SEATTLE<|>LOCATION<|>City where TechCorp is located)##<|COMPLETE|>',
            "Y",  # Continue extraction
            # Community report
            '{"title": "Technology Employment Network", "summary": "A professional network centered around TechCorp", "rating": 7.5, "rating_explanation": "Significant employment relationships", "findings": [{"summary": "Employment Hub", "explanation": "TechCorp serves as central employer"}, {"summary": "Geographic Concentration", "explanation": "Activities concentrated in Seattle"}]}',
            # Claim extraction
            "ALICE<|>TECHCORP<|>employee<|>confirmed<|>2023-01-01<|>present<|>Alice works as software engineer at TechCorp<|>Alice is a software engineer at TechCorp##<|COMPLETE|>",
            "Y",  # Continue claims
            # Description summary
            "Alice is a software engineer who works at TechCorp, a technology company located in Seattle. She is a key member of the engineering team.",
        ]

    async def achat(self, prompt: str, history: list | None = None, **kwargs) -> Any:
        """Mock async chat completion."""
        self.call_count += 1
        # Estimate tokens (rough approximation: 4 chars per token)
        self.total_tokens += len(prompt) // 4

        # Call parent implementation
        response = await super().achat(prompt, history, **kwargs)

        # Count response tokens
        if response.output and response.output.content:
            self.total_tokens += len(str(response.output.content)) // 4

        return response


class DSPyMockLLM(dspy.LM):
    """Mock LM for DSPy that counts calls."""

    def __init__(self):
        super().__init__(model="mock-model")
        self.call_count = 0
        self.total_tokens = 0

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.call_count += 1

        # Estimate tokens
        prompt_text = str(prompt) if prompt else str(messages) if messages else ""
        self.total_tokens += len(prompt_text) // 4

        import json

        # Smart response based on expected fields (simplified from test suite)
        prompt_lower = prompt_text.lower()

        if "formatted_output" in prompt_lower:
            response = {
                "formatted_output": '("entity"<|>ALICE<|>PERSON<|>A person)##("entity"<|>TECHCORP<|>ORG<|>A company)##<|COMPLETE|>'
            }
        elif "refinement_rationale" in prompt_lower or "refined_entities" in prompt_lower:
            response = {
                "reasoning": "Checking quality",
                "refinement_rationale": "Good",
                "needs_refinement": False,
                "refined_entities": 'ALICE<|>PERSON<|>A person',
                "refined_relationships": 'ALICE<|>TECHCORP<|>works at<|>1.0'
            }
        elif "rationale" in prompt_lower and "relationships" in prompt_lower:
            response = {
                "reasoning": "Finding relationships",
                "rationale": "Found relationships",
                "relationships": 'ALICE<|>TECHCORP<|>works at<|>9'
            }
        elif "rationale" in prompt_lower and "entities" in prompt_lower:
            response = {
                "reasoning": "Finding entities",
                "rationale": "Found entities",
                "entities": 'ALICE<|>PERSON<|>A person##TECHCORP<|>ORG<|>A company'
            }
        elif "report" in prompt_lower and "structure_analysis" in prompt_lower:
            response = {
                "reasoning": "Synthesizing",
                "report": json.dumps({
                    "title": "Report",
                    "summary": "Summary",
                    "findings": [{"summary": "Finding", "explanation": "Details"}],
                    "rating": 7.5,
                    "rating_explanation": "Explanation"
                })
            }
        elif "structure_analysis" in prompt_lower:
            response = {
                "reasoning": "Analyzing",
                "structure_analysis": "Structure details",
                "key_entities": "ALICE, TECHCORP"
            }
        elif "impact_analysis" in prompt_lower:
            response = {
                "reasoning": "Assessing",
                "impact_analysis": "High impact",
                "severity_rating": 8.0,
                "rating_rationale": "Important"
            }
        elif "findings_list" in prompt_lower:
            response = {
                "reasoning": "Extracting",
                "findings_list": "1. Finding one\\n2. Finding two"
            }
        elif "validated_claims" in prompt_lower:
            response = {
                "reasoning": "Validating",
                "validation_thought": "Valid",
                "validated_claims": "Claim text",
                "needs_more_extraction": False
            }
        elif "initial_claims" in prompt_lower:
            response = {
                "reasoning": "Identifying",
                "thought": "Analyzing",
                "initial_claims": "Claim text"
            }
        elif "fused_summary" in prompt_lower:
            response = {
                "reasoning": "Fusing",
                "fused_summary": "Combined summary"
            }
        elif "key_points" in prompt_lower:
            response = {
                "reasoning": "Extracting",
                "key_points": "Points"
            }
        elif "abstract_summary" in prompt_lower:
            response = {
                "reasoning": "Abstracting",
                "abstract_summary": "Summary"
            }
        else:
            response = {"response": "Default"}

        response_text = json.dumps(response)
        self.total_tokens += len(response_text) // 4

        return [response_text]


async def benchmark_graph_extractor(use_dspy: bool) -> BenchmarkResult:
    """Benchmark GraphExtractor."""
    mode = "dspy" if use_dspy else "legacy"
    print(f"  Benchmarking GraphExtractor ({mode})...")

    if use_dspy:
        # Create DSPy mock that counts calls
        import dspy
        from graphrag.language_model.providers.dspy.adapter import GraphRAGDSpyLM

        mock_llm = MockCountingLLM()
        dspy_mock = DSPyMockLLM()

        try:
            start_time = time.time()

            extractor = GraphExtractor(
                model_invoker=mock_llm,
                max_gleanings=1,
                use_dspy=True,
            )

            # Override the dspy_lm with our counting mock
            extractor._dspy_lm = dspy_mock

            result = await extractor(
                {"input_text": "Alice works at TechCorp in Seattle.", "entity_types": ["PERSON", "ORGANIZATION", "LOCATION"]},
                {},
            )

            execution_time = (time.time() - start_time) * 1000

            return BenchmarkResult(
                extractor_name="GraphExtractor",
                mode=mode,
                llm_calls=dspy_mock.call_count,
                total_tokens=dspy_mock.total_tokens,
                execution_time_ms=execution_time,
                output_size=len(str(result.output)),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                extractor_name="GraphExtractor",
                mode=mode,
                llm_calls=0,
                total_tokens=0,
                execution_time_ms=0,
                output_size=0,
                success=False,
                error=str(e),
            )
    else:
        mock_llm = MockCountingLLM()

        try:
            start_time = time.time()

            extractor = GraphExtractor(
                model_invoker=mock_llm,
                max_gleanings=1,
                use_dspy=False,
            )

            result = await extractor(
                {"input_text": "Alice works at TechCorp in Seattle.", "entity_types": ["PERSON", "ORGANIZATION", "LOCATION"]},
                {},
            )

            execution_time = (time.time() - start_time) * 1000

            return BenchmarkResult(
                extractor_name="GraphExtractor",
                mode=mode,
                llm_calls=mock_llm.call_count,
                total_tokens=mock_llm.total_tokens,
                execution_time_ms=execution_time,
                output_size=len(str(result.output)),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                extractor_name="GraphExtractor",
                mode=mode,
                llm_calls=0,
                total_tokens=0,
                execution_time_ms=0,
                output_size=0,
                success=False,
                error=str(e),
            )


async def benchmark_community_reports(use_dspy: bool) -> BenchmarkResult:
    """Benchmark CommunityReportsExtractor."""
    mode = "dspy" if use_dspy else "legacy"
    print(f"  Benchmarking CommunityReportsExtractor ({mode})...")

    if use_dspy:
        import dspy

        mock_llm = MockCountingLLM()
        dspy_mock = DSPyMockLLM()

        try:
            start_time = time.time()

            extractor = CommunityReportsExtractor(
                model_invoker=mock_llm,
                max_report_length=1500,
                use_dspy=True,
            )
            extractor._dspy_lm = dspy_mock

            result = await extractor(
                input_text="Alice works at TechCorp. TechCorp is in Seattle.",
            )

            execution_time = (time.time() - start_time) * 1000

            return BenchmarkResult(
                extractor_name="CommunityReportsExtractor",
                mode=mode,
                llm_calls=dspy_mock.call_count,
                total_tokens=dspy_mock.total_tokens,
                execution_time_ms=execution_time,
                output_size=len(str(result.output)),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                extractor_name="CommunityReportsExtractor",
                mode=mode,
                llm_calls=0,
                total_tokens=0,
                execution_time_ms=0,
                output_size=0,
                success=False,
                error=str(e),
            )
    else:
        mock_llm = MockCountingLLM()

        try:
            start_time = time.time()

            extractor = CommunityReportsExtractor(
                model_invoker=mock_llm,
                max_report_length=1500,
                use_dspy=False,
            )

            result = await extractor(
                input_text="Alice works at TechCorp. TechCorp is in Seattle.",
            )

            execution_time = (time.time() - start_time) * 1000

            return BenchmarkResult(
                extractor_name="CommunityReportsExtractor",
                mode=mode,
                llm_calls=mock_llm.call_count,
                total_tokens=mock_llm.total_tokens,
                execution_time_ms=execution_time,
                output_size=len(str(result.output)),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                extractor_name="CommunityReportsExtractor",
                mode=mode,
                llm_calls=0,
                total_tokens=0,
                execution_time_ms=0,
                output_size=0,
                success=False,
                error=str(e),
            )


async def benchmark_claim_extractor(use_dspy: bool) -> BenchmarkResult:
    """Benchmark ClaimExtractor."""
    mode = "dspy" if use_dspy else "legacy"
    print(f"  Benchmarking ClaimExtractor ({mode})...")

    if use_dspy:
        import dspy

        mock_llm = MockCountingLLM()
        dspy_mock = DSPyMockLLM()

        try:
            start_time = time.time()

            extractor = ClaimExtractor(
                model_invoker=mock_llm,
                max_gleanings=2,
                use_dspy=True,
            )
            extractor._dspy_lm = dspy_mock

            result = await extractor(
                {
                    "input_text": ["Alice claims to be CEO of TechCorp."],
                    "entity_specs": "PERSON: Alice; ORGANIZATION: TechCorp",
                    "claim_description": "Employment claims",
                }
            )

            execution_time = (time.time() - start_time) * 1000

            return BenchmarkResult(
                extractor_name="ClaimExtractor",
                mode=mode,
                llm_calls=dspy_mock.call_count,
                total_tokens=dspy_mock.total_tokens,
                execution_time_ms=execution_time,
                output_size=len(str(result.output)),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                extractor_name="ClaimExtractor",
                mode=mode,
                llm_calls=0,
                total_tokens=0,
                execution_time_ms=0,
                output_size=0,
                success=False,
                error=str(e),
            )
    else:
        mock_llm = MockCountingLLM()

        try:
            start_time = time.time()

            extractor = ClaimExtractor(
                model_invoker=mock_llm,
                max_gleanings=2,
                use_dspy=False,
            )

            result = await extractor(
                {
                    "input_text": ["Alice claims to be CEO of TechCorp."],
                    "entity_specs": "PERSON: Alice; ORGANIZATION: TechCorp",
                    "claim_description": "Employment claims",
                }
            )

            execution_time = (time.time() - start_time) * 1000

            return BenchmarkResult(
                extractor_name="ClaimExtractor",
                mode=mode,
                llm_calls=mock_llm.call_count,
                total_tokens=mock_llm.total_tokens,
                execution_time_ms=execution_time,
                output_size=len(str(result.output)),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                extractor_name="ClaimExtractor",
                mode=mode,
                llm_calls=0,
                total_tokens=0,
                execution_time_ms=0,
                output_size=0,
                success=False,
                error=str(e),
            )


async def benchmark_description_summary(use_dspy: bool) -> BenchmarkResult:
    """Benchmark SummarizeExtractor."""
    mode = "dspy" if use_dspy else "legacy"
    print(f"  Benchmarking SummarizeExtractor ({mode})...")

    if use_dspy:
        import dspy

        mock_llm = MockCountingLLM()
        dspy_mock = DSPyMockLLM()

        try:
            start_time = time.time()

            extractor = SummarizeExtractor(
                model_invoker=mock_llm,
                max_summary_length=500,
                max_input_tokens=4000,
                use_dspy=True,
            )
            extractor._dspy_lm = dspy_mock

            result = await extractor(
                id="entity_1",
                descriptions=[
                    "Alice is a software engineer.",
                    "Alice works at TechCorp.",
                    "TechCorp is in Seattle.",
                ],
            )

            execution_time = (time.time() - start_time) * 1000

            return BenchmarkResult(
                extractor_name="SummarizeExtractor",
                mode=mode,
                llm_calls=dspy_mock.call_count,
                total_tokens=dspy_mock.total_tokens,
                execution_time_ms=execution_time,
                output_size=len(str(result.description)),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                extractor_name="SummarizeExtractor",
                mode=mode,
                llm_calls=0,
                total_tokens=0,
                execution_time_ms=0,
                output_size=0,
                success=False,
                error=str(e),
            )
    else:
        mock_llm = MockCountingLLM()

        try:
            start_time = time.time()

            extractor = SummarizeExtractor(
                model_invoker=mock_llm,
                max_summary_length=500,
                max_input_tokens=4000,
                use_dspy=False,
            )

            result = await extractor(
                id="entity_1",
                descriptions=[
                    "Alice is a software engineer.",
                    "Alice works at TechCorp.",
                    "TechCorp is in Seattle.",
                ],
            )

            execution_time = (time.time() - start_time) * 1000

            return BenchmarkResult(
                extractor_name="SummarizeExtractor",
                mode=mode,
                llm_calls=mock_llm.call_count,
                total_tokens=mock_llm.total_tokens,
                execution_time_ms=execution_time,
                output_size=len(str(result.description)),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                extractor_name="SummarizeExtractor",
                mode=mode,
                llm_calls=0,
                total_tokens=0,
                execution_time_ms=0,
                output_size=0,
                success=False,
                error=str(e),
            )


async def run_all_benchmarks():
    """Run all benchmarks and display results."""
    print("=" * 80)
    print("GraphRAG DSPy Performance Benchmark")
    print("=" * 80)
    print()

    results = []

    # Benchmark each extractor in both modes
    print("Running GraphExtractor benchmarks...")
    results.append(await benchmark_graph_extractor(use_dspy=False))
    results.append(await benchmark_graph_extractor(use_dspy=True))
    print()

    print("Running CommunityReportsExtractor benchmarks...")
    results.append(await benchmark_community_reports(use_dspy=False))
    results.append(await benchmark_community_reports(use_dspy=True))
    print()

    print("Running ClaimExtractor benchmarks...")
    results.append(await benchmark_claim_extractor(use_dspy=False))
    results.append(await benchmark_claim_extractor(use_dspy=True))
    print()

    print("Running SummarizeExtractor benchmarks...")
    results.append(await benchmark_description_summary(use_dspy=False))
    results.append(await benchmark_description_summary(use_dspy=True))
    print()

    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Group by extractor
    extractors = {}
    for result in results:
        if result.extractor_name not in extractors:
            extractors[result.extractor_name] = {}
        extractors[result.extractor_name][result.mode] = result

    for extractor_name, modes in extractors.items():
        print(f"\n{extractor_name}:")
        print("-" * 80)

        legacy = modes.get("legacy")
        dspy = modes.get("dspy")

        if legacy and dspy:
            print(f"{'Metric':<30} {'Legacy':<20} {'DSPy':<20} {'Change':<20}")
            print("-" * 80)

            if legacy.success and dspy.success:
                # LLM Calls
                call_change = ((dspy.llm_calls - legacy.llm_calls) / legacy.llm_calls * 100) if legacy.llm_calls > 0 else 0
                call_indicator = "📈" if call_change > 0 else "📉" if call_change < 0 else "➡️"
                print(f"{'LLM Calls':<30} {legacy.llm_calls:<20} {dspy.llm_calls:<20} {call_indicator} {call_change:+.1f}%")

                # Tokens
                token_change = ((dspy.total_tokens - legacy.total_tokens) / legacy.total_tokens * 100) if legacy.total_tokens > 0 else 0
                token_indicator = "📈" if token_change > 0 else "📉" if token_change < 0 else "➡️"
                print(f"{'Total Tokens (approx)':<30} {legacy.total_tokens:<20} {dspy.total_tokens:<20} {token_indicator} {token_change:+.1f}%")

                # Execution Time
                time_change = ((dspy.execution_time_ms - legacy.execution_time_ms) / legacy.execution_time_ms * 100) if legacy.execution_time_ms > 0 else 0
                time_indicator = "📈" if time_change > 0 else "📉" if time_change < 0 else "➡️"
                print(f"{'Execution Time (ms)':<30} {legacy.execution_time_ms:<20.2f} {dspy.execution_time_ms:<20.2f} {time_indicator} {time_change:+.1f}%")

                # Output Size
                size_change = ((dspy.output_size - legacy.output_size) / legacy.output_size * 100) if legacy.output_size > 0 else 0
                size_indicator = "📈" if size_change > 0 else "📉" if size_change < 0 else "➡️"
                print(f"{'Output Size (chars)':<30} {legacy.output_size:<20} {dspy.output_size:<20} {size_indicator} {size_change:+.1f}%")
            else:
                if not legacy.success:
                    print(f"  Legacy mode failed: {legacy.error}")
                if not dspy.success:
                    print(f"  DSPy mode failed: {dspy.error}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Calculate averages
    total_legacy_calls = sum(r.llm_calls for r in results if r.mode == "legacy" and r.success)
    total_dspy_calls = sum(r.llm_calls for r in results if r.mode == "dspy" and r.success)
    total_legacy_tokens = sum(r.total_tokens for r in results if r.mode == "legacy" and r.success)
    total_dspy_tokens = sum(r.total_tokens for r in results if r.mode == "dspy" and r.success)

    avg_call_change = ((total_dspy_calls - total_legacy_calls) / total_legacy_calls * 100) if total_legacy_calls > 0 else 0
    avg_token_change = ((total_dspy_tokens - total_legacy_tokens) / total_legacy_tokens * 100) if total_legacy_tokens > 0 else 0

    print(f"Total LLM Calls - Legacy: {total_legacy_calls}, DSPy: {total_dspy_calls} ({avg_call_change:+.1f}%)")
    print(f"Total Tokens - Legacy: {total_legacy_tokens}, DSPy: {total_dspy_tokens} ({avg_token_change:+.1f}%)")
    print()

    if avg_call_change > 50:
        print("⚠️  WARNING: DSPy mode shows >50% increase in LLM calls")
        print("   This is expected due to multi-agent patterns (entity + relationship + refinement + formatting)")
        print("   Consider this as trading cost for better prompt engineering and composability")
    elif avg_call_change < -10:
        print("✅ DSPy mode shows improvement in LLM call efficiency!")
    else:
        print("➡️  DSPy mode shows similar LLM call patterns to legacy")

    print()
    print("=" * 80)


def main():
    """Main entry point."""
    asyncio.run(run_all_benchmarks())


if __name__ == "__main__":
    main()
