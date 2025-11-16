# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Advanced DSPy modules for GraphRAG index operations using multi-agent patterns."""

import dspy
from pydantic import BaseModel, Field


# ============================================================================
# GRAPH EXTRACTION - Multi-Agent Pattern with ReAct
# ============================================================================


class EntityIdentificationSignature(dspy.Signature):
    """Agent 1: Identify entities from text."""

    entity_types: str = dspy.InputField(desc="Allowed entity types")
    input_text: str = dspy.InputField(desc="Text to analyze")
    rationale: str = dspy.OutputField(desc="Reasoning about entities present")
    entities: str = dspy.OutputField(
        desc="Identified entities with types and descriptions"
    )


class RelationshipIdentificationSignature(dspy.Signature):
    """Agent 2: Identify relationships between entities."""

    entities: str = dspy.InputField(desc="Previously identified entities")
    input_text: str = dspy.InputField(desc="Original text")
    rationale: str = dspy.OutputField(desc="Reasoning about relationships")
    relationships: str = dspy.OutputField(
        desc="Relationships with source, target, description, strength"
    )


class ExtractionRefinementSignature(dspy.Signature):
    """Agent 3: Refine and validate extractions."""

    entities: str = dspy.InputField(desc="Extracted entities")
    relationships: str = dspy.InputField(desc="Extracted relationships")
    entity_types: str = dspy.InputField(desc="Allowed types")
    refinement_rationale: str = dspy.OutputField(desc="What needs improvement")
    needs_refinement: bool = dspy.OutputField(desc="True if more work needed")
    refined_entities: str = dspy.OutputField(desc="Refined entity list")
    refined_relationships: str = dspy.OutputField(desc="Refined relationship list")


class ExtractionFormattingSignature(dspy.Signature):
    """Agent 4: Format extractions into GraphRAG format."""

    entities: str = dspy.InputField(desc="Refined entities")
    relationships: str = dspy.InputField(desc="Refined relationships")
    tuple_delimiter: str = dspy.InputField(desc="Tuple delimiter")
    record_delimiter: str = dspy.InputField(desc="Record delimiter")
    completion_delimiter: str = dspy.InputField(desc="Completion marker")
    formatted_output: str = dspy.OutputField(
        desc="Properly formatted extraction output"
    )


class GraphExtractionModule(dspy.Module):
    """
    Advanced multi-agent graph extraction using DSPy.

    Uses 4 specialized agents:
    1. Entity Identifier (ChainOfThought)
    2. Relationship Identifier (ChainOfThought)
    3. Extraction Refiner (ReAct-style iteration)
    4. Output Formatter (Predict)

    This demonstrates DSPy's composition and multi-agent capabilities.
    """

    def __init__(self, max_gleanings: int = 1):
        """Initialize multi-agent graph extraction."""
        super().__init__()
        self.max_gleanings = max_gleanings

        # Agent 1: Entity identification with reasoning
        self.entity_agent = dspy.ChainOfThought(EntityIdentificationSignature)

        # Agent 2: Relationship identification with reasoning
        self.relationship_agent = dspy.ChainOfThought(
            RelationshipIdentificationSignature
        )

        # Agent 3: Refinement agent (checks and improves)
        self.refinement_agent = dspy.ChainOfThought(ExtractionRefinementSignature)

        # Agent 4: Formatter
        self.formatter = dspy.Predict(ExtractionFormattingSignature)

    def forward(
        self,
        entity_types: str,
        input_text: str,
        tuple_delimiter: str = "<|>",
        record_delimiter: str = "##",
        completion_delimiter: str = "<|COMPLETE|>",
    ) -> dspy.Prediction:
        """
        Multi-agent extraction with iterative refinement.

        Flow:
        1. Entity Agent identifies entities with rationale
        2. Relationship Agent identifies relationships with rationale
        3. Refinement Agent checks and refines (iterative)
        4. Formatter produces final output

        Args:
            entity_types: Allowed entity types
            input_text: Text to process
            tuple_delimiter, record_delimiter, completion_delimiter: Formatting

        Returns:
            dspy.Prediction with extracted_data
        """
        # Agent 1: Identify entities
        entity_result = self.entity_agent(
            entity_types=entity_types, input_text=input_text
        )

        # Agent 2: Identify relationships based on entities
        relationship_result = self.relationship_agent(
            entities=entity_result.entities, input_text=input_text
        )

        # Agent 3: Iterative refinement (ReAct-style)
        entities = entity_result.entities
        relationships = relationship_result.relationships

        for i in range(self.max_gleanings + 1):
            refinement = self.refinement_agent(
                entities=entities,
                relationships=relationships,
                entity_types=entity_types,
            )

            entities = refinement.refined_entities
            relationships = refinement.refined_relationships

            # Stop if no more refinement needed
            if not refinement.needs_refinement:
                break

        # Agent 4: Format the output
        formatted = self.formatter(
            entities=entities,
            relationships=relationships,
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
        )

        return dspy.Prediction(extracted_data=formatted.formatted_output)


# ============================================================================
# COMMUNITY REPORT - Parallel Analysis with Synthesis
# ============================================================================


class CommunityStructureAnalysisSignature(dspy.Signature):
    """Analyst 1: Analyze community structure."""

    input_text: str = dspy.InputField(desc="Community data")
    structure_analysis: str = dspy.OutputField(
        desc="Analysis of how entities are connected"
    )
    key_entities: str = dspy.OutputField(desc="Most important entities")


class CommunityImpactAnalysisSignature(dspy.Signature):
    """Analyst 2: Analyze community impact and importance."""

    input_text: str = dspy.InputField(desc="Community data")
    impact_analysis: str = dspy.OutputField(desc="Analysis of community impact")
    severity_rating: float = dspy.OutputField(desc="Impact rating 0-10")
    rating_rationale: str = dspy.OutputField(desc="Why this rating")


class CommunityFindingsSignature(dspy.Signature):
    """Analyst 3: Extract key findings."""

    input_text: str = dspy.InputField(desc="Community data")
    findings_list: str = dspy.OutputField(desc="List of 5-10 key insights")


class ReportSynthesisSignature(dspy.Signature):
    """Synthesizer: Combine analyses into final report."""

    structure_analysis: str = dspy.InputField(desc="Structure analysis")
    impact_analysis: str = dspy.InputField(desc="Impact analysis")
    findings: str = dspy.InputField(desc="Key findings")
    severity_rating: float = dspy.InputField(desc="Impact rating")
    rating_rationale: str = dspy.InputField(desc="Rating explanation")
    max_length: int = dspy.InputField(desc="Max word count")

    report: str = dspy.OutputField(
        desc="Complete structured report as JSON string"
    )


class FindingModel(BaseModel):
    """A model for community report findings."""

    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")


class CommunityReportResponse(BaseModel):
    """A model for community report structure."""

    title: str = Field(description="The title of the report.")
    summary: str = Field(description="A summary of the report.")
    findings: list[FindingModel] = Field(
        description="A list of findings in the report."
    )
    rating: float = Field(description="The impact severity rating (0-10).")
    rating_explanation: str = Field(description="An explanation of the rating.")


class CommunityReportModule(dspy.Module):
    """
    Multi-analyst community report generation.

    Uses parallel analysis with synthesis pattern:
    - 3 specialist analysts work in parallel
    - Synthesizer combines their outputs
    - Demonstrates DSPy's composability
    """

    def __init__(self):
        """Initialize multi-analyst report generation."""
        super().__init__()

        # Three parallel analysts with ChainOfThought
        self.structure_analyst = dspy.ChainOfThought(
            CommunityStructureAnalysisSignature
        )
        self.impact_analyst = dspy.ChainOfThought(CommunityImpactAnalysisSignature)
        self.findings_analyst = dspy.ChainOfThought(CommunityFindingsSignature)

        # Synthesizer combines analyses
        self.synthesizer = dspy.ChainOfThought(ReportSynthesisSignature)

    def forward(
        self, input_text: str, max_report_length: int = 1500
    ) -> dspy.Prediction:
        """
        Generate report using parallel analysts + synthesizer.

        Flow:
        1. Three analysts work in parallel
        2. Synthesizer combines their insights
        3. Produces structured final report

        Args:
            input_text: Community data
            max_report_length: Max words

        Returns:
            dspy.Prediction with report field
        """
        # Parallel analysis by three specialists
        structure_result = self.structure_analyst(input_text=input_text)
        impact_result = self.impact_analyst(input_text=input_text)
        findings_result = self.findings_analyst(input_text=input_text)

        # Synthesize into final report
        synthesis_result = self.synthesizer(
            structure_analysis=structure_result.structure_analysis,
            impact_analysis=impact_result.impact_analysis,
            findings=findings_result.findings_list,
            severity_rating=impact_result.severity_rating,
            rating_rationale=impact_result.rating_rationale,
            max_length=max_report_length,
        )

        # Parse the JSON report string into CommunityReportResponse
        import json
        report_data = json.loads(synthesis_result.report) if isinstance(synthesis_result.report, str) else synthesis_result.report
        report_obj = CommunityReportResponse(**report_data) if isinstance(report_data, dict) else report_data

        return dspy.Prediction(report=report_obj)


# ============================================================================
# CLAIM EXTRACTION - ReAct Pattern
# ============================================================================


class ClaimIdentificationSignature(dspy.Signature):
    """Identify potential claims in text."""

    input_text: str = dspy.InputField(desc="Text to analyze")
    entity_specs: str = dspy.InputField(desc="Entity specifications")
    thought: str = dspy.OutputField(desc="Reasoning about claims")
    initial_claims: str = dspy.OutputField(desc="Initial claim candidates")


class ClaimValidationSignature(dspy.Signature):
    """Validate and refine claims."""

    claims: str = dspy.InputField(desc="Claims to validate")
    input_text: str = dspy.InputField(desc="Original text")
    validation_thought: str = dspy.OutputField(desc="Validation reasoning")
    validated_claims: str = dspy.OutputField(desc="Validated claims")
    needs_more_extraction: bool = dspy.OutputField(desc="Should extract more?")


class ClaimExtractionModule(dspy.Module):
    """
    ReAct-style claim extraction with validation loop.

    Demonstrates iterative reasoning pattern.
    """

    def __init__(self, max_iterations: int = 2):
        """Initialize ReAct claim extraction."""
        super().__init__()
        self.max_iterations = max_iterations

        self.identifier = dspy.ChainOfThought(ClaimIdentificationSignature)
        self.validator = dspy.ChainOfThought(ClaimValidationSignature)

    def forward(self, input_text: str, entity_specs: str = "") -> dspy.Prediction:
        """
        Extract claims using ReAct pattern.

        Args:
            input_text: Text to process
            entity_specs: Entity specifications

        Returns:
            dspy.Prediction with extracted_claims
        """
        # Initial identification
        result = self.identifier(input_text=input_text, entity_specs=entity_specs)
        claims = result.initial_claims

        # Validation loop (ReAct)
        for _ in range(self.max_iterations):
            validation = self.validator(claims=claims, input_text=input_text)
            claims = validation.validated_claims

            if not validation.needs_more_extraction:
                break

        return dspy.Prediction(extracted_claims=claims)


# ============================================================================
# DESCRIPTION SUMMARY - Dual-Path Reasoning
# ============================================================================


class ExtractiveSummarySignature(dspy.Signature):
    """Path 1: Extractive summarization."""

    descriptions: str = dspy.InputField(desc="Input descriptions")
    key_points: str = dspy.OutputField(desc="Extracted key points")


class AbstractiveSummarySignature(dspy.Signature):
    """Path 2: Abstractive summarization."""

    descriptions: str = dspy.InputField(desc="Input descriptions")
    abstract_summary: str = dspy.OutputField(desc="Abstract summary")


class SummaryFusionSignature(dspy.Signature):
    """Fuse extractive and abstractive summaries."""

    key_points: str = dspy.InputField(desc="Extractive key points")
    abstract_summary: str = dspy.InputField(desc="Abstractive summary")
    fused_summary: str = dspy.OutputField(desc="Best combined summary")


class DescriptionSummaryModule(dspy.Module):
    """
    Dual-path summarization with fusion.

    Two parallel reasoning paths:
    - Extractive (preserve exact info)
    - Abstractive (synthesize concepts)
    Then fuses both for best result.
    """

    def __init__(self):
        """Initialize dual-path summarization."""
        super().__init__()

        self.extractive_path = dspy.ChainOfThought(ExtractiveSummarySignature)
        self.abstractive_path = dspy.ChainOfThought(AbstractiveSummarySignature)
        self.fusion = dspy.ChainOfThought(SummaryFusionSignature)

    def forward(self, descriptions: str) -> dspy.Prediction:
        """
        Summarize using dual-path + fusion.

        Args:
            descriptions: Input descriptions

        Returns:
            dspy.Prediction with summary
        """
        # Two parallel paths
        extractive = self.extractive_path(descriptions=descriptions)
        abstractive = self.abstractive_path(descriptions=descriptions)

        # Fuse results
        fused = self.fusion(
            key_points=extractive.key_points,
            abstract_summary=abstractive.abstract_summary,
        )

        return dspy.Prediction(summary=fused.fused_summary)
