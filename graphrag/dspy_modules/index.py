# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""DSPy modules for GraphRAG index operations."""

import dspy
from pydantic import BaseModel, Field


# ============================================================================
# GRAPH EXTRACTION
# ============================================================================


class GraphExtractionSignature(dspy.Signature):
    """
    Extract entities and relationships from text.

    Given a text document and entity types, identify all entities of those
    types and all relationships among the identified entities.
    """

    # Input fields
    entity_types: str = dspy.InputField(
        desc="Comma-separated list of entity types to extract (e.g., 'ORGANIZATION,PERSON,GEO')"
    )
    input_text: str = dspy.InputField(desc="The text document to analyze")
    tuple_delimiter: str = dspy.InputField(
        desc="Delimiter to use between tuple fields", default="<|>"
    )
    record_delimiter: str = dspy.InputField(
        desc="Delimiter to use between records", default="##"
    )
    completion_delimiter: str = dspy.InputField(
        desc="Delimiter to indicate completion", default="<|COMPLETE|>"
    )

    # Output field
    extracted_data: str = dspy.OutputField(
        desc="""Extracted entities and relationships in the following format:

For each entity: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
For each relationship: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

Separate records with {record_delimiter}.
End with {completion_delimiter}.

Example:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal is a stock listed on the Global Exchange)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}Vision Holdings owned TechGlobal from 2014{tuple_delimiter}5)
{completion_delimiter}"""
    )


class GraphExtractionModule(dspy.Module):
    """
    DSPy module for extracting graph entities and relationships.

    This module uses iterative refinement (gleanings) to extract comprehensive
    entity and relationship information from text documents.
    """

    def __init__(self, max_gleanings: int = 1):
        """
        Initialize the graph extraction module.

        Args:
            max_gleanings: Maximum number of refinement iterations
        """
        super().__init__()
        self.max_gleanings = max_gleanings

        # Use ChainOfThought for more structured extraction
        self.extract = dspy.ChainOfThought(GraphExtractionSignature)
        self.continue_extract = dspy.Predict(
            "previous_extraction: str -> additional_extraction: str"
        )
        self.check_completion = dspy.Predict(
            "extractions: str -> needs_more: bool"
        )

    def forward(
        self,
        entity_types: str,
        input_text: str,
        tuple_delimiter: str = "<|>",
        record_delimiter: str = "##",
        completion_delimiter: str = "<|COMPLETE|>",
    ) -> dspy.Prediction:
        """
        Extract entities and relationships from text.

        Args:
            entity_types: Comma-separated entity types
            input_text: Text to analyze
            tuple_delimiter: Delimiter for tuple fields
            record_delimiter: Delimiter for records
            completion_delimiter: Completion marker

        Returns:
            dspy.Prediction with extracted_data field
        """
        # Initial extraction
        result = self.extract(
            entity_types=entity_types,
            input_text=input_text,
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
        )

        extracted_data = result.extracted_data

        # Perform gleanings if configured
        if self.max_gleanings > 0:
            for i in range(self.max_gleanings):
                # Ask for additional entities/relationships
                continue_prompt = (
                    "MANY entities and relationships were missed in the last extraction. "
                    "Remember to ONLY emit entities that match the specified types. "
                    "Add them below using the same format:"
                )

                additional = self.continue_extract(
                    previous_extraction=extracted_data + "\n" + continue_prompt
                )
                extracted_data += additional.additional_extraction

                # Check if more extraction is needed (skip on last iteration)
                if i < self.max_gleanings - 1:
                    check = self.check_completion(extractions=extracted_data)
                    # If needs_more is False, we can stop early
                    if not getattr(check, "needs_more", True):
                        break

        return dspy.Prediction(extracted_data=extracted_data)


# ============================================================================
# COMMUNITY REPORT
# ============================================================================


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


class CommunityReportSignature(dspy.Signature):
    """Generate a comprehensive community report from entities and relationships."""

    # Input fields
    input_text: str = dspy.InputField(
        desc="Text containing entities, relationships, and claims about the community"
    )
    max_report_length: int = dspy.InputField(
        desc="Maximum word count for the report", default=1500
    )

    # Output field (structured JSON)
    report: CommunityReportResponse = dspy.OutputField(
        desc="Structured community report with title, summary, rating, and findings"
    )


class CommunityReportModule(dspy.Module):
    """DSPy module for generating community reports."""

    def __init__(self):
        """Initialize the community report module."""
        super().__init__()
        self.generate = dspy.ChainOfThought(CommunityReportSignature)

    def forward(
        self, input_text: str, max_report_length: int = 1500
    ) -> dspy.Prediction:
        """
        Generate a community report.

        Args:
            input_text: Text with entity/relationship data
            max_report_length: Maximum word count

        Returns:
            dspy.Prediction with report field
        """
        return self.generate(
            input_text=input_text, max_report_length=max_report_length
        )


# ============================================================================
# CLAIM EXTRACTION
# ============================================================================


class ClaimExtractionSignature(dspy.Signature):
    """Extract claims (covariates) from text."""

    # Input fields
    input_text: str = dspy.InputField(desc="Text to extract claims from")
    entity_specs: str = dspy.InputField(
        desc="Specification of entities to focus on"
    )

    # Output field
    extracted_claims: str = dspy.OutputField(
        desc="Extracted claims in structured format"
    )


class ClaimExtractionModule(dspy.Module):
    """DSPy module for extracting claims."""

    def __init__(self):
        """Initialize the claim extraction module."""
        super().__init__()
        self.extract = dspy.ChainOfThought(ClaimExtractionSignature)

    def forward(self, input_text: str, entity_specs: str = "") -> dspy.Prediction:
        """
        Extract claims from text.

        Args:
            input_text: Text to analyze
            entity_specs: Entity specifications

        Returns:
            dspy.Prediction with extracted_claims
        """
        return self.extract(input_text=input_text, entity_specs=entity_specs)


# ============================================================================
# DESCRIPTION SUMMARY
# ============================================================================


class DescriptionSummarySignature(dspy.Signature):
    """Summarize entity or relationship descriptions."""

    # Input fields
    descriptions: str = dspy.InputField(
        desc="List of descriptions to summarize, separated by newlines"
    )

    # Output field
    summary: str = dspy.OutputField(
        desc="Concise summary that captures the key information from all descriptions"
    )


class DescriptionSummaryModule(dspy.Module):
    """DSPy module for summarizing descriptions."""

    def __init__(self):
        """Initialize the description summary module."""
        super().__init__()
        self.summarize = dspy.ChainOfThought(DescriptionSummarySignature)

    def forward(self, descriptions: str) -> dspy.Prediction:
        """
        Summarize descriptions.

        Args:
            descriptions: Newline-separated descriptions

        Returns:
            dspy.Prediction with summary field
        """
        return self.summarize(descriptions=descriptions)
