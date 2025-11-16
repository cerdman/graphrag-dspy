# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""DSPy-based Community Report generation module for GraphRAG."""

from typing import Any

import dspy
from pydantic import BaseModel, Field


class CommunityFinding(BaseModel):
    """A single finding in a community report."""

    summary: str = Field(description="Brief summary of the finding")
    explanation: str = Field(
        description="Detailed explanation with data references"
    )


class CommunityReportOutput(BaseModel):
    """Structured output for community report."""

    title: str = Field(description="Community name representing key entities")
    summary: str = Field(description="Executive summary of community structure")
    rating: float = Field(
        description="Impact severity rating (0-10)", ge=0.0, le=10.0
    )
    rating_explanation: str = Field(
        description="Single sentence explanation of rating"
    )
    findings: list[CommunityFinding] = Field(
        description="List of 5-10 key insights about the community"
    )


class CommunityReportSignature(dspy.Signature):
    """Signature for generating community reports.

    This signature defines the interface for community report generation using DSPy.
    """

    # Input fields
    input_text: str = dspy.InputField(
        desc="Entities, relationships, and optional claims for the community"
    )
    max_report_length: int = dspy.InputField(
        desc="Maximum report length in words", default=1500
    )

    # Output field
    report_json: str = dspy.OutputField(
        desc="""Well-formed JSON string with format:
{
    "title": "<community_name>",
    "summary": "<executive_summary>",
    "rating": <0-10_float>,
    "rating_explanation": "<single_sentence>",
    "findings": [
        {
            "summary": "<insight_summary>",
            "explanation": "<detailed_explanation_with_data_refs>"
        }
    ]
}

Use data references like: [Data: Entities (5, 7); Relationships (23); Claims (2, 7, +more)]
Limit to {max_report_length} words total."""
    )


class CommunityReportGenerator(dspy.Module):
    """DSPy module for community report generation.

    This module uses DSPy to generate comprehensive community reports
    from entity and relationship data.
    """

    def __init__(self):
        """Initialize CommunityReportGenerator."""
        super().__init__()

        # Use ChainOfThought for better reasoning
        self.generator = dspy.ChainOfThought(CommunityReportSignature)

    def forward(
        self, input_text: str, max_report_length: int = 1500
    ) -> dict[str, Any]:
        """Generate community report from input data.

        Args:
            input_text: Entities, relationships, and claims data
            max_report_length: Maximum length in words

        Returns:
            Dictionary with report structure
        """
        result = self.generator(
            input_text=input_text, max_report_length=max_report_length
        )

        # Parse JSON response
        import json

        try:
            report_data = json.loads(result.report_json)
            return report_data
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract from response
            # Return minimal structure
            return {
                "title": "Community Report",
                "summary": result.report_json[:500],
                "rating": 5.0,
                "rating_explanation": "Default rating",
                "findings": [],
            }


# Convenience function for standalone usage
def generate_community_report_dspy(
    input_text: str, max_report_length: int = 1500, **kwargs: Any
) -> dict[str, Any]:
    """Generate community report using DSPy.

    Args:
        input_text: Input data with entities and relationships
        max_report_length: Maximum report length in words
        **kwargs: Additional arguments

    Returns:
        Dictionary with report structure
    """
    generator = CommunityReportGenerator()
    result = generator(
        input_text=input_text, max_report_length=max_report_length
    )
    return result
