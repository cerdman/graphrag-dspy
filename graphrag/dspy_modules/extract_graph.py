# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""DSPy-based Graph Extraction module for GraphRAG."""

from typing import Any

import dspy


class GraphExtractionSignature(dspy.Signature):
    """Signature for extracting entities and relationships from text.

    This signature defines the interface for graph extraction using DSPy.
    It extracts entities and relationships in a structured format.
    """

    # Input fields
    text: str = dspy.InputField(
        desc="The text document to extract entities and relationships from"
    )
    entity_types: str = dspy.InputField(
        desc="Comma-separated list of entity types to extract (e.g., 'PERSON,ORGANIZATION,GEO')"
    )
    tuple_delimiter: str = dspy.InputField(
        desc="Delimiter for tuple fields", default="<|>"
    )
    record_delimiter: str = dspy.InputField(
        desc="Delimiter between records", default="##"
    )
    completion_delimiter: str = dspy.InputField(
        desc="Delimiter to mark completion", default="<|COMPLETE|>"
    )

    # Output fields
    extracted_data: str = dspy.OutputField(
        desc="""Extracted entities and relationships in the format:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
{record_delimiter}
("relationship"{tuple_delimiter}<source>{tuple_delimiter}<target>{tuple_delimiter}<description>{tuple_delimiter}<strength>)
{record_delimiter}
{completion_delimiter}

Example:
("entity"{tuple_delimiter}MICROSOFT{tuple_delimiter}ORGANIZATION{tuple_delimiter}Microsoft is a technology company)
{record_delimiter}
("entity"{tuple_delimiter}SATYA NADELLA{tuple_delimiter}PERSON{tuple_delimiter}Satya Nadella is CEO of Microsoft)
{record_delimiter}
("relationship"{tuple_delimiter}SATYA NADELLA{tuple_delimiter}MICROSOFT{tuple_delimiter}Satya Nadella is the CEO of Microsoft{tuple_delimiter}9)
{record_delimiter}
{completion_delimiter}"""
    )


class GraphExtractor(dspy.Module):
    """DSPy module for graph extraction.

    This module uses DSPy to extract entities and relationships from text
    in a structured format compatible with GraphRAG.
    """

    def __init__(self, max_gleanings: int = 1):
        """Initialize GraphExtractor.

        Args:
            max_gleanings: Maximum number of follow-up extraction rounds
        """
        super().__init__()
        self.max_gleanings = max_gleanings

        # Use ChainOfThought for better reasoning
        self.extractor = dspy.ChainOfThought(GraphExtractionSignature)

        # For gleanings (follow-up extractions)
        self.gleaning_signature = dspy.Signature(
            "previous_extractions, context -> additional_extractions",
            "Given previous extractions, find any additional entities or relationships that were missed.",
        )
        self.gleaner = dspy.ChainOfThought(self.gleaning_signature)

    def forward(
        self,
        text: str,
        entity_types: str = "organization,person,geo,event",
        tuple_delimiter: str = "<|>",
        record_delimiter: str = "##",
        completion_delimiter: str = "<|COMPLETE|>",
    ) -> str:
        """Extract entities and relationships from text.

        Args:
            text: The input text to process
            entity_types: Comma-separated list of entity types
            tuple_delimiter: Delimiter for tuple fields
            record_delimiter: Delimiter between records
            completion_delimiter: Delimiter marking completion

        Returns:
            Extracted entities and relationships as formatted string
        """
        # Initial extraction
        result = self.extractor(
            text=text,
            entity_types=entity_types,
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
        )

        extracted_data = result.extracted_data

        # Perform gleanings if configured
        if self.max_gleanings > 0:
            for i in range(self.max_gleanings):
                try:
                    # Ask for additional entities/relationships
                    gleaning_result = self.gleaner(
                        previous_extractions=extracted_data,
                        context=f"Original text: {text}\nEntity types: {entity_types}",
                    )

                    additional = gleaning_result.additional_extractions

                    # Check if model says there are more
                    if additional and additional.strip().upper() not in [
                        "NONE",
                        "N",
                        "NO",
                        "",
                    ]:
                        # Append additional extractions
                        extracted_data += f"\n{record_delimiter}\n{additional}"
                    else:
                        # No more entities found
                        break
                except Exception:
                    # If gleaning fails, continue with what we have
                    break

        return extracted_data


# Convenience function for standalone usage
def extract_graph_dspy(
    text: str,
    entity_types: list[str] | None = None,
    max_gleanings: int = 1,
    **kwargs: Any,
) -> str:
    """Extract graph from text using DSPy.

    Args:
        text: Input text
        entity_types: List of entity types to extract
        max_gleanings: Number of follow-up extraction rounds
        **kwargs: Additional arguments

    Returns:
        Extracted graph data as formatted string
    """
    extractor = GraphExtractor(max_gleanings=max_gleanings)

    entity_types_str = (
        ",".join(entity_types)
        if entity_types
        else "organization,person,geo,event"
    )

    result = extractor(text=text, entity_types=entity_types_str, **kwargs)

    return result
