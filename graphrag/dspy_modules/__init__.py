# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""DSPy modules for GraphRAG prompt execution."""

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

__all__ = [
    # Index modules
    "GraphExtractionModule",
    "CommunityReportModule",
    "ClaimExtractionModule",
    "DescriptionSummaryModule",
    # Query modules
    "LocalSearchModule",
    "GlobalSearchMapModule",
    "GlobalSearchReduceModule",
    "DriftSearchModule",
    "QuestionGenModule",
]
