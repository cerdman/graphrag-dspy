# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""DSPy-based modules for GraphRAG prompt execution."""

from graphrag.dspy_modules.community_reports import (
    CommunityReportGenerator as DSPyCommunityReportGenerator,
)
from graphrag.dspy_modules.extract_graph import GraphExtractor as DSPyGraphExtractor

__all__ = ["DSPyGraphExtractor", "DSPyCommunityReportGenerator"]
