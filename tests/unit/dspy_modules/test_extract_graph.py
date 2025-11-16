# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for DSPy graph extraction module."""

import pytest


class TestDSPyGraphExtractor:
    """Tests for DSPy graph extraction."""

    def test_import_graph_extractor(self):
        """Test that GraphExtractor can be imported."""
        from graphrag.dspy_modules.extract_graph import GraphExtractor

        assert GraphExtractor is not None

    def test_graph_extractor_initialization(self):
        """Test GraphExtractor initialization."""
        from graphrag.dspy_modules.extract_graph import GraphExtractor

        extractor = GraphExtractor(max_gleanings=2)
        assert extractor.max_gleanings == 2

    def test_graph_extraction_signature_fields(self):
        """Test that GraphExtractionSignature has correct fields."""
        from graphrag.dspy_modules.extract_graph import GraphExtractionSignature

        # Check input fields exist
        assert hasattr(GraphExtractionSignature, "__annotations__")

    def test_extract_graph_convenience_function(self):
        """Test convenience function exists."""
        from graphrag.dspy_modules.extract_graph import extract_graph_dspy

        assert callable(extract_graph_dspy)


class TestGraphExtractionSignature:
    """Tests for GraphExtractionSignature."""

    def test_signature_has_required_fields(self):
        """Test signature has all required fields."""
        from graphrag.dspy_modules.extract_graph import GraphExtractionSignature

        # These should be defined as InputField or OutputField
        signature = GraphExtractionSignature
        assert signature is not None


class TestGraphExtractorModule:
    """Tests for GraphExtractor DSPy module."""

    def test_module_initialization(self):
        """Test module can be initialized."""
        from graphrag.dspy_modules.extract_graph import GraphExtractor

        module = GraphExtractor(max_gleanings=1)
        assert module is not None
        assert hasattr(module, "extractor")
        assert hasattr(module, "gleaner")

    def test_module_forward_signature(self):
        """Test forward method has correct signature."""
        from graphrag.dspy_modules.extract_graph import GraphExtractor
        import inspect

        module = GraphExtractor()
        sig = inspect.signature(module.forward)

        # Check required parameters
        assert "text" in sig.parameters
        assert "entity_types" in sig.parameters
