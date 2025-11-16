# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for DSPy community reports module."""

import pytest


class TestDSPyCommunityReportGenerator:
    """Tests for DSPy community report generation."""

    def test_import_community_report_generator(self):
        """Test that CommunityReportGenerator can be imported."""
        from graphrag.dspy_modules.community_reports import (
            CommunityReportGenerator,
        )

        assert CommunityReportGenerator is not None

    def test_community_report_generator_initialization(self):
        """Test CommunityReportGenerator initialization."""
        from graphrag.dspy_modules.community_reports import (
            CommunityReportGenerator,
        )

        generator = CommunityReportGenerator()
        assert generator is not None

    def test_community_report_signature_fields(self):
        """Test that CommunityReportSignature has correct fields."""
        from graphrag.dspy_modules.community_reports import (
            CommunityReportSignature,
        )

        assert CommunityReportSignature is not None

    def test_generate_community_report_convenience_function(self):
        """Test convenience function exists."""
        from graphrag.dspy_modules.community_reports import (
            generate_community_report_dspy,
        )

        assert callable(generate_community_report_dspy)


class TestCommunityReportOutput:
    """Tests for CommunityReportOutput model."""

    def test_community_report_output_model(self):
        """Test CommunityReportOutput Pydantic model."""
        from graphrag.dspy_modules.community_reports import (
            CommunityReportOutput,
            CommunityFinding,
        )

        # Create sample output
        output = CommunityReportOutput(
            title="Test Community",
            summary="Test summary",
            rating=7.5,
            rating_explanation="High impact community",
            findings=[
                CommunityFinding(
                    summary="Finding 1", explanation="Explanation 1"
                )
            ],
        )

        assert output.title == "Test Community"
        assert output.rating == 7.5
        assert len(output.findings) == 1

    def test_community_report_output_validation(self):
        """Test rating validation (0-10 range)."""
        from graphrag.dspy_modules.community_reports import (
            CommunityReportOutput,
        )
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

        # Invalid rating (should fail if > 10)
        with pytest.raises(ValidationError):
            CommunityReportOutput(
                title="Test",
                summary="Summary",
                rating=15.0,  # Invalid: > 10
                rating_explanation="Explanation",
                findings=[],
            )


class TestCommunityReportGeneratorModule:
    """Tests for CommunityReportGenerator DSPy module."""

    def test_module_initialization(self):
        """Test module can be initialized."""
        from graphrag.dspy_modules.community_reports import (
            CommunityReportGenerator,
        )

        module = CommunityReportGenerator()
        assert module is not None
        assert hasattr(module, "generator")

    def test_module_forward_signature(self):
        """Test forward method has correct signature."""
        from graphrag.dspy_modules.community_reports import (
            CommunityReportGenerator,
        )
        import inspect

        module = CommunityReportGenerator()
        sig = inspect.signature(module.forward)

        # Check required parameters
        assert "input_text" in sig.parameters
        assert "max_report_length" in sig.parameters
