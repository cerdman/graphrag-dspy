# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CommunityReportsResult' and 'CommunityReportsExtractor' models."""

import logging
import traceback
from dataclasses import dataclass

import dspy
from pydantic import BaseModel, Field

from graphrag.dspy_modules.index import CommunityReportModule
from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.language_model.protocol.base import ChatModel
from graphrag.language_model.providers.dspy.adapter import GraphRAGDSpyLM
from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT

logger = logging.getLogger(__name__)

# these tokens are used in the prompt
INPUT_TEXT_KEY = "input_text"
MAX_LENGTH_KEY = "max_report_length"


class FindingModel(BaseModel):
    """A model for the expected LLM response shape."""

    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")


class CommunityReportResponse(BaseModel):
    """A model for the expected LLM response shape."""

    title: str = Field(description="The title of the report.")
    summary: str = Field(description="A summary of the report.")
    findings: list[FindingModel] = Field(
        description="A list of findings in the report."
    )
    rating: float = Field(description="The rating of the report.")
    rating_explanation: str = Field(description="An explanation of the rating.")


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: CommunityReportResponse | None


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _model: ChatModel
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int
    _use_dspy: bool
    _dspy_lm: GraphRAGDSpyLM | None
    _dspy_module: CommunityReportModule | None

    def __init__(
        self,
        model_invoker: ChatModel,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
        use_dspy: bool = True,  # Enable DSPy by default
    ):
        """Init method definition."""
        self._model = model_invoker
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500
        self._use_dspy = use_dspy

        # Initialize DSPy components if enabled
        if self._use_dspy:
            self._dspy_lm = GraphRAGDSpyLM(chat_model=model_invoker)
            # Don't use global dspy.configure() - use context manager instead
            self._dspy_module = CommunityReportModule()
        else:
            self._dspy_lm = None
            self._dspy_module = None

    async def __call__(self, input_text: str):
        """Call method definition."""
        if self._use_dspy and self._dspy_module is not None:
            return await self._generate_report_dspy(input_text)
        else:
            return await self._generate_report_legacy(input_text)

    async def _generate_report_dspy(self, input_text: str):
        """Generate report using DSPy module."""
        output = None
        try:
            import asyncio

            def run_dspy():
                if self._dspy_module is None or self._dspy_lm is None:
                    return None
                # Use context manager to avoid global state pollution
                with dspy.context(lm=self._dspy_lm):
                    result = self._dspy_module.forward(
                        input_text=input_text,
                        max_report_length=self._max_report_length,
                    )
                    # DSPy returns a Prediction with a 'report' field
                    return result.report

            # Run DSPy in thread pool
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, run_dspy)

        except Exception as e:
            logger.exception("error generating community report (dspy)")
            self._on_error(e, traceback.format_exc(), None)

        text_output = self._get_text_output(output) if output else ""
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )

    async def _generate_report_legacy(self, input_text: str):
        """Generate report using legacy prompt-based approach."""
        output = None
        try:
            prompt = self._extraction_prompt.format(**{
                INPUT_TEXT_KEY: input_text,
                MAX_LENGTH_KEY: str(self._max_report_length),
            })
            response = await self._model.achat(
                prompt,
                json=True,  # Leaving this as True to avoid creating new cache entries
                name="create_community_report",
                json_model=CommunityReportResponse,  # A model is required when using json mode
            )

            output = response.parsed_response
        except Exception as e:
            logger.exception("error generating community report")
            self._on_error(e, traceback.format_exc(), None)

        text_output = self._get_text_output(output) if output else ""
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )

    def _get_text_output(self, report: CommunityReportResponse) -> str:
        report_sections = "\n\n".join(
            f"## {f.summary}\n\n{f.explanation}" for f in report.findings
        )
        return f"# {report.title}\n\n{report.summary}\n\n{report_sections}"
