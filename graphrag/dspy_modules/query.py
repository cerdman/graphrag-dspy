# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""DSPy modules for GraphRAG query operations."""

import dspy


# ============================================================================
# LOCAL SEARCH
# ============================================================================


class LocalSearchSignature(dspy.Signature):
    """Answer questions using local context from knowledge graph."""

    # Input fields
    context_data: str = dspy.InputField(
        desc="Data tables containing entities, relationships, reports, and sources"
    )
    response_type: str = dspy.InputField(
        desc="Target response length and format specification"
    )
    question: str = dspy.InputField(desc="User's question to answer")

    # Output field
    answer: str = dspy.OutputField(
        desc="""Answer to the question based on the provided context data.

Include data references in the format:
[Data: <dataset name> (record ids); <dataset name> (record ids)]

Example: "Person X is the owner of Company Y [Data: Entities (5, 7); Relationships (23)]."

Do not include information where supporting evidence is not provided.
Style the response in markdown with appropriate sections."""
    )


class LocalSearchModule(dspy.Module):
    """DSPy module for local search queries."""

    def __init__(self):
        """Initialize the local search module."""
        super().__init__()
        self.search = dspy.ChainOfThought(LocalSearchSignature)

    def forward(
        self, context_data: str, response_type: str, question: str
    ) -> dspy.Prediction:
        """
        Answer a question using local context.

        Args:
            context_data: Context data tables
            response_type: Response format specification
            question: User's question

        Returns:
            dspy.Prediction with answer field
        """
        return self.search(
            context_data=context_data,
            response_type=response_type,
            question=question,
        )


# ============================================================================
# GLOBAL SEARCH - MAP
# ============================================================================


class GlobalSearchMapSignature(dspy.Signature):
    """Map operation for global search: analyze community reports for relevance."""

    # Input fields
    community_report: str = dspy.InputField(
        desc="A community report to analyze for relevance"
    )
    question: str = dspy.InputField(desc="User's question")
    response_type: str = dspy.InputField(desc="Target response format")

    # Output field
    analysis: str = dspy.OutputField(
        desc="Analysis of how this community report relates to the question, with data references"
    )


class GlobalSearchMapModule(dspy.Module):
    """DSPy module for global search map operation."""

    def __init__(self):
        """Initialize the global search map module."""
        super().__init__()
        self.map = dspy.ChainOfThought(GlobalSearchMapSignature)

    def forward(
        self, community_report: str, question: str, response_type: str
    ) -> dspy.Prediction:
        """
        Analyze a community report for relevance to question.

        Args:
            community_report: Report to analyze
            question: User's question
            response_type: Response format

        Returns:
            dspy.Prediction with analysis field
        """
        return self.map(
            community_report=community_report,
            question=question,
            response_type=response_type,
        )


# ============================================================================
# GLOBAL SEARCH - REDUCE
# ============================================================================


class GlobalSearchReduceSignature(dspy.Signature):
    """Reduce operation for global search: synthesize analyses into final answer."""

    # Input fields
    analyses: str = dspy.InputField(
        desc="Collection of community analyses from the map step"
    )
    question: str = dspy.InputField(desc="User's question")
    response_type: str = dspy.InputField(desc="Target response format")

    # Output field
    answer: str = dspy.OutputField(
        desc="Comprehensive answer synthesizing all analyses, with data references in markdown format"
    )


class GlobalSearchReduceModule(dspy.Module):
    """DSPy module for global search reduce operation."""

    def __init__(self):
        """Initialize the global search reduce module."""
        super().__init__()
        self.reduce = dspy.ChainOfThought(GlobalSearchReduceSignature)

    def forward(
        self, analyses: str, question: str, response_type: str
    ) -> dspy.Prediction:
        """
        Synthesize analyses into final answer.

        Args:
            analyses: Combined analyses from map step
            question: User's question
            response_type: Response format

        Returns:
            dspy.Prediction with answer field
        """
        return self.reduce(
            analyses=analyses, question=question, response_type=response_type
        )


# ============================================================================
# DRIFT SEARCH
# ============================================================================


class DriftSearchSignature(dspy.Signature):
    """Follow-up question answering with conversation drift handling."""

    # Input fields
    conversation_history: str = dspy.InputField(
        desc="Previous conversation turns"
    )
    current_question: str = dspy.InputField(desc="Current follow-up question")
    context_data: str = dspy.InputField(desc="Relevant context data")
    response_type: str = dspy.InputField(desc="Target response format")

    # Output field
    answer: str = dspy.OutputField(
        desc="Answer that accounts for conversation history and drift, with data references"
    )


class DriftSearchModule(dspy.Module):
    """DSPy module for drift search (follow-up questions)."""

    def __init__(self):
        """Initialize the drift search module."""
        super().__init__()
        self.search = dspy.ChainOfThought(DriftSearchSignature)

    def forward(
        self,
        conversation_history: str,
        current_question: str,
        context_data: str,
        response_type: str,
    ) -> dspy.Prediction:
        """
        Answer follow-up question with drift handling.

        Args:
            conversation_history: Previous conversation
            current_question: Current question
            context_data: Context data
            response_type: Response format

        Returns:
            dspy.Prediction with answer field
        """
        return self.search(
            conversation_history=conversation_history,
            current_question=current_question,
            context_data=context_data,
            response_type=response_type,
        )


# ============================================================================
# QUESTION GENERATION
# ============================================================================


class QuestionGenSignature(dspy.Signature):
    """Generate candidate questions from context."""

    # Input fields
    context: str = dspy.InputField(desc="Context to generate questions from")
    num_questions: int = dspy.InputField(
        desc="Number of questions to generate", default=5
    )

    # Output field
    questions: str = dspy.OutputField(
        desc="Generated questions, one per line, that could be answered by the context"
    )


class QuestionGenModule(dspy.Module):
    """DSPy module for question generation."""

    def __init__(self):
        """Initialize the question generation module."""
        super().__init__()
        self.generate = dspy.ChainOfThought(QuestionGenSignature)

    def forward(self, context: str, num_questions: int = 5) -> dspy.Prediction:
        """
        Generate questions from context.

        Args:
            context: Context text
            num_questions: Number of questions to generate

        Returns:
            dspy.Prediction with questions field
        """
        return self.generate(context=context, num_questions=num_questions)
