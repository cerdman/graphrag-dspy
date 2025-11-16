# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Advanced DSPy modules for GraphRAG query operations using ensemble and ReAct patterns."""

import dspy


# ============================================================================
# LOCAL SEARCH - Ensemble Reasoning Pattern
# ============================================================================


class ContextUnderstandingSignature(dspy.Signature):
    """Agent 1: Understand the provided context."""

    context_data: str = dspy.InputField(desc="Context data tables")
    question: str = dspy.InputField(desc="User's question")
    context_summary: str = dspy.OutputField(desc="Summary of relevant context")
    key_entities: str = dspy.OutputField(desc="Key entities for this question")


class DirectAnswerSignature(dspy.Signature):
    """Agent 2: Generate direct answer from context."""

    context_summary: str = dspy.InputField(desc="Context summary")
    question: str = dspy.InputField(desc="User's question")
    response_type: str = dspy.InputField(desc="Response format")
    direct_answer: str = dspy.OutputField(desc="Direct answer with citations")


class  ReasonedAnswerSignature(dspy.Signature):
    """Agent 3: Generate answer with step-by-step reasoning."""

    context_summary: str = dspy.InputField(desc="Context summary")
    question: str = dspy.InputField(desc="User's question")
    response_type: str = dspy.InputField(desc="Response format")
    reasoning_steps: str = dspy.OutputField(desc="Step-by-step reasoning")
    reasoned_answer: str = dspy.OutputField(desc="Answer from reasoning")


class AnswerSynthesisSignature(dspy.Signature):
    """Synthesizer: Combine direct and reasoned answers."""

    direct_answer: str = dspy.InputField(desc="Direct answer")
    reasoned_answer: str = dspy.InputField(desc="Reasoned answer")
    response_type: str = dspy.InputField(desc="Response format")
    final_answer: str = dspy.OutputField(
        desc="Best synthesized answer with data references in markdown"
    )


class LocalSearchModule(dspy.Module):
    """
    Ensemble local search with multiple reasoning paths.

    Uses 3 agents + synthesizer:
    1. Context Understander
    2. Direct Answerer
    3. Reasoned Answerer (Chain-of-Thought)
    4. Answer Synthesizer
    """

    def __init__(self):
        """Initialize ensemble local search."""
        super().__init__()

        self.context_understander = dspy.ChainOfThought(
            ContextUnderstandingSignature
        )
        self.direct_answerer = dspy.Predict(DirectAnswerSignature)
        self.reasoned_answerer = dspy.ChainOfThought(ReasonedAnswerSignature)
        self.synthesizer = dspy.ChainOfThought(AnswerSynthesisSignature)

    def forward(
        self, context_data: str, response_type: str, question: str
    ) -> dspy.Prediction:
        """
        Answer using ensemble of reasoning paths.

        Args:
            context_data: Context data tables
            response_type: Response format
            question: User's question

        Returns:
            dspy.Prediction with answer
        """
        # Agent 1: Understand context
        context_result = self.context_understander(
            context_data=context_data, question=question
        )

        # Agents 2 & 3: Parallel answering paths
        direct_result = self.direct_answerer(
            context_summary=context_result.context_summary,
            question=question,
            response_type=response_type,
        )

        reasoned_result = self.reasoned_answerer(
            context_summary=context_result.context_summary,
            question=question,
            response_type=response_type,
        )

        # Agent 4: Synthesize
        final = self.synthesizer(
            direct_answer=direct_result.direct_answer,
            reasoned_answer=reasoned_result.reasoned_answer,
            response_type=response_type,
        )

        return dspy.Prediction(answer=final.final_answer)


# ============================================================================
# GLOBAL SEARCH - Hierarchical Map-Reduce with Reflection
# ============================================================================


class CommunityRelevanceSignature(dspy.Signature):
    """Assess relevance of community report to question."""

    community_report: str = dspy.InputField(desc="Community report")
    question: str = dspy.InputField(desc="User's question")
    relevance_score: float = dspy.OutputField(desc="Relevance 0-1")
    relevance_rationale: str = dspy.OutputField(desc="Why relevant/not")


class CommunityAnalysisSignature(dspy.Signature):
    """Analyze community report for question (if relevant)."""

    community_report: str = dspy.InputField(desc="Community report")
    question: str = dspy.InputField(desc="User's question")
    response_type: str = dspy.InputField(desc="Response format")
    analysis: str = dspy.OutputField(desc="Analysis with data references")


class GlobalSearchMapModule(dspy.Module):
    """
    Map operation with relevance filtering.

    Two-stage process:
    1. Relevance checker (filters)
    2. Detailed analyzer (only if relevant)
    """

    def __init__(self, relevance_threshold: float = 0.3):
        """Initialize map module with filtering."""
        super().__init__()
        self.relevance_threshold = relevance_threshold

        self.relevance_checker = dspy.ChainOfThought(CommunityRelevanceSignature)
        self.analyzer = dspy.ChainOfThought(CommunityAnalysisSignature)

    def forward(
        self, community_report: str, question: str, response_type: str
    ) -> dspy.Prediction:
        """
        Analyze community with relevance filtering.

        Args:
            community_report: Report to analyze
            question: User's question
            response_type: Response format

        Returns:
            dspy.Prediction with analysis (or empty if not relevant)
        """
        # Stage 1: Check relevance
        relevance = self.relevance_checker(
            community_report=community_report, question=question
        )

        # Stage 2: Analyze only if relevant
        if relevance.relevance_score >= self.relevance_threshold:
            result = self.analyzer(
                community_report=community_report,
                question=question,
                response_type=response_type,
            )
            analysis = result.analysis
        else:
            analysis = ""  # Skip irrelevant communities

        return dspy.Prediction(analysis=analysis)


class AnswerRefinementSignature(dspy.Signature):
    """Refine and improve the synthesized answer."""

    initial_answer: str = dspy.InputField(desc="Initial synthesized answer")
    question: str = dspy.InputField(desc="Original question")
    reflection: str = dspy.OutputField(desc="What could be improved")
    refined_answer: str = dspy.OutputField(desc="Improved answer")


class GlobalSearchReduceModule(dspy.Module):
    """
    Reduce operation with self-refinement.

    Two-stage synthesis:
    1. Initial synthesis
    2. Self-refinement (reflection)
    """

    def __init__(self):
        """Initialize reduce module with refinement."""
        super().__init__()

        self.initial_synthesizer = dspy.ChainOfThought(
            "analyses: str, question: str, response_type: str -> initial_answer: str"
        )
        self.refiner = dspy.ChainOfThought(AnswerRefinementSignature)

    def forward(
        self, analyses: str, question: str, response_type: str
    ) -> dspy.Prediction:
        """
        Synthesize with self-refinement.

        Args:
            analyses: Combined analyses from map step
            question: User's question
            response_type: Response format

        Returns:
            dspy.Prediction with refined answer
        """
        # Stage 1: Initial synthesis
        initial = self.initial_synthesizer(
            analyses=analyses, question=question, response_type=response_type
        )

        # Stage 2: Self-refinement
        refined = self.refiner(
            initial_answer=initial.initial_answer, question=question
        )

        return dspy.Prediction(answer=refined.refined_answer)


# ============================================================================
# DRIFT SEARCH - Memory-Augmented ReAct
# ============================================================================


class ConversationMemorySignature(dspy.Signature):
    """Extract key information from conversation history."""

    conversation_history: str = dspy.InputField(desc="Previous turns")
    memory_summary: str = dspy.OutputField(desc="Key points to remember")
    topics: str = dspy.OutputField(desc="Topics discussed")


class DriftDetectionSignature(dspy.Signature):
    """Detect if conversation has drifted."""

    memory_summary: str = dspy.InputField(desc="Conversation memory")
    current_question: str = dspy.InputField(desc="Current question")
    drift_analysis: str = dspy.OutputField(desc="Analysis of topic drift")
    has_drifted: bool = dspy.OutputField(desc="True if drifted")


class ContextualAnswerSignature(dspy.Signature):
    """Answer with conversation context."""

    memory_summary: str = dspy.InputField(desc="Conversation memory")
    current_question: str = dspy.InputField(desc="Current question")
    context_data: str = dspy.InputField(desc="Relevant data")
    response_type: str = dspy.InputField(desc="Response format")
    has_drifted: bool = dspy.InputField(desc="Topic drift flag")
    answer: str = dspy.OutputField(desc="Contextual answer with drift handling")


class DriftSearchModule(dspy.Module):
    """
    Conversation-aware search with drift detection.

    Three-stage ReAct-style process:
    1. Memory extraction
    2. Drift detection
    3. Contextual answering
    """

    def __init__(self):
        """Initialize drift-aware search."""
        super().__init__()

        self.memory_extractor = dspy.ChainOfThought(ConversationMemorySignature)
        self.drift_detector = dspy.ChainOfThought(DriftDetectionSignature)
        self.contextual_answerer = dspy.ChainOfThought(ContextualAnswerSignature)

    def forward(
        self,
        conversation_history: str,
        current_question: str,
        context_data: str,
        response_type: str,
    ) -> dspy.Prediction:
        """
        Answer with drift-aware reasoning.

        Args:
            conversation_history: Previous conversation
            current_question: Current question
            context_data: Context data
            response_type: Response format

        Returns:
            dspy.Prediction with drift-aware answer
        """
        # Stage 1: Extract memory
        memory = self.memory_extractor(conversation_history=conversation_history)

        # Stage 2: Detect drift
        drift = self.drift_detector(
            memory_summary=memory.memory_summary, current_question=current_question
        )

        # Stage 3: Answer with context
        answer_result = self.contextual_answerer(
            memory_summary=memory.memory_summary,
            current_question=current_question,
            context_data=context_data,
            response_type=response_type,
            has_drifted=drift.has_drifted,
        )

        return answer_result


# ============================================================================
# QUESTION GENERATION - Creative Divergence + Validation
# ============================================================================


class CreativeQuestionGenSignature(dspy.Signature):
    """Generate diverse candidate questions."""

    context: str = dspy.InputField(desc="Context text")
    num_questions: int = dspy.InputField(desc="Questions to generate")
    creative_questions: str = dspy.OutputField(
        desc="Diverse candidate questions, one per line"
    )


class QuestionValidationSignature(dspy.Signature):
    """Validate and filter generated questions."""

    candidate_questions: str = dspy.InputField(desc="Generated questions")
    context: str = dspy.InputField(desc="Original context")
    validation_rationale: str = dspy.OutputField(desc="Why questions are good/bad")
    filtered_questions: str = dspy.OutputField(
        desc="Valid questions that can be answered by context"
    )


class QuestionGenModule(dspy.Module):
    """
    Two-stage question generation: creative + validation.

    Demonstrates divergence-convergence pattern.
    """

    def __init__(self):
        """Initialize question generation."""
        super().__init__()

        # Stage 1: Creative generation (diverge)
        self.creative_generator = dspy.ChainOfThought(CreativeQuestionGenSignature)

        # Stage 2: Validation (converge)
        self.validator = dspy.ChainOfThought(QuestionValidationSignature)

    def forward(self, context: str, num_questions: int = 5) -> dspy.Prediction:
        """
        Generate and validate questions.

        Args:
            context: Context text
            num_questions: Number of questions

        Returns:
            dspy.Prediction with validated questions
        """
        # Stage 1: Generate creative candidates
        candidates = self.creative_generator(
            context=context, num_questions=num_questions * 2  # Over-generate
        )

        # Stage 2: Validate and filter
        validated = self.validator(
            candidate_questions=candidates.creative_questions, context=context
        )

        return dspy.Prediction(questions=validated.filtered_questions)
