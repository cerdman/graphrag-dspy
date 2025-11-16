# GraphRAG DSPy Conversion

This document describes the conversion of GraphRAG's prompt and model execution mechanisms to use [DSPy](https://dspy.ai/), a framework for programming language models.

## Overview

GraphRAG has been enhanced to use DSPy for prompt execution while maintaining backward compatibility with the existing prompt-based approach. DSPy provides:

- **Declarative Signatures**: Type-safe input/output definitions for LM tasks
- **Composable Modules**: Reusable components with automatic prompt optimization
- **Better Organization**: Structured approach to prompt engineering
- **Claude Support**: Native integration with Claude and other LLMs

## Architecture

### DSPy Integration Components

1. **DSPy LM Adapter** (`graphrag/language_model/providers/dspy/adapter.py`)
   - Bridges GraphRAG's `ChatModel` protocol with DSPy's `LM` interface
   - Enables GraphRAG's existing LM implementations to work with DSPy modules
   - Handles async/sync conversion automatically

2. **DSPy Modules** (`graphrag/dspy_modules/`)
   - `index.py`: Modules for index operations (Graph Extraction, Community Reports, etc.)
   - `query.py`: Modules for query operations (Local Search, Global Search, etc.)
   - `claude_support.py`: Helper utilities for Claude model integration

3. **Updated Extractors**
   - Graph Extractor: Uses `GraphExtractionModule` (DSPy) or legacy prompts
   - Community Reports Extractor: Uses `CommunityReportModule` (DSPy) or legacy prompts
   - Both support `use_dspy=True/False` flag for gradual migration

## Usage

### Enabling DSPy (Default)

By default, DSPy is enabled for all extractors:

```python
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor

extractor = GraphExtractor(model_invoker=chat_model)  # Uses DSPy by default
```

### Using Legacy Prompts

To use the original prompt-based approach:

```python
extractor = GraphExtractor(model_invoker=chat_model, use_dspy=False)
```

### Claude Support

To use Claude models with GraphRAG:

```python
import os
import dspy
from graphrag.dspy_modules.claude_support import configure_claude

# Set your API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Configure Claude
lm = configure_claude(model="anthropic/claude-sonnet-4-20250514")

# Now all DSPy operations will use Claude
# GraphRAG extractors will automatically use Claude when use_dspy=True
```

### Available Claude Models

- `anthropic/claude-sonnet-4-20250514` (recommended)
- `anthropic/claude-3-7-sonnet-20250219`
- `anthropic/claude-3-5-haiku-20241022`
- `anthropic/claude-3-opus-20240229`

## DSPy Modules Reference

### Index Modules

#### GraphExtractionModule
- **Inputs**: entity_types, input_text, delimiters
- **Output**: extracted_data (entities and relationships)
- **Features**: Iterative refinement with gleanings

#### CommunityReportModule
- **Inputs**: input_text, max_report_length
- **Output**: Structured report (title, summary, findings, rating)
- **Features**: ChainOfThought reasoning for comprehensive reports

#### ClaimExtractionModule
- **Inputs**: input_text, entity_specs
- **Output**: extracted_claims
- **Features**: Focused claim extraction

#### DescriptionSummaryModule
- **Inputs**: descriptions (newline-separated)
- **Output**: summary
- **Features**: Concise summarization

### Query Modules

#### LocalSearchModule
- **Inputs**: context_data, response_type, question
- **Output**: answer with data references

#### GlobalSearchMapModule/GlobalSearchReduceModule
- **Map**: Analyze individual community reports
- **Reduce**: Synthesize analyses into final answer

#### DriftSearchModule
- **Inputs**: conversation_history, current_question, context_data
- **Output**: Context-aware answer handling conversation drift

#### QuestionGenModule
- **Inputs**: context, num_questions
- **Output**: Generated questions

## Migration Status

### Completed ✅
- DSPy dependency added to `pyproject.toml`
- GraphRAGDSpyLM adapter created
- All DSPy signatures and modules implemented
- Graph Extraction converted to DSPy
- Community Reports converted to DSPy
- Claude support added

### In Progress 🚧
- Converting remaining extractors (Claims, Descriptions)
- Query operations integration
- Test suite validation

### Future Enhancements 🔮
- DSPy prompt optimization (BootstrapFewShot, MIPRO)
- Fine-tuning signatures based on GraphRAG datasets
- Performance benchmarking (DSPy vs legacy)
- Additional LLM provider integrations

## Benefits of DSPy Conversion

1. **Type Safety**: Input/output schemas prevent runtime errors
2. **Composability**: Modules can be combined and reused
3. **Optimization**: DSPy can automatically improve prompts
4. **Maintainability**: Centralized prompt logic in signatures
5. **Flexibility**: Easy to swap LLM providers (OpenAI, Claude, etc.)
6. **Better Testing**: Mock DSPy modules for unit tests

## Backward Compatibility

All existing GraphRAG code continues to work without modification. The `use_dspy` flag allows gradual migration:

- Set `use_dspy=True` (default) for new DSPy-based execution
- Set `use_dspy=False` to use original prompt-based approach
- Both approaches produce equivalent results

## Contributing

When adding new prompt-based operations to GraphRAG:

1. Create a DSPy Signature defining inputs/outputs
2. Create a DSPy Module using the signature
3. Add to `graphrag/dspy_modules/index.py` or `query.py`
4. Update the corresponding extractor with `use_dspy` flag
5. Test both DSPy and legacy modes

## References

- DSPy Documentation: https://dspy.ai/learn/
- DSPy API Reference: https://dspy.ai/api/
- DSPy GitHub: https://github.com/stanfordnlp/dspy
- GraphRAG Documentation: https://microsoft.github.io/graphrag/
