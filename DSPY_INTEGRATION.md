# DSPy Integration for GraphRAG 🚀

This document describes the DSPy integration for GraphRAG, enabling programmatic prompt engineering and support for multiple LLM providers including Claude.

## What is DSPy?

DSPy is a framework for programming—not prompting—language models. Instead of manually crafting prompts, you define signatures (input/output interfaces) and let DSPy handle the prompt construction and optimization.

## Features

✅ **DSPy Provider**: New `dspy_chat` model type
✅ **Claude Support**: Native support for Anthropic Claude models
✅ **Modular Prompts**: Graph extraction and community reports as DSPy modules
✅ **Backward Compatible**: Traditional prompts still work
✅ **Optimizable**: Can use DSPy optimizers to improve prompts

## Installation

DSPy is included in the dependencies. Install with:

```bash
pip install -e .
```

## Configuration

### Using Claude with DSPy

To use Claude models with DSPy, configure your GraphRAG settings:

**settings.yaml:**
```yaml
models:
  chat:
    type: dspy_chat
    model_provider: anthropic
    model: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}
    max_tokens: 4096
    temperature: 0.0
```

**Environment Variables:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Using OpenAI with DSPy

```yaml
models:
  chat:
    type: dspy_chat
    model_provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    max_tokens: 4096
    temperature: 0.0
```

### Using Azure OpenAI with DSPy

```yaml
models:
  chat:
    type: dspy_chat
    model_provider: azure
    deployment_name: your-deployment
    api_base: https://your-resource.openai.azure.com
    api_version: 2024-02-15-preview
    api_key: ${AZURE_OPENAI_API_KEY}
    max_tokens: 4096
    temperature: 0.0
```

## DSPy Modules

### Graph Extraction

The graph extraction prompt has been converted to a DSPy module:

**Location**: `graphrag/dspy_modules/extract_graph.py`

**Usage**:
```python
from graphrag.dspy_modules import DSPyGraphExtractor

extractor = DSPyGraphExtractor(max_gleanings=1)
result = extractor(
    text="Your input text here",
    entity_types="PERSON,ORGANIZATION,GEO"
)
```

**Features**:
- Structured input/output signatures
- Multi-turn "gleanings" for comprehensive extraction
- Compatible with existing graph processing pipeline

### Community Reports

Community report generation using DSPy:

**Location**: `graphrag/dspy_modules/community_reports.py`

**Usage**:
```python
from graphrag.dspy_modules import DSPyCommunityReportGenerator

generator = DSPyCommunityReportGenerator()
report = generator(
    input_text="Entity and relationship data",
    max_report_length=1500
)
```

**Output Format**:
- JSON-structured reports
- Impact severity ratings
- Detailed findings with data references

## Architecture

### DSPy Provider Layer

```
graphrag/language_model/providers/dspy/
├── __init__.py
└── chat_model.py          # DSPyChatModel implementing ChatModel protocol
```

**Key Features**:
- Implements `ChatModel` protocol for seamless integration
- Supports multiple providers (Claude, OpenAI, Azure)
- Async and streaming support (via thread pool)
- Conversation history management

### DSPy Modules Layer

```
graphrag/dspy_modules/
├── __init__.py
├── extract_graph.py        # Graph extraction signatures
└── community_reports.py    # Community report generation
```

## Migration Guide

### From Traditional to DSPy

**Before** (Traditional):
```yaml
models:
  chat:
    type: chat  # or openai_chat
    model: gpt-4
```

**After** (DSPy):
```yaml
models:
  chat:
    type: dspy_chat
    model_provider: openai
    model: gpt-4
```

### Backward Compatibility

The traditional model types still work:
- `openai_chat` - OpenAI via FNLLM
- `azure_openai_chat` - Azure OpenAI via FNLLM
- `chat` - Generic via LiteLLM

Choose `dspy_chat` to use DSPy's programmatic approach.

## Benefits of DSPy

1. **Type Safety**: Signatures enforce clear input/output contracts
2. **Modularity**: Compose complex workflows from simple components
3. **Optimization**: Can auto-tune prompts with DSPy optimizers
4. **Testing**: Easier to unit test modular components
5. **Portability**: Switch between models easily

## Optimization (Advanced)

DSPy supports automatic prompt optimization. Future versions may include:

```python
import dspy
from dspy.teleprompt import MIPROv2

# Define optimizer
optimizer = MIPROv2(
    metric=your_metric_function,
    num_candidates=10
)

# Optimize module
optimized_extractor = optimizer.compile(
    extractor,
    trainset=your_training_data
)
```

## Troubleshooting

### Import Errors

Ensure DSPy is installed:
```bash
pip install dspy>=2.6.0
```

### API Key Issues

Check environment variables:
```bash
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
```

### Model Provider Errors

Verify model provider name matches:
- `anthropic` for Claude
- `openai` for OpenAI
- `azure` for Azure OpenAI

## Examples

### Complete Claude Configuration

```yaml
# settings.yaml
models:
  chat:
    type: dspy_chat
    model_provider: anthropic
    model: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}
    max_tokens: 8192
    temperature: 0.0

  embedding:
    type: embedding
    model_provider: openai
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
```

### Programmatic Usage

```python
from graphrag.language_model.factory import ModelFactory
from graphrag.config.models.language_model_config import LanguageModelConfig

# Create DSPy model
config = LanguageModelConfig(
    type="dspy_chat",
    model_provider="anthropic",
    model="claude-sonnet-4",
    api_key="your-key",
    max_tokens=4096,
    temperature=0.0
)

model = ModelFactory.create_chat_model(
    model_type="dspy_chat",
    name="my_model",
    config=config
)

# Use like any ChatModel
response = await model.achat("Hello, Claude!")
print(response.output.content)
```

## Contributing

To add new DSPy modules:

1. Create signature in `graphrag/dspy_modules/your_module.py`
2. Define DSPy module class extending `dspy.Module`
3. Implement `forward()` method
4. Add to `__all__` in `__init__.py`

## References

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [Anthropic Claude API](https://docs.anthropic.com/)

## License

Copyright (c) 2025 Microsoft Corporation.
Licensed under the MIT License.
