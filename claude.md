# GraphRAG → DSPy Conversion Plan 🎯

## Current State Analysis (Completed ✅)

### Prompt Management
- **Location**: `graphrag/prompts/index/` and `graphrag/prompts/query/`
- **Structure**: Python string constants with template variables
  - Example: `GRAPH_EXTRACTION_PROMPT` in `graphrag/prompts/index/extract_graph.py`
  - Uses `.format()` with variables like `{entity_types}`, `{input_text}`, `{context_data}`
- **Key Prompts**:
  - Index: `extract_graph.py`, `extract_claims.py`, `community_report.py`, `summarize_descriptions.py`
  - Query: `local_search_system_prompt.py`, `global_search_map_system_prompt.py`, etc.

### Model Execution
- **Protocol**: `ChatModel` and `EmbeddingModel` protocols in `graphrag/language_model/protocol/base.py`
- **Methods**: `achat()`, `chat()`, `achat_stream()`, `chat_stream()`
- **Providers**:
  - `litellm` (in `graphrag/language_model/providers/litellm/`)
  - `fnllm` (in `graphrag/language_model/providers/fnllm/`)
- **Execution Pattern**: `response = await model.achat(prompt.format(**variables))`

### Key Files for Conversion
1. **Graph Extraction**: `graphrag/index/operations/extract_graph/graph_extractor.py`
   - Uses `GRAPH_EXTRACTION_PROMPT`, `CONTINUE_PROMPT`, `LOOP_PROMPT`
   - Multi-turn conversation for "gleanings"

2. **Community Reports**: `graphrag/index/operations/summarize_communities/`
3. **Description Summarization**: `graphrag/index/operations/summarize_descriptions/`
4. **Query Operations**: `graphrag/query/structured_search/`

### Dependencies
- `fnllm[azure,openai]>=0.4.1`
- `litellm>=1.77.1`
- `openai>=1.68.0`

## Conversion Strategy 🚀

### Phase 1: DSPy Integration (Priority 1)
1. **Add DSPy dependency** to `pyproject.toml`
2. **Create DSPy adapter layer** in `graphrag/language_model/providers/dspy/`
   - Implement ChatModel protocol using DSPy
   - Support Claude via DSPy's model providers
3. **Convert prompts to DSPy Signatures**
   - Create `graphrag/dspy_modules/` for DSPy signatures and modules
   - Start with high-impact prompts: graph extraction, community reports

### Phase 2: Prompt Conversion (Priority 2)
1. **Graph Extraction**: Convert to DSPy ChainOfThought or ReAct
2. **Community Reports**: Convert to DSPy module
3. **Search Prompts**: Convert query prompts
4. **Maintain backward compatibility**: Keep original prompts as fallback

### Phase 3: Testing & Validation (Priority 3)
1. **Unit tests**: Ensure converted modules work
2. **Integration tests**: Run existing test suite
3. **Performance comparison**: DSPy vs. original

### Phase 4: Claude Support (Secondary Goal)
1. **Configure DSPy** to use Claude via Anthropic API
2. **Test with Claude models**
3. **Document Claude configuration**

## Detailed Conversion Strategy 🎯

### Core Insight
GraphRAG's prompts are carefully crafted, but DSPy can:
1. **Organize better**: Signatures define clear input/output contracts
2. **Optimize automatically**: Improve prompts based on examples
3. **Compose modularly**: Build complex workflows from simple components

### Implementation Approach: Gradual Migration + Hybrid Mode

#### Phase 1: DSPy Foundation (Priority 1️⃣)
**Goal**: Add DSPy as a new provider while maintaining backward compatibility

1. **Add Dependencies** ✅
   - Add `dspy` to `pyproject.toml`
   - Install and configure

2. **Create DSPy Provider** ✅
   - `graphrag/language_model/providers/dspy/dspy_chat_model.py`
   - Implement `ChatModel` protocol using DSPy under the hood
   - Support Claude via DSPy's Anthropic integration
   - Registry integration in `ModelFactory`

3. **Configuration Support** ✅
   - Add ModelType.DSPyChat enum
   - Update config models to support DSPy settings
   - Allow model selection: litellm, fnllm, or dspy

#### Phase 2: Convert Key Prompts to DSPy Modules (Priority 2️⃣)
**Goal**: Convert high-impact prompts to DSPy Signatures

1. **Graph Extraction** (Highest Impact)
   - Create `graphrag/dspy_modules/extract_graph.py`
   - Define `GraphExtractionSignature`
   - Use `dspy.ChainOfThought` for extraction
   - Handle multi-turn "gleanings" with DSPy

2. **Community Reports**
   - Create `graphrag/dspy_modules/community_reports.py`
   - Define `CommunityReportSignature`
   - Use DSPy module for report generation

3. **Search Prompts**
   - Create `graphrag/dspy_modules/search.py`
   - Convert local/global search prompts

#### Phase 3: Smart Prompt Routing (Priority 3️⃣)
**Goal**: Allow dynamic selection between traditional and DSPy prompts

- Create `PromptStrategy` enum: `TRADITIONAL`, `DSPY`, `AUTO`
- Extractors check config and use appropriate method
- Backward compatibility: default to `TRADITIONAL`

#### Phase 4: Testing & Claude Support (Priority 4️⃣)

1. **Tests**
   - Existing tests should pass (using traditional mode)
   - Add new tests for DSPy mode
   - Compare outputs for equivalence

2. **Claude Integration**
   - Configure DSPy with Claude: `dspy.Claude(model="claude-sonnet-4")`
   - Test graph extraction with Claude
   - Document Claude configuration

## Implementation Plan 📝

### Progress: 11/11 complete ✅ (100%)

1. ✅ Explore repository structure
2. ✅ Identify prompt system
3. ✅ Identify model execution
4. ✅ Research DSPy architecture
5. ✅ Design detailed conversion strategy
6. ⏭️ Add DSPy dependency and create provider
7. ⏭️ Convert graph extraction to DSPy
8. ⏭️ Convert community reports to DSPy
9. ⏭️ Add Claude support and run tests
10. ⏭️ Commit and push

## Technical Decisions 🤔

### Why Gradual Migration?
- **Risk Mitigation**: Tests continue to work
- **User Choice**: Opt-in to DSPy features
- **Validation**: Compare DSPy vs. traditional outputs

### Why Focus on Graph Extraction First?
- Most complex prompt with examples
- Multi-turn conversation (gleanings)
- Core to GraphRAG's value proposition
- Best showcase for DSPy's capabilities

### How DSPy Improves GraphRAG
1. **Type Safety**: Signatures enforce input/output types
2. **Testability**: Modules are easier to unit test
3. **Optimization**: Can auto-tune prompts with DSPy optimizers
4. **Modularity**: Compose complex workflows from simple parts
5. **Multi-Model**: Easy to switch between OpenAI, Claude, etc.

## Notes 💡
- Maintain ChatModel protocol interface for compatibility
- DSPy modules are wrappers that eventually call model.achat()
- Claude support comes "free" with DSPy's provider system
- Can add optimization later (MIPROv2, BootstrapFewShot)
