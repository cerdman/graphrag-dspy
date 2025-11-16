# DSPy 3.0 API Update

## Critical Fix: API Compatibility

The initial implementation was written for DSPy 2.x API but DSPy 3.0.4 is installed, causing incompatibility.

### Changes Made

#### 1. **Provider Initialization** (`chat_model.py:68-111`)

**Before (DSPy 2.x API):**
```python
# Separate classes for each provider
self._lm = dspy.Claude(model="claude-sonnet-4", api_key="...", ...)
self._lm = dspy.OpenAI(model="gpt-4", api_key="...", ...)
self._lm = dspy.AzureOpenAI(deployment_id="...", ...)
```

**After (DSPy 3.0+ API):**
```python
# Unified LM class with provider/model string format
self._lm = dspy.LM(model="anthropic/claude-sonnet-4", api_key="...", ...)
self._lm = dspy.LM(model="openai/gpt-4", api_key="...", ...)
self._lm = dspy.LM(model="azure/deployment", api_base="...", ...)
```

#### 2. **Dependency Version** (`pyproject.toml:42`)

**Before:**
```toml
"dspy>=2.6.0"  # Allowed 3.x which broke everything
```

**After:**
```toml
"dspy>=3.0.0,<4.0.0"  # Explicit 3.x pin, prevents future breaks
```

#### 3. **Test Updates** (`test_chat_model.py`)

- Mocks changed from `@patch("dspy.Claude")` to `@patch("dspy.LM")`
- Model strings verified to contain provider prefix: `"anthropic/claude-sonnet-4"`
- Added explicit `encoding_model="cl100k_base"` for Claude configs (tiktoken doesn't recognize Claude models)

### Test Results

**Before Fix:** 8/24 tests passing (provider tests failed with AttributeError)

**After Fix:** ✅ **24/24 tests passing**

```bash
$ python -m pytest tests/unit/dspy_modules/ tests/unit/language_model/providers/dspy_provider/ -v
======================== 24 passed, 2 warnings in 2.61s =========================
```

### API Differences Summary

| Feature | DSPy 2.x | DSPy 3.0 |
|---------|----------|----------|
| **LM Class** | `dspy.Claude`, `dspy.OpenAI`, `dspy.AzureOpenAI` | `dspy.LM` (unified) |
| **Model String** | `model="claude-sonnet-4"` | `model="anthropic/claude-sonnet-4"` |
| **Provider Param** | Implicit from class | Explicit in model string |
| **Configuration** | Class-specific kwargs | Unified kwargs |

### Usage Examples

#### Claude
```python
from graphrag.language_model.factory import ModelFactory
from graphrag.config.models.language_model_config import LanguageModelConfig

config = LanguageModelConfig(
    type='dspy_chat',
    model_provider='anthropic',
    model='claude-sonnet-4',
    api_key='sk-ant-...',
    encoding_model='cl100k_base'  # Required for tiktoken
)

model = ModelFactory.create_chat_model('dspy_chat', name='test', config=config)
response = model.chat('Hello, Claude!')
```

#### OpenAI
```python
config = LanguageModelConfig(
    type='dspy_chat',
    model_provider='openai',
    model='gpt-4',
    api_key='sk-...'
)
```

#### Azure OpenAI
```python
config = LanguageModelConfig(
    type='dspy_chat',
    model_provider='azure',
    deployment_name='my-gpt4-deployment',
    api_base='https://my-resource.openai.azure.com',
    api_version='2024-02-15-preview',
    api_key='...'
)
```

### Why This Matters

1. **Version Skew Prevention**: Pinning `dspy>=3.0.0,<4.0.0` prevents accidental upgrades that break the API
2. **Unified Interface**: DSPy 3.0's unified `LM` class simplifies provider management
3. **Future-Proof**: When DSPy 4.0 releases, we'll know to test before upgrading

### Files Modified

- ✅ `graphrag/language_model/providers/dspy/chat_model.py` - Unified LM API
- ✅ `tests/unit/language_model/providers/dspy_provider/test_chat_model.py` - Updated mocks
- ✅ `pyproject.toml` - Version constraint
- ✅ `tests/unit/language_model/providers/dspy_provider/` - Renamed from `dspy/` to avoid shadowing

### Known Issues Resolved

1. ❌ **Test directory shadowing** - `tests/.../dspy/` shadowed real dspy package
   - ✅ **Fixed**: Renamed to `dspy_provider/`

2. ❌ **AttributeError: module 'dspy' has no attribute 'Claude'**
   - ✅ **Fixed**: Use `dspy.LM(model="anthropic/...")`

3. ❌ **tiktoken KeyError for Claude models**
   - ✅ **Fixed**: Explicit `encoding_model` in test configs

---

**Status**: ✅ All tests passing, DSPy 3.0 fully compatible
