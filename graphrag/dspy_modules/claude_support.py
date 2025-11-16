# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Claude model support for GraphRAG via DSPy."""

import dspy


def configure_claude(
    model: str = "anthropic/claude-sonnet-4-20250514",
    api_key: str | None = None,
    **kwargs,
) -> dspy.LM:
    """
    Configure DSPy to use Claude models via Anthropic API.

    This function sets up a Claude language model for use with GraphRAG's
    DSPy-based prompt execution. It leverages DSPy's built-in support for
    Anthropic models through LiteLLM.

    Args:
        model: The Claude model identifier (default: claude-sonnet-4)
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        **kwargs: Additional parameters passed to dspy.LM

    Returns:
        Configured dspy.LM instance for Claude

    Example:
        >>> import os
        >>> os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
        >>> lm = configure_claude()
        >>> dspy.configure(lm=lm)
        >>> # Now GraphRAG will use Claude for all DSPy operations
    """
    # Create DSPy LM for Claude
    lm = dspy.LM(model=model, api_key=api_key, **kwargs)

    # Configure as default LM
    dspy.configure(lm=lm)

    return lm


def list_claude_models():
    """
    List available Claude models that can be used with DSPy.

    Returns:
        List of Claude model identifiers
    """
    return [
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-3-7-sonnet-20250219",
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-3-opus-20240229",
    ]


# Example usage for testing Claude integration
if __name__ == "__main__":
    import os

    if "ANTHROPIC_API_KEY" in os.environ:
        # Configure Claude
        lm = configure_claude()

        # Test with a simple prompt
        response = lm("Say 'Hello from Claude!'")
        print(f"Claude response: {response}")
    else:
        print("Set ANTHROPIC_API_KEY environment variable to test Claude integration")
        print(f"Available models: {list_claude_models()}")
