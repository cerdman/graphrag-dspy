# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Pytest configuration for DSPy module tests."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
