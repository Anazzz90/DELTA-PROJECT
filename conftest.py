"""
conftest.py
===========
Adds the dmars project root to sys.path so pytest can import
modules like `core`, `agents`, `llm`, etc. directly without
needing to install the package.
"""
import sys
from pathlib import Path

# Insert the project root (dmars/) into sys.path
sys.path.insert(0, str(Path(__file__).parent))
