"""
Test configuration and common fixtures for fem3d.
"""

import sys
from pathlib import Path
import pytest

# Ensure fem3d is on sys.path
pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))
