#!/usr/bin/env python3
"""Shared AWA2/CUB/SUN class relation diagnostic; the AWA2 entry remains usable."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_awa2_relations import main


if __name__ == "__main__":
    main()
