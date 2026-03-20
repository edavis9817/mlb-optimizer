"""
run_pipeline.py
---------------
CLI entry point for the MLB optimizer pipeline.

Usage
-----
  python run_pipeline.py                           # uses configs/default_config.json
  python run_pipeline.py configs/my_config.json   # custom config
"""

import sys
import os

# Ensure src/ is importable when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline import run_pipeline

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default_config.json"
    run_dir = run_pipeline(config_path)
    print(f"\nRun complete: {run_dir}")
