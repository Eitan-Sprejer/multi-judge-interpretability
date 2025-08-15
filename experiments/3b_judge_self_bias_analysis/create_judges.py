#!/usr/bin/env python3
"""
Thin wrapper to keep backward compatibility.
Delegates to src/main_logic.py which follows the standard experiment layout.
"""
import argparse
from pathlib import Path

from src.main_logic import main as create_main


def cli():
    p = argparse.ArgumentParser(description="Create multi-LLM judges")
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "default_config.yaml")
    args = p.parse_args()
    created = create_main(args.config)
    print(f"Created/updated {len(created)} judges")


if __name__ == "__main__":
    cli()
