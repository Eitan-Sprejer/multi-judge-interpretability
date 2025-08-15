#!/usr/bin/env python3
"""
Core logic for creating multi-LLM rubric judges.
Split out to follow the experiments standard structure.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def import_rubrics():
    import importlib.util
    rubrics_path = ROOT / "pipeline" / "utils" / "judge_rubrics.py"
    spec = importlib.util.spec_from_file_location("judge_rubrics", str(rubrics_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {rubrics_path}")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def get_logger(log_file: str = "judge_creation.log") -> logging.Logger:
    logger = logging.getLogger("exp3b")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def create_all_judges(cfg: dict) -> list[str]:
    from martian_apart_hack_sdk import judge_specs, martian_client
    from martian_apart_hack_sdk.models import llm_models

    api_key = os.getenv("MARTIAN_API_KEY")
    if not api_key:
        raise RuntimeError("MARTIAN_API_KEY is not set")

    jr = import_rubrics()
    logger = get_logger(cfg.get("logging", {}).get("file", "judge_creation.log"))
    client = martian_client.MartianClient(api_url=cfg["api"]["base_url"], api_key=api_key)

    # Map names to enums if available
    name_to_enum = {
        "gpt-4o-mini": getattr(llm_models, "GPT_4O_MINI", "gpt-4o-mini"),
        "claude-3-5-sonnet": getattr(llm_models, "CLAUDE_3_5_SONNET", "claude-3-5-sonnet"),
        "gemini-1.5-flash": getattr(llm_models, "GEMINI_1_5_FLASH", "gemini-1.5-flash"),
        "llama-3.1-70b": getattr(llm_models, "LLAMA_3_1_405B", "llama-3.1-70b"),
        "llama-3.1-8b": getattr(llm_models, "LLAMA_3_1_8B", "llama-3.1-8b"),
    }

    def mk_spec(rubric_text: str, model_key: str):
        model = name_to_enum.get(model_key, model_key)
        return judge_specs.RubricJudgeSpec(
            model_type="rubric_judge",
            rubric=rubric_text,
            model=model,
            min_score=cfg["judge_config"]["min_score"],
            max_score=cfg["judge_config"]["max_score"],
        )

    created: list[str] = []
    for pkey, pconf in cfg["llm_providers"].items():
        logger.info(f"Creating judges for provider {pkey} -> {pconf['model']}")
        for rkey, rfunc in jr.JUDGE_RUBRICS.items():
            judge_id = f"{pkey}-{rkey}"
            try:
                spec = mk_spec(rfunc(), pconf["model"])
                desc = f"{pconf['description']} for {jr.JUDGE_DESCRIPTIONS.get(rkey, rkey)}"
                try:
                    client.judges.create_judge(judge_id=judge_id, judge_spec=spec, description=desc)
                    logger.info(f"Created {judge_id}")
                except Exception:
                    client.judges.update_judge(judge_id=judge_id, judge_spec=spec)
                    logger.info(f"Updated {judge_id}")
                created.append(judge_id)
            except Exception as e:
                logger.error(f"Failed {judge_id}: {e}")

    return created


def main(config_path: Optional[Path] = None):
    import yaml
    from dotenv import load_dotenv
    load_dotenv()
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    created = create_all_judges(cfg)
    return created
