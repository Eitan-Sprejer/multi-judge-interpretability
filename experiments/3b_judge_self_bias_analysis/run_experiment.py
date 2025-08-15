#!/usr/bin/env python3
"""
Experiment 3b: Multi-LLM Judge Creation

Standard entry point following experiments/README structure.

Usage:
  python run_experiment.py [--config configs/default_config.yaml] [--quick]

--quick will only create a small subset of judges (1 provider x 2 rubrics) as a smoke test.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add repo root so we can import pipeline utils directly
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def load_config(path: Path) -> dict:
    import yaml
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_results_dirs(exp_dir: Path) -> Path:
    results_dir = exp_dir / "results"
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "data").mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    (results_dir / "reports").mkdir(parents=True, exist_ok=True)
    return results_dir


def timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_report(results_dir: Path, created_ids: list[str]):
    ts = timestamp()
    # Raw json
    (results_dir / f"{ts}_results.json").write_text(json.dumps({
        "created_judges": created_ids,
        "count": len(created_ids),
        "timestamp": ts,
    }, indent=2))
    # Simple report
    report = results_dir / "reports" / "experiment_report.md"
    lines = [
        "# Experiment 3b Report",
        "",
        f"Date: {ts}",
        f"Total judges created/updated: {len(created_ids)}",
        "",
        "## Judges",
    ]
    for jid in sorted(created_ids):
        lines.append(f"- {jid}")
    report.write_text("\n".join(lines))


def main():
    exp_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Run Experiment 3b: create multi-LLM judges")
    parser.add_argument("--config", type=Path, default=exp_dir / "configs" / "default_config.yaml")
    parser.add_argument("--quick", action="store_true", help="Create a tiny subset for a fast smoke test")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    api_key = os.getenv("MARTIAN_API_KEY")
    if not api_key:
        print("MARTIAN_API_KEY is not set. Export it and retry.")
        sys.exit(1)

    # Late import to avoid hard dependency when only inspecting
    from martian_apart_hack_sdk import judge_specs, martian_client

    # Import rubrics from repo
    import importlib.util
    rubrics_path = ROOT / "pipeline" / "utils" / "judge_rubrics.py"
    spec = importlib.util.spec_from_file_location("judge_rubrics", str(rubrics_path))
    if spec is None or spec.loader is None:
        print(f"Failed to load rubrics from {rubrics_path}")
        sys.exit(1)
    jr = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(jr)

    # Map config models to SDK enums when possible, else pass string
    from martian_apart_hack_sdk.models import llm_models
    MODEL_MAP = {
        "gpt-4o-mini": getattr(llm_models, "GPT_4O_MINI", "gpt-4o-mini"),
        "claude-3-5-sonnet": getattr(llm_models, "CLAUDE_3_5_SONNET", "claude-3-5-sonnet"),
        "gemini-1.5-flash": getattr(llm_models, "GEMINI_1_5_FLASH", "gemini-1.5-flash"),
        "llama-3.1-70b": getattr(llm_models, "LLAMA_3_1_405B", "llama-3.1-70b"),
        "llama-3.1-70b": getattr(llm_models, "LLAMA_3_1_70B", "llama-3.1-70b"),
        "llama-3.1-8b": getattr(llm_models, "LLAMA_3_1_8B", "llama-3.1-8b"),
    }

    def create_spec(rubric_text: str, model_key: str, min_s: float, max_s: float):
        model = MODEL_MAP.get(model_key, model_key)
        return judge_specs.RubricJudgeSpec(
            model_type="rubric_judge",
            rubric=rubric_text,
            model=model,
            min_score=min_s,
            max_score=max_s,
        )

    client = martian_client.MartianClient(api_url=cfg["api"]["base_url"], api_key=api_key)

    # Determine providers and rubrics
    providers = list(cfg["llm_providers"].items())
    rubric_keys = list(jr.JUDGE_RUBRICS.keys())
    if args.quick:
        providers = providers[:1]
        rubric_keys = rubric_keys[:2]

    created = []
    for provider_key, pconf in providers:
        for rkey in rubric_keys:
            rubric_text = jr.JUDGE_RUBRICS[rkey]()
            spec_obj = create_spec(
                rubric_text,
                pconf["model"],
                cfg["judge_config"]["min_score"],
                cfg["judge_config"]["max_score"],
            )
            judge_id = f"{provider_key}-{rkey}"
            try:
                client.judges.create_judge(judge_id=judge_id, judge_spec=spec_obj, description=f"{pconf['description']} for {jr.JUDGE_DESCRIPTIONS.get(rkey, rkey)}")
                print(f"Created {judge_id}")
                created.append(judge_id)
            except Exception:
                # Try update if exists
                try:
                    client.judges.update_judge(judge_id=judge_id, judge_spec=spec_obj)
                    print(f"Updated {judge_id}")
                    created.append(judge_id)
                except Exception as e:
                    print(f"Failed {judge_id}: {e}")

    results_dir = ensure_results_dirs(exp_dir)
    write_report(results_dir, created)
    print(f"Done. Wrote report and results to {results_dir}")


if __name__ == "__main__":
    main()
