#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "07_train_generate_timevae_paperstyle.ipynb"


def parse_json_value(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}

    candidate = Path(raw)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))

    return json.loads(raw)


def load_first_code_cell(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            return source
    raise RuntimeError(f"No executable code cell found in {notebook_path}")


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    config.update(overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run notebook 07 TimeVAE training+generation with CLI config overrides.")
    parser.add_argument("--project-root", default=str(REPO_ROOT), help="Project root used in TIMEVAE_CONFIG.")
    parser.add_argument(
        "--timevae-overrides",
        default="{}",
        help="JSON string or path for TIMEVAE_CONFIG overrides.",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional path to save JSON summary output.",
    )
    args = parser.parse_args()

    source = load_first_code_cell(NOTEBOOK_PATH)
    namespace: Dict[str, Any] = {"__name__": "nb_07_train_generate_timevae_paperstyle"}
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    config = namespace["TIMEVAE_CONFIG"]
    config["project_root"] = str(Path(args.project_root).resolve())
    apply_overrides(config, parse_json_value(args.timevae_overrides))

    print("Running TimeVAE notebook pipeline (07)...")
    outputs = namespace["main"](config)
    print("TimeVAE pipeline finished.")

    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(outputs, indent=2, default=str), encoding="utf-8")
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
