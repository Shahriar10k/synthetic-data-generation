#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def parse_json_value(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}

    candidate = Path(raw)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
        return json.loads(text)

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


def exec_notebook_code(notebook_path: Path, source_override: str | None = None) -> Dict[str, Any]:
    source = source_override if source_override is not None else load_first_code_cell(notebook_path)
    namespace: Dict[str, Any] = {"__name__": f"nb_{notebook_path.stem}"}
    exec(compile(source, str(notebook_path), "exec"), namespace)
    return namespace


def apply_overrides(config_obj: Any, overrides: Dict[str, Any]) -> None:
    if isinstance(config_obj, dict):
        config_obj.update(overrides)
        return

    for key, value in overrides.items():
        if not hasattr(config_obj, key):
            raise AttributeError(f"Unknown config key '{key}' for {type(config_obj).__name__}")
        setattr(config_obj, key, value)


def run_stage_with_config(
    notebook_name: str,
    config_name: str,
    function_name: str,
    project_root: Path,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    namespace = exec_notebook_code(NOTEBOOKS_DIR / notebook_name)
    config_obj = namespace[config_name]

    if isinstance(config_obj, dict):
        config_obj["project_root"] = str(project_root)
    elif hasattr(config_obj, "project_root"):
        setattr(config_obj, "project_root", str(project_root))

    apply_overrides(config_obj, overrides)
    return namespace[function_name](config_obj)


def python_expr(value: Any) -> str:
    return repr(value)


def patch_assignment(source: str, name: str, expr: str) -> str:
    pattern = rf"(?m)^\s*{re.escape(name)}\s*=.*$"
    replacement = f"{name} = {expr}"
    patched, count = re.subn(pattern, replacement, source, count=1)
    if count == 0:
        raise RuntimeError(f"Could not patch assignment for {name}")
    return patched


def run_stage_06(
    project_root: Path,
    selected_model_family: str,
    selected_synthetic_method: str,
    run_real_to_real_only: bool,
    use_real_models: bool,
    modalities: Iterable[str],
    stage_06_overrides: Dict[str, Any],
) -> Any:
    notebook_path = NOTEBOOKS_DIR / "06_downstream_cnnbilstm.ipynb"
    source = load_first_code_cell(notebook_path)

    source = patch_assignment(source, "PROJECT_ROOT", f"Path({python_expr(str(project_root))})")
    source = patch_assignment(source, "SELECTED_MODEL_FAMILY", python_expr(selected_model_family))
    source = patch_assignment(source, "SELECTED_SYNTHETIC_METHOD", python_expr(selected_synthetic_method))
    source = patch_assignment(source, "RUN_REAL_TO_REAL_ONLY", python_expr(run_real_to_real_only))
    source = patch_assignment(source, "USE_REAL_MODELS", python_expr(use_real_models))
    source = patch_assignment(source, "MODALITIES", python_expr(list(modalities)))

    for key, value in stage_06_overrides.items():
        source = patch_assignment(source, key, python_expr(value))

    namespace = exec_notebook_code(notebook_path, source_override=source)
    return namespace["main"]()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run notebooks 01-06 as a single configurable pipeline.",
    )
    parser.add_argument(
        "--project-root",
        default=str(REPO_ROOT),
        help="Project root used by notebook configs (default: repository root).",
    )
    parser.add_argument(
        "--steps",
        default="1,2,3,4,5,6",
        help="Comma-separated stage numbers to run (subset of 1,2,3,4,5,6).",
    )

    parser.add_argument("--preprocessing-overrides", default="{}", help="JSON string or path for notebook 01 CONFIG overrides.")
    parser.add_argument("--train-overrides", default="{}", help="JSON string or path for notebook 02 CONFIG overrides.")
    parser.add_argument("--generation-overrides", default="{}", help="JSON string or path for notebook 03 GEN_CONFIG overrides.")
    parser.add_argument("--comparison-overrides", default="{}", help="JSON string or path for notebook 04 COMPARISON_CONFIG overrides.")
    parser.add_argument("--realism-overrides", default="{}", help="JSON string or path for notebook 05 EVAL_CONFIG overrides.")

    parser.add_argument("--selected-model-family", default="kovae", help="Notebook 06 SELECTED_MODEL_FAMILY.")
    parser.add_argument("--selected-synthetic-method", default="rollout_v1", help="Notebook 06 SELECTED_SYNTHETIC_METHOD.")
    parser.add_argument("--run-real-to-real-only", action="store_true", help="Notebook 06 RUN_REAL_TO_REAL_ONLY=True.")
    parser.add_argument("--disable-use-real-models", action="store_true", help="Notebook 06 USE_REAL_MODELS=False.")
    parser.add_argument(
        "--modalities",
        default="acc,bvp,eda,temp,fused",
        help="Comma-separated notebook 06 modalities (default: acc,bvp,eda,temp,fused).",
    )
    parser.add_argument(
        "--downstream-overrides",
        default="{}",
        help="JSON string or path of extra assignment overrides for notebook 06 (e.g., EPOCHS_CNNBILSTM, BATCH_SIZE).",
    )

    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional path to save JSON summary of executed steps.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    steps = [int(token.strip()) for token in args.steps.split(",") if token.strip()]

    preprocessing_overrides = parse_json_value(args.preprocessing_overrides)
    train_overrides = parse_json_value(args.train_overrides)
    generation_overrides = parse_json_value(args.generation_overrides)
    comparison_overrides = parse_json_value(args.comparison_overrides)
    realism_overrides = parse_json_value(args.realism_overrides)
    downstream_overrides = parse_json_value(args.downstream_overrides)

    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    use_real_models = not args.disable_use_real_models

    summary: Dict[str, Any] = {"project_root": str(project_root), "steps": steps, "outputs": {}}

    if 1 in steps:
        print("[1/6] Running 01_preprocessing...")
        summary["outputs"]["1"] = run_stage_with_config(
            notebook_name="01_preprocessing.ipynb",
            config_name="CONFIG",
            function_name="run_preprocessing",
            project_root=project_root,
            overrides=preprocessing_overrides,
        )

    if 2 in steps:
        print("[2/6] Running 02_train_multibranch_kovae...")
        summary["outputs"]["2"] = run_stage_with_config(
            notebook_name="02_train_multibranch_kovae.ipynb",
            config_name="CONFIG",
            function_name="run_kovae_training",
            project_root=project_root,
            overrides=train_overrides,
        )

    if 3 in steps:
        print("[3/6] Running 03_generate_kovae_synthetic_subjects...")
        summary["outputs"]["3"] = run_stage_with_config(
            notebook_name="03_generate_kovae_synthetic_subjects.ipynb",
            config_name="GEN_CONFIG",
            function_name="run_generation",
            project_root=project_root,
            overrides=generation_overrides,
        )

    if 4 in steps:
        print("[4/6] Running 04_comparison...")
        summary["outputs"]["4"] = run_stage_with_config(
            notebook_name="04_comparison.ipynb",
            config_name="COMPARISON_CONFIG",
            function_name="main",
            project_root=project_root,
            overrides=comparison_overrides,
        )

    if 5 in steps:
        print("[5/6] Running 05_realism_diversity...")
        summary["outputs"]["5"] = run_stage_with_config(
            notebook_name="05_realism_diversity.ipynb",
            config_name="EVAL_CONFIG",
            function_name="main",
            project_root=project_root,
            overrides=realism_overrides,
        )

    if 6 in steps:
        print("[6/6] Running 06_downstream_cnnbilstm...")
        summary["outputs"]["6"] = {
            "completed": True,
            "selected_model_family": args.selected_model_family,
            "selected_synthetic_method": args.selected_synthetic_method,
            "modalities": modalities,
            "run_real_to_real_only": args.run_real_to_real_only,
            "use_real_models": use_real_models,
        }
        run_stage_06(
            project_root=project_root,
            selected_model_family=args.selected_model_family,
            selected_synthetic_method=args.selected_synthetic_method,
            run_real_to_real_only=args.run_real_to_real_only,
            use_real_models=use_real_models,
            modalities=modalities,
            stage_06_overrides=downstream_overrides,
        )

    print("Pipeline finished.")

    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
