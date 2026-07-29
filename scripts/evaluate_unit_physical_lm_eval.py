import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from amcprune.compensation import apply_boundary_affine_compensation_from_metadata
from amcprune.models import get_transformer_blocks, load_causal_lm
from amcprune.pruning import remove_transformer_blocks
from amcprune.unit_pruning import apply_unit_physical_pruning


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an AMCPrune unit-physical model with lm-evaluation-harness."
    )
    parser.add_argument(
        "--pruned-model-dir",
        required=True,
        help="Directory containing amcprune_pruning_config.json and unit_physical_state_dict.pt.",
    )
    parser.add_argument("--model", default=None, help="Override base model from pruning config.")
    parser.add_argument("--tasks", default="hellaswag,piqa,arc_easy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--compare-dense",
        action="store_true",
        help="Also evaluate the dense base model under the same lm-eval setting.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path, payload):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)


def make_lm(model, tokenizer, batch_size):
    from lm_eval.models.huggingface import HFLM

    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=batch_size,
    )


def run_lm_eval(model, tokenizer, tasks, batch_size, limit, num_fewshot):
    import lm_eval

    lm = make_lm(model, tokenizer, batch_size)
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )


def extract_metric_values(results):
    summary = {}
    for task, metrics in results.get("results", {}).items():
        summary[task] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary[task][key] = value
    return summary


def compare_metrics(dense_summary, pruned_summary):
    comparison = {}
    for task, dense_metrics in dense_summary.items():
        comparison[task] = {}
        pruned_metrics = pruned_summary.get(task, {})
        for metric, dense_value in dense_metrics.items():
            if metric not in pruned_metrics:
                continue
            pruned_value = pruned_metrics[metric]
            comparison[task][metric] = {
                "dense": dense_value,
                "pruned": pruned_value,
                "delta": pruned_value - dense_value,
            }
    return comparison


def load_unit_physical_model(pruned_model_dir, base_model, device, dtype):
    config_path = os.path.join(pruned_model_dir, "amcprune_pruning_config.json")
    state_path = os.path.join(pruned_model_dir, "unit_physical_state_dict.pt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing pruning config: {config_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing unit physical state_dict: {state_path}")

    pruning_config = load_json(config_path)
    model_name = base_model or pruning_config["base_model"]
    model, tokenizer, _ = load_causal_lm(model_name, device=device, dtype=dtype)
    blocks, detected_block_path = get_transformer_blocks(model)
    block_path = pruning_config.get("block_path", detected_block_path)
    if block_path != detected_block_path:
        print(
            "[AMCPrune lm_eval] warning: config block_path "
            f"{block_path} != detected {detected_block_path}; using detected blocks"
        )
    unit_plan = pruning_config.get("unit_objective_plan")
    if not unit_plan:
        raise ValueError("unit_physical evaluation requires unit_objective_plan in pruning config.")

    physical_pruning = apply_unit_physical_pruning(blocks, unit_plan)
    if pruning_config.get("pruning_mode") == "depth_width_physical":
        depth_blocks = pruning_config.get("depth_pruned_blocks") or pruning_config.get("selected_blocks") or []
        depth_pruning = remove_transformer_blocks(model, detected_block_path, depth_blocks)
        physical_pruning.update({
            "pruning_mode": "depth_width_physical",
            "depth_pruning": depth_pruning,
            "block_path": detected_block_path,
            "original_num_blocks": depth_pruning["original_num_blocks"],
            "pruned_num_blocks": depth_pruning["pruned_num_blocks"],
            "remaining_num_blocks": depth_pruning["remaining_num_blocks"],
            "removed_original_indices": depth_pruning["removed_original_indices"],
            "kept_original_indices": depth_pruning["kept_original_indices"],
        })
        compensation = pruning_config.get("boundary_compensation")
        if compensation and compensation.get("enabled"):
            physical_pruning["boundary_compensation"] = apply_boundary_affine_compensation_from_metadata(
                model,
                detected_block_path,
                depth_pruning["kept_original_indices"],
                compensation,
            )
    state = torch.load(state_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, tokenizer, pruning_config, physical_pruning, missing, unexpected


def main():
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    try:
        import lm_eval  # noqa: F401
        from lm_eval.models.huggingface import HFLM  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "lm-evaluation-harness is required. Install with: pip install lm_eval[hf]"
        ) from error

    pruned_model, tokenizer, pruning_config, physical_pruning, missing, unexpected = load_unit_physical_model(
        args.pruned_model_dir,
        args.model,
        args.device,
        args.dtype,
    )
    base_model = args.model or pruning_config["base_model"]
    print(f"[AMCPrune unit lm_eval] base_model={base_model}")
    print(f"[AMCPrune unit lm_eval] pruned_model_dir={args.pruned_model_dir}")
    print(f"[AMCPrune unit lm_eval] physical_pruning={physical_pruning}")
    if missing:
        print(f"[AMCPrune unit lm_eval] missing_keys={len(missing)}")
    if unexpected:
        print(f"[AMCPrune unit lm_eval] unexpected_keys={len(unexpected)}")

    pruned_results = run_lm_eval(
        pruned_model,
        tokenizer,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
    )
    pruned_summary = extract_metric_values(pruned_results)

    dense_results = None
    dense_summary = None
    comparison = None
    if args.compare_dense:
        dense_model, dense_tokenizer, _ = load_causal_lm(
            base_model,
            device=args.device,
            dtype=args.dtype,
        )
        dense_model.eval()
        dense_results = run_lm_eval(
            dense_model,
            dense_tokenizer,
            tasks=tasks,
            batch_size=args.batch_size,
            limit=args.limit,
            num_fewshot=args.num_fewshot,
        )
        dense_summary = extract_metric_values(dense_results)
        comparison = compare_metrics(dense_summary, pruned_summary)

    payload = {
        "base_model": base_model,
        "pruned_model_dir": args.pruned_model_dir,
        "tasks": tasks,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "physical_pruning_reapplied": physical_pruning,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "pruned_summary": pruned_summary,
        "dense_summary": dense_summary,
        "comparison": comparison,
        "pruned_results": pruned_results,
        "dense_results": dense_results,
    }
    save_json(args.output_path, payload)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[AMCPrune unit lm_eval] saved={args.output_path}")


if __name__ == "__main__":
    main()
