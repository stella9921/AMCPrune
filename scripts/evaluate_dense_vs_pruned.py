import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from amcprune.models import get_transformer_blocks, load_causal_lm
from amcprune.pruning import apply_block_skip


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare dense and AMCPrune block-skip models with lm-evaluation-harness."
    )
    parser.add_argument("--pruning-config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tasks", default="hellaswag")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--output-path", required=True)
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


def main():
    args = parse_args()
    pruning_config = load_json(args.pruning_config)
    base_model = args.model or pruning_config["base_model"]
    selected_blocks = pruning_config["selected_blocks"]
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    try:
        import lm_eval  # noqa: F401
        from lm_eval.models.huggingface import HFLM  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "lm-evaluation-harness is required. Install with: pip install lm_eval[hf]"
        ) from error

    dense_model, tokenizer, _ = load_causal_lm(
        base_model,
        device=args.device,
        dtype=args.dtype,
    )
    dense_model.eval()
    print(f"[AMCPrune benchmark] dense model={base_model}")
    dense_results = run_lm_eval(
        dense_model,
        tokenizer,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
    )

    pruned_model, pruned_tokenizer, _ = load_causal_lm(
        base_model,
        device=args.device,
        dtype=args.dtype,
    )
    _, detected_block_path = get_transformer_blocks(pruned_model)
    block_path = pruning_config.get("block_path", detected_block_path)
    apply_block_skip(pruned_model, block_path, selected_blocks)
    pruned_model.eval()
    print(f"[AMCPrune benchmark] pruned selected_blocks={selected_blocks}")
    pruned_results = run_lm_eval(
        pruned_model,
        pruned_tokenizer,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
    )

    dense_summary = extract_metric_values(dense_results)
    pruned_summary = extract_metric_values(pruned_results)
    payload = {
        "base_model": base_model,
        "pruning_config": args.pruning_config,
        "selected_blocks": selected_blocks,
        "tasks": tasks,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "dense_summary": dense_summary,
        "pruned_summary": pruned_summary,
        "comparison": compare_metrics(dense_summary, pruned_summary),
        "dense_results": dense_results,
        "pruned_results": pruned_results,
    }
    save_json(args.output_path, payload)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[AMCPrune benchmark] saved={args.output_path}")


if __name__ == "__main__":
    main()
