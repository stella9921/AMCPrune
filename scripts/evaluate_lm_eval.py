import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from amcprune.models import get_transformer_blocks, load_causal_lm
from amcprune.pruning import remove_transformer_blocks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a physically pruned AMCPrune model with lm-evaluation-harness."
    )
    parser.add_argument(
        "--pruning-config",
        required=True,
        help="Path to amcprune_pruning_config.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override base model. Defaults to base_model in pruning config.",
    )
    parser.add_argument("--tasks", default="hellaswag")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def load_pruning_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def save_results(path, payload):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)


def main():
    args = parse_args()
    config = load_pruning_config(args.pruning_config)
    base_model = args.model or config["base_model"]
    selected_blocks = config["selected_blocks"]

    model, tokenizer, _ = load_causal_lm(
        base_model,
        device=args.device,
        dtype=args.dtype,
    )
    _, detected_block_path = get_transformer_blocks(model)
    block_path = config.get("block_path", detected_block_path)
    remove_transformer_blocks(model, block_path, selected_blocks)
    model.eval()

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise ImportError(
            "lm-evaluation-harness is required. Install with: pip install lm_eval[hf]"
        ) from error

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    payload = {
        "base_model": base_model,
        "pruning_config": args.pruning_config,
        "selected_blocks": selected_blocks,
        "tasks": args.tasks,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "results": results,
    }
    save_results(args.output_path, payload)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[AMCPrune lm_eval] base_model={base_model}")
    print(f"[AMCPrune lm_eval] selected_blocks={selected_blocks}")
    print(f"[AMCPrune lm_eval] saved={args.output_path}")


if __name__ == "__main__":
    main()

