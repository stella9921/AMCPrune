import argparse
import os

import torch

from amcprune.data import load_tokenized_text_dataset
from amcprune.evaluate import evaluate_perplexity
from amcprune.metrics import cuda_memory_mb, model_parameter_memory_mb, save_json
from amcprune.models import get_transformer_blocks, load_causal_lm
from amcprune.pruning import select_blocks, temporary_block_skip


def parse_args():
    parser = argparse.ArgumentParser(description="AMCPrune LLM prototype")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--pruning-ratio", type=float, default=0.25)
    parser.add_argument("--score", choices=["block_index", "early_block"], default="block_index")
    parser.add_argument("--output-dir", default="exp/smoke")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, device = load_causal_lm(args.model, dtype=args.dtype)
    dataset = load_tokenized_text_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
    )
    blocks, block_path = get_transformer_blocks(model)
    selected_blocks = select_blocks(
        num_blocks=len(blocks),
        pruning_ratio=args.pruning_ratio,
        score=args.score,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    baseline = evaluate_perplexity(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
    )
    with temporary_block_skip(model, blocks, block_path, selected_blocks):
        pruned = evaluate_perplexity(
            model,
            dataset,
            device=device,
            batch_size=args.batch_size,
        )

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "num_blocks": len(blocks),
        "block_path": block_path,
        "pruning_unit": "block_skip",
        "pruning_ratio": args.pruning_ratio,
        "score": args.score,
        "selected_blocks": selected_blocks,
        "parameter_memory_mb": model_parameter_memory_mb(model),
        "baseline": baseline,
        "pruned": pruned,
        "perplexity_delta": pruned["perplexity"] - baseline["perplexity"],
        **cuda_memory_mb(),
    }
    path = save_json(args.output_dir, "result.json", result)

    print(f"[AMCPrune] model={args.model}")
    print(f"[AMCPrune] blocks={len(blocks)} path={block_path}")
    print(f"[AMCPrune] selected_blocks={selected_blocks}")
    print(f"[AMCPrune] baseline_ppl={baseline['perplexity']:.4f}")
    print(f"[AMCPrune] pruned_ppl={pruned['perplexity']:.4f}")
    print(f"[AMCPrune] saved={path}")


if __name__ == "__main__":
    main()

