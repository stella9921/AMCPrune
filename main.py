import argparse
import os
from contextlib import contextmanager

import torch

from amcprune.data import load_tokenized_text_dataset
from amcprune.evaluate import evaluate_perplexity, evaluate_preservation
from amcprune.metrics import (
    MemoryTrace,
    cuda_memory_mb,
    model_parameter_memory_mb,
    reset_cuda_peak,
    save_json,
)
from amcprune.models import get_transformer_blocks, load_causal_lm
from amcprune.pruning import (
    apply_block_skip,
    select_blocks,
    select_blocks_from_ranking,
    temporary_block_skip,
)
from amcprune.scoring import (
    rank_blocks_by_scores,
    score_blocks_by_activation,
    score_blocks_by_loss_delta,
)


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
    parser.add_argument(
        "--score",
        choices=["block_index", "early_block", "activation", "loss_delta"],
        default="block_index",
    )
    parser.add_argument("--score-max-batches", type=int, default=8)
    parser.add_argument("--preservation-max-batches", type=int, default=8)
    parser.add_argument("--export-pruned-model", action="store_true")
    parser.add_argument("--output-dir", default="exp/smoke")
    return parser.parse_args()


def build_pruning_context(model, blocks, block_path, selected_blocks):
    @contextmanager
    def apply_pruning():
        with temporary_block_skip(model, blocks, block_path, selected_blocks):
            yield
    return apply_pruning


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    memory_trace = MemoryTrace()
    reset_cuda_peak()
    memory_trace.record("start")

    model, tokenizer, device = load_causal_lm(args.model, dtype=args.dtype)
    memory_trace.record("model_loaded")
    dataset = load_tokenized_text_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
    )
    memory_trace.record("dataset_loaded")
    blocks, block_path = get_transformer_blocks(model)
    score_rows = []
    if args.score == "activation":
        score_rows = score_blocks_by_activation(
            model=model,
            blocks=blocks,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            max_batches=args.score_max_batches,
        )
        memory_trace.record("scoring_activation")
        selected_blocks = select_blocks_from_ranking(
            ranking=rank_blocks_by_scores(score_rows, descending=False),
            num_blocks=len(blocks),
            pruning_ratio=args.pruning_ratio,
        )
    elif args.score == "loss_delta":
        score_rows = score_blocks_by_loss_delta(
            model=model,
            blocks=blocks,
            block_path=block_path,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            max_batches=args.score_max_batches,
        )
        memory_trace.record("scoring_loss_delta")
        selected_blocks = select_blocks_from_ranking(
            ranking=rank_blocks_by_scores(score_rows, descending=False),
            num_blocks=len(blocks),
            pruning_ratio=args.pruning_ratio,
        )
    else:
        selected_blocks = select_blocks(
            num_blocks=len(blocks),
            pruning_ratio=args.pruning_ratio,
            score=args.score,
        )
        memory_trace.record("scoring_rule_based")

    baseline = evaluate_perplexity(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
    )
    memory_trace.record("baseline_eval")
    with temporary_block_skip(model, blocks, block_path, selected_blocks):
        pruned = evaluate_perplexity(
            model,
            dataset,
            device=device,
            batch_size=args.batch_size,
        )
    memory_trace.record("pruned_eval")
    preservation = evaluate_preservation(
        model,
        dataset,
        device=device,
        apply_pruning=build_pruning_context(
            model,
            blocks,
            block_path,
            selected_blocks,
        ),
        batch_size=args.batch_size,
        max_batches=args.preservation_max_batches,
    )
    memory_trace.record("preservation_eval")

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
        "score_max_batches": args.score_max_batches,
        "preservation_max_batches": args.preservation_max_batches,
        "block_scores": score_rows,
        "selected_blocks": selected_blocks,
        "parameter_memory_mb": model_parameter_memory_mb(model),
        "baseline": baseline,
        "pruned": pruned,
        "preservation": preservation,
        "perplexity_delta": pruned["perplexity"] - baseline["perplexity"],
        "memory_trace": memory_trace.rows,
        **cuda_memory_mb(),
    }

    export_dir = None
    if args.export_pruned_model:
        export_dir = os.path.join(args.output_dir, "pruned_model")
        apply_block_skip(model, block_path, selected_blocks)
        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        save_json(export_dir, "amcprune_pruning_config.json", {
            "base_model": args.model,
            "pruning_unit": "block_skip",
            "block_path": block_path,
            "selected_blocks": selected_blocks,
            "score": args.score,
            "pruning_ratio": args.pruning_ratio,
        })
        memory_trace.record("export_pruned_model")
        result["exported_pruned_model"] = export_dir
        result["memory_trace"] = memory_trace.rows

    path = save_json(args.output_dir, "result.json", result)

    print(f"[AMCPrune] model={args.model}")
    print(f"[AMCPrune] blocks={len(blocks)} path={block_path}")
    if score_rows:
        print(f"[AMCPrune] block {args.score} scores:")
        for row in score_rows:
            marker = "*" if row["block"] in selected_blocks else " "
            if args.score == "activation":
                print(
                    f"  {marker} block={row['block']:02d} "
                    f"activation_abs_mean={row['activation_abs_mean']:.6e}"
                )
            elif args.score == "loss_delta":
                print(
                    f"  {marker} block={row['block']:02d} "
                    f"loss_delta={row['loss_delta']:.6e} "
                    f"skipped_ppl={row['skipped_perplexity']:.4f}"
                )
    print(f"[AMCPrune] selected_blocks={selected_blocks}")
    print(f"[AMCPrune] baseline_ppl={baseline['perplexity']:.4f}")
    print(f"[AMCPrune] pruned_ppl={pruned['perplexity']:.4f}")
    print(
        "[AMCPrune] preservation "
        f"hidden_cos={preservation['hidden_cosine_similarity']:.6f} "
        f"logit_kl={preservation['logit_kl_divergence']:.6f}"
    )
    if export_dir:
        print(f"[AMCPrune] exported_pruned_model={export_dir}")
    print("[AMCPrune] memory trace:")
    for row in memory_trace.rows:
        print(
            f"  {row['stage']}: "
            f"allocated={row['cuda_allocated_mb']:.2f}MB "
            f"reserved={row['cuda_reserved_mb']:.2f}MB "
            f"peak={row['cuda_peak_allocated_mb']:.2f}MB"
        )
    print(f"[AMCPrune] saved={path}")


if __name__ == "__main__":
    main()
