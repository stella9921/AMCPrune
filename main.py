import argparse
import os
from contextlib import contextmanager

from amcprune.data import load_tokenized_text_dataset
from amcprune.evaluate import evaluate_perplexity, evaluate_preservation
from amcprune.experiment import (
    TeeLogger,
    TimingTrace,
    prepare_output_dir,
    resolve_config,
    save_block_scores,
    save_command,
    save_json as save_json_file,
    set_seed,
)
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


DEFAULT_CONFIG = {
    "model": "sshleifer/tiny-gpt2",
    "dataset": "wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "split": "test",
    "max_samples": 32,
    "seq_len": 128,
    "batch_size": 1,
    "dtype": "auto",
    "pruning_ratio": 0.25,
    "score": "block_index",
    "score_max_batches": 8,
    "preservation_max_batches": 8,
    "export_pruned_model": False,
    "output_root": "exp/runs",
    "output_dir": None,
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(description="AMCPrune LLM prototype")
    parser.add_argument("--strategy", default=None, help="Preset name in configs/strategy/{name}.yaml")
    parser.add_argument("--config", default=None, help="Explicit YAML config path")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-config", dest="dataset_config", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--pruning-ratio", dest="pruning_ratio", type=float, default=None)
    parser.add_argument(
        "--score",
        choices=["block_index", "early_block", "activation", "loss_delta"],
        default=None,
    )
    parser.add_argument("--score-max-batches", dest="score_max_batches", type=int, default=None)
    parser.add_argument(
        "--preservation-max-batches",
        dest="preservation_max_batches",
        type=int,
        default=None,
    )
    parser.add_argument("--export-pruned-model", dest="export_pruned_model", action="store_true", default=None)
    parser.add_argument("--no-export-pruned-model", dest="export_pruned_model", action="store_false")
    parser.add_argument("--output-root", dest="output_root", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_pruning_context(model, blocks, block_path, selected_blocks):
    @contextmanager
    def apply_pruning():
        with temporary_block_skip(model, blocks, block_path, selected_blocks):
            yield
    return apply_pruning


def print_config_summary(config):
    print(f"[Config] strategy={config['strategy']} file={config.get('config_path')}")
    print(
        f"[Config] model={config['model']} dataset={config['dataset']} "
        f"split={config['split']} score={config['score']}"
    )
    print(
        f"[Config] pruning_ratio={float(config['pruning_ratio']):.4f} "
        f"max_samples={config['max_samples']} seq_len={config['seq_len']} "
        f"batch_size={config['batch_size']} seed={config.get('seed')}"
    )


def main():
    args = parse_args()
    config = resolve_config(args, DEFAULT_CONFIG)
    output_dir, run_id = prepare_output_dir(config)
    logger = TeeLogger(os.path.join(output_dir, f"{run_id}__run.log"))
    logger.start()
    try:
        print(f"[Experiment] run_id={run_id}")
        print(f"[Experiment] output_dir={output_dir}")
        save_command(os.path.join(output_dir, "command.txt"))
        save_json_file(os.path.join(output_dir, "resolved_config.json"), config)
        print_config_summary(config)
        set_seed(config.get("seed"))
        print(f"[Reproducibility] seed={config.get('seed')}")

        memory_trace = MemoryTrace()
        timing_trace = TimingTrace()
        reset_cuda_peak()
        memory_trace.record("start")

        with timing_trace.stage("model_load"):
            model, tokenizer, device = load_causal_lm(config["model"], dtype=config["dtype"])
        memory_trace.record("model_loaded")

        with timing_trace.stage("dataset_load"):
            dataset = load_tokenized_text_dataset(
                tokenizer=tokenizer,
                dataset_name=config["dataset"],
                dataset_config=config["dataset_config"],
                split=config["split"],
                max_samples=int(config["max_samples"]),
                seq_len=int(config["seq_len"]),
            )
        memory_trace.record("dataset_loaded")

        blocks, block_path = get_transformer_blocks(model)
        print(f"[Topology] blocks={len(blocks)} path={block_path}")

        score_rows = []
        with timing_trace.stage(f"scoring_{config['score']}"):
            if config["score"] == "activation":
                score_rows = score_blocks_by_activation(
                    model=model,
                    blocks=blocks,
                    dataset=dataset,
                    device=device,
                    batch_size=int(config["batch_size"]),
                    max_batches=int(config["score_max_batches"]),
                )
                selected_blocks = select_blocks_from_ranking(
                    ranking=rank_blocks_by_scores(score_rows, descending=False),
                    num_blocks=len(blocks),
                    pruning_ratio=float(config["pruning_ratio"]),
                )
            elif config["score"] == "loss_delta":
                score_rows = score_blocks_by_loss_delta(
                    model=model,
                    blocks=blocks,
                    block_path=block_path,
                    dataset=dataset,
                    device=device,
                    batch_size=int(config["batch_size"]),
                    max_batches=int(config["score_max_batches"]),
                )
                selected_blocks = select_blocks_from_ranking(
                    ranking=rank_blocks_by_scores(score_rows, descending=False),
                    num_blocks=len(blocks),
                    pruning_ratio=float(config["pruning_ratio"]),
                )
            else:
                selected_blocks = select_blocks(
                    num_blocks=len(blocks),
                    pruning_ratio=float(config["pruning_ratio"]),
                    score=config["score"],
                )
        memory_trace.record(f"scoring_{config['score']}")

        score_json_path = os.path.join(output_dir, "block_scores.json")
        score_csv_path = os.path.join(output_dir, "block_scores.csv")
        save_json_file(score_json_path, {
            "score": config["score"],
            "selected_blocks": selected_blocks,
            "rows": score_rows,
        })
        save_block_scores(score_csv_path, score_rows, selected_blocks)

        with timing_trace.stage("baseline_eval"):
            baseline = evaluate_perplexity(
                model,
                dataset,
                device=device,
                batch_size=int(config["batch_size"]),
            )
        memory_trace.record("baseline_eval")

        with timing_trace.stage("pruned_eval"):
            with temporary_block_skip(model, blocks, block_path, selected_blocks):
                pruned = evaluate_perplexity(
                    model,
                    dataset,
                    device=device,
                    batch_size=int(config["batch_size"]),
                )
        memory_trace.record("pruned_eval")

        with timing_trace.stage("preservation_eval"):
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
                batch_size=int(config["batch_size"]),
                max_batches=int(config["preservation_max_batches"]),
            )
        memory_trace.record("preservation_eval")

        result = {
            "run_id": run_id,
            "config": config,
            "model": config["model"],
            "dataset": config["dataset"],
            "dataset_config": config["dataset_config"],
            "split": config["split"],
            "num_blocks": len(blocks),
            "block_path": block_path,
            "pruning_unit": "block_skip",
            "pruning_ratio": float(config["pruning_ratio"]),
            "score": config["score"],
            "score_max_batches": int(config["score_max_batches"]),
            "preservation_max_batches": int(config["preservation_max_batches"]),
            "block_scores": score_rows,
            "selected_blocks": selected_blocks,
            "parameter_memory_mb": model_parameter_memory_mb(model),
            "baseline": baseline,
            "pruned": pruned,
            "preservation": preservation,
            "perplexity_delta": pruned["perplexity"] - baseline["perplexity"],
            "memory_trace": memory_trace.rows,
            "timing_trace": timing_trace.rows,
            **cuda_memory_mb(),
        }

        export_dir = None
        if config.get("export_pruned_model"):
            with timing_trace.stage("export_pruned_model"):
                export_dir = os.path.join(output_dir, "pruned_model")
                apply_block_skip(model, block_path, selected_blocks)
                model.save_pretrained(export_dir)
                tokenizer.save_pretrained(export_dir)
                save_json(export_dir, "amcprune_pruning_config.json", {
                    "base_model": config["model"],
                    "pruning_unit": "block_skip",
                    "block_path": block_path,
                    "selected_blocks": selected_blocks,
                    "score": config["score"],
                    "pruning_ratio": float(config["pruning_ratio"]),
                    "run_id": run_id,
                })
            memory_trace.record("export_pruned_model")
            result["exported_pruned_model"] = export_dir
            result["memory_trace"] = memory_trace.rows
            result["timing_trace"] = timing_trace.rows

        path = save_json(output_dir, "result.json", result)
        save_json_file(os.path.join(output_dir, "timing_trace.json"), timing_trace.rows)
        save_json_file(os.path.join(output_dir, "memory_trace.json"), memory_trace.rows)

        print(f"[AMCPrune] model={config['model']}")
        print(f"[AMCPrune] blocks={len(blocks)} path={block_path}")
        if score_rows:
            print(f"[AMCPrune] block {config['score']} scores:")
            for row in score_rows:
                marker = "*" if row["block"] in selected_blocks else " "
                if config["score"] == "activation":
                    print(
                        f"  {marker} block={row['block']:02d} "
                        f"activation_abs_mean={row['activation_abs_mean']:.6e}"
                    )
                elif config["score"] == "loss_delta":
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
        print("[AMCPrune] timing trace:")
        for row in timing_trace.rows:
            print(
                f"  {row['stage']}: seconds={row['seconds']:.2f} "
                f"elapsed={row['elapsed_seconds']:.2f}"
            )
        print(f"[AMCPrune] saved={path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
