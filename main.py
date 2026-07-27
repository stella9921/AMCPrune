import argparse
import gc
import json
import os
from contextlib import contextmanager

from amcprune.data import load_tokenized_text_dataset
from amcprune.evaluate import benchmark_generation, evaluate_perplexity, evaluate_preservation
from amcprune.experiment import (
    TeeLogger,
    TimingTrace,
    build_pruning_plan,
    prepare_output_dir,
    resolve_config,
    save_block_scores,
    save_command,
    save_json as save_json_file,
    save_pruning_plan_units_csv,
    save_unit_decision_log,
    set_seed,
)
from amcprune.importance_scores import AMCImportanceScores
from amcprune.metrics import (
    MemoryTrace,
    cuda_memory_mb,
    model_parameter_count,
    model_parameter_memory_mb,
    reset_cuda_peak,
    save_json,
)
from amcprune.models import get_transformer_blocks, load_causal_lm
from amcprune.outliers import measure_block_outliers, save_outlier_metrics_csv
from amcprune.objective import apply_outlier_aware_objective, build_unit_objective_plan
from amcprune.pruning import (
    remove_transformer_blocks,
    select_blocks,
    select_blocks_from_ranking,
    temporary_block_skip,
)
from amcprune.scoring import (
    rank_blocks_by_scores,
    score_blocks_by_activation,
    score_blocks_by_activation_weight,
    score_blocks_by_hidden_cosine,
    score_blocks_by_loss_delta,
)
from amcprune.unit_pruning import (
    apply_unit_mask_pruning,
    apply_unit_physical_pruning,
    temporary_unit_mask_pruning,
)
from amcprune.unit_scoring import score_candidate_units_by_hessian_proxy, save_unit_scores_csv
from amcprune.visualization import plot_run_artifacts
from amcprune.units import inspect_block_units, save_unit_inventory_csv


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
    "score_cache": None,
    "score_max_batches": 8,
    "selection_objective": "score",
    "outlier_metric": "outlier_ratio",
    "outlier_weight": 0.25,
    "pruning_mode": "block",
    "unit_score": "none",
    "unit_pruning_ratio": 0.1,
    "unit_score_max_batches": 4,
    "unit_hvp_k_horizon": 1,
    "memory_weight": 0.25,
    "preservation_max_batches": 8,
    "inference_prompt": "The future of artificial intelligence is",
    "inference_max_new_tokens": 32,
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
        choices=[
            "block_index",
            "early_block",
            "activation",
            "activation_weight",
            "hidden_cosine",
            "loss_delta",
        ],
        default=None,
    )
    parser.add_argument("--score-cache", dest="score_cache", default=None)
    parser.add_argument("--score-max-batches", dest="score_max_batches", type=int, default=None)
    parser.add_argument(
        "--selection-objective",
        dest="selection_objective",
        choices=["score", "outlier_aware"],
        default=None,
    )
    parser.add_argument("--outlier-metric", dest="outlier_metric", default=None)
    parser.add_argument("--outlier-weight", dest="outlier_weight", type=float, default=None)
    parser.add_argument(
        "--pruning-mode",
        dest="pruning_mode",
        choices=["block", "unit_mask", "unit_physical", "depth_width_physical"],
        default=None,
    )
    parser.add_argument("--unit-score", dest="unit_score", choices=["none", "hessian_proxy", "hvp"], default=None)
    parser.add_argument("--unit-pruning-ratio", dest="unit_pruning_ratio", type=float, default=None)
    parser.add_argument("--unit-score-max-batches", dest="unit_score_max_batches", type=int, default=None)
    parser.add_argument("--unit-hvp-k-horizon", dest="unit_hvp_k_horizon", type=int, default=None)
    parser.add_argument("--memory-weight", dest="memory_weight", type=float, default=None)
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
    parser.add_argument("--inference-prompt", dest="inference_prompt", default=None)
    parser.add_argument("--inference-max-new-tokens", dest="inference_max_new_tokens", type=int, default=None)
    return parser.parse_args()


def _fmt_float(value, digits=6, default="NA"):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return f"{value:.{digits}f}"


def _fmt_sci(value, default="NA"):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return f"{value:.6e}"


def build_pruning_context(model, blocks, block_path, selected_blocks):
    @contextmanager
    def apply_pruning():
        with temporary_block_skip(model, blocks, block_path, selected_blocks):
            yield
    return apply_pruning


def build_unit_pruning_context(blocks, unit_objective_plan):
    @contextmanager
    def apply_pruning():
        with temporary_unit_mask_pruning(blocks, unit_objective_plan):
            yield
    return apply_pruning


def build_depth_width_pruning_context(model, blocks, block_path, depth_blocks, unit_objective_plan):
    @contextmanager
    def apply_pruning():
        with temporary_unit_mask_pruning(blocks, unit_objective_plan):
            with temporary_block_skip(model, blocks, block_path, depth_blocks):
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
    if config.get("score_cache"):
        print(f"[Config] score_cache={config['score_cache']}")
    print(
        f"[Config] selection_objective={config.get('selection_objective')} "
        f"outlier_metric={config.get('outlier_metric')} "
        f"outlier_weight={config.get('outlier_weight')}"
    )
    print(
        f"[Config] pruning_mode={config.get('pruning_mode')} "
        f"unit_score={config.get('unit_score')} "
        f"unit_pruning_ratio={config.get('unit_pruning_ratio')} "
        f"unit_hvp_k_horizon={config.get('unit_hvp_k_horizon')} "
        f"memory_weight={config.get('memory_weight')}"
    )


def load_score_cache(path):
    if path.endswith(".pt") or path.endswith(".pth"):
        return AMCImportanceScores.load(path).to_block_rows()
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict) and "rows" in payload:
        return payload["rows"]
    if isinstance(payload, dict) and "block_scores" in payload:
        return payload["block_scores"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported score cache format: {path}")


def select_blocks_for_config(config, model, blocks, block_path, dataset, device):
    if config.get("score_cache"):
        score_rows = load_score_cache(config["score_cache"])
        selected_blocks = select_blocks_from_ranking(
            ranking=rank_blocks_by_scores(score_rows, descending=False),
            num_blocks=len(blocks),
            pruning_ratio=float(config["pruning_ratio"]),
        )
        return score_rows, selected_blocks

    if config["score"] == "activation":
        score_rows = score_blocks_by_activation(
            model=model,
            blocks=blocks,
            dataset=dataset,
            device=device,
            batch_size=int(config["batch_size"]),
            max_batches=int(config["score_max_batches"]),
        )
    elif config["score"] == "activation_weight":
        score_rows = score_blocks_by_activation_weight(
            model=model,
            blocks=blocks,
            dataset=dataset,
            device=device,
            batch_size=int(config["batch_size"]),
            max_batches=int(config["score_max_batches"]),
        )
    elif config["score"] == "hidden_cosine":
        score_rows = score_blocks_by_hidden_cosine(
            model=model,
            blocks=blocks,
            dataset=dataset,
            device=device,
            batch_size=int(config["batch_size"]),
            max_batches=int(config["score_max_batches"]),
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
    else:
        selected_blocks = select_blocks(
            num_blocks=len(blocks),
            pruning_ratio=float(config["pruning_ratio"]),
            score=config["score"],
        )
        return [], selected_blocks

    selected_blocks = select_blocks_from_ranking(
        ranking=rank_blocks_by_scores(score_rows, descending=False),
        num_blocks=len(blocks),
        pruning_ratio=float(config["pruning_ratio"]),
    )
    return score_rows, selected_blocks


def print_score_rows(config, score_rows, selected_blocks):
    if not score_rows:
        return
    print(f"[AMCPrune] block {config['score']} scores:")
    for row in score_rows:
        marker = "*" if row["block"] in selected_blocks else " "
        if config["score"] == "activation":
            print(
                f"  {marker} block={row['block']:02d} "
                f"activation_abs_mean={row['activation_abs_mean']:.6e}"
            )
        elif config["score"] == "activation_weight":
            print(
                f"  {marker} block={row['block']:02d} "
                f"activation_abs_mean={row['activation_abs_mean']:.6e} "
                f"weight_abs_mean={row['weight_abs_mean']:.6e} "
                f"score={row['score']:.6e}"
            )
        elif config["score"] == "loss_delta":
            print(
                f"  {marker} block={row['block']:02d} "
                f"loss_delta={row['loss_delta']:.6e} "
                f"skipped_ppl={row['skipped_perplexity']:.4f}"
            )
        elif config["score"] == "hidden_cosine":
            print(
                f"  {marker} block={row['block']:02d} "
                f"hidden_cos={row['hidden_cosine_similarity']:.6f} "
                f"repr_delta={row['representation_delta']:.6e} "
                f"score={row['score']:.6e}"
            )
        else:
            print(f"  {marker} block={row['block']:02d} score={row['score']:.6e}")


def print_pruning_plan(pruning_plan):
    print(
        f"[Pruning Plan] unit={pruning_plan['pruning_unit']} "
        f"selected={pruning_plan['selected_units']}/{pruning_plan['total_units']} "
        f"actual_ratio={pruning_plan['actual_unit_pruning_ratio']:.4f}"
    )
    for unit in pruning_plan["units"]:
        marker = "*" if unit["selected"] else " "
        score_value = unit["score_value"]
        score_text = "NA" if score_value is None else f"{score_value:.6e}"
        print(
            f"  {marker} {unit['unit_name']} type={unit['unit_type']} "
            f"score={score_text} reason={unit['reason']}"
        )


def print_unit_objective_plan(unit_plan, max_rows=40):
    if not unit_plan:
        return
    selected = [unit for unit in unit_plan["units"] if unit.get("selected")]
    print(
        "[Unit Objective] "
        f"objective={unit_plan['objective']} "
        f"cost_key={unit_plan.get('cost_key', 'memory_cost')} "
        f"selected={unit_plan['selected_units']}/{unit_plan['total_units']} "
        f"unit_ratio={unit_plan['actual_unit_pruning_ratio']:.4f} "
        f"memory_ratio={unit_plan['actual_memory_pruning_ratio']:.4f}"
    )
    print(
        "[Unit Objective] "
        f"target_pruned_cost={unit_plan['target_pruned_memory_cost']:.2f} "
        f"selected_cost={unit_plan['selected_memory_cost']:.2f} "
        f"total_cost={unit_plan['total_memory_cost']:.2f}"
    )
    for unit in selected[:max_rows]:
        print(
            "  * "
            f"{unit['unit_name']} type={unit['unit_type']} "
            f"hvp={_fmt_sci(unit.get('hessian_score'))} "
            f"outlier={_fmt_sci(unit.get('outlier_risk'))} "
            f"resource_cost={_fmt_float(unit.get('resource_cost_effective', unit.get('memory_cost')), digits=2)} "
            f"param_cost={_fmt_float(unit.get('parameter_cost'), digits=2)} "
            f"flops={_fmt_float(unit.get('linear_flops', 0.0) + unit.get('attention_flops', 0.0), digits=2)} "
            f"kv={_fmt_float(unit.get('kv_cache_cost'), digits=2)} "
            f"cost_type={unit.get('resource_cost_type', 'parameter_proxy')} "
            f"keep={_fmt_sci(unit.get('keep_score'))} "
            f"objective={_fmt_sci(unit.get('objective_score'))}"
        )
    if len(selected) > max_rows:
        print(f"  ... {len(selected) - max_rows} more selected units")


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
            print("[Model Load] returned to main", flush=True)
        memory_trace.record("model_loaded")
        print("[Model Load] memory trace recorded", flush=True)

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

        with timing_trace.stage(f"scoring_{config['score']}"):
            score_rows, selected_blocks = select_blocks_for_config(
                config=config,
                model=model,
                blocks=blocks,
                block_path=block_path,
                dataset=dataset,
                device=device,
            )
        memory_trace.record(f"scoring_{config['score']}")

        with timing_trace.stage("outlier_metrics"):
            outlier_metrics = measure_block_outliers(
                model=model,
                blocks=blocks,
                block_path=block_path,
                dataset=dataset,
                device=device,
                batch_size=int(config["batch_size"]),
                max_batches=int(config["score_max_batches"]),
                selected_blocks=selected_blocks,
            )
        memory_trace.record("outlier_metrics")

        if score_rows and config.get("selection_objective") == "outlier_aware":
            with timing_trace.stage("outlier_aware_selection"):
                score_rows = apply_outlier_aware_objective(
                    score_rows=score_rows,
                    outlier_rows=outlier_metrics,
                    outlier_metric=config.get("outlier_metric", "outlier_ratio"),
                    outlier_weight=float(config.get("outlier_weight", 0.25)),
                )
                selected_blocks = select_blocks_from_ranking(
                    ranking=rank_blocks_by_scores(score_rows, descending=False),
                    num_blocks=len(blocks),
                    pruning_ratio=float(config["pruning_ratio"]),
                )
                selected_set = set(selected_blocks)
                for row in outlier_metrics:
                    row["selected_block"] = row["block"] in selected_set
                print(
                    "[Objective] outlier-aware selection active: "
                    f"metric={config.get('outlier_metric')} "
                    f"weight={float(config.get('outlier_weight', 0.25)):.4f} "
                    f"selected_blocks={selected_blocks}"
                )
            memory_trace.record("outlier_aware_selection")

        save_json_file(os.path.join(output_dir, "outlier_metrics.json"), outlier_metrics)
        save_outlier_metrics_csv(os.path.join(output_dir, "outlier_metrics.csv"), outlier_metrics)
        print(f"[Outlier Metrics] saved {len(outlier_metrics)} block records")

        pruning_plan = build_pruning_plan(
            model_name=config["model"],
            block_path=block_path,
            num_blocks=len(blocks),
            pruning_unit="transformer_block",
            pruning_ratio=float(config["pruning_ratio"]),
            score_name=config["score"],
            score_rows=score_rows,
            selected_blocks=selected_blocks,
        )
        score_json = {
            "score": config["score"],
            "selection_objective": config.get("selection_objective"),
            "outlier_metric": config.get("outlier_metric"),
            "outlier_weight": config.get("outlier_weight"),
            "selected_blocks": selected_blocks,
            "rows": score_rows,
        }
        save_json_file(os.path.join(output_dir, "block_scores.json"), score_json)
        save_block_scores(os.path.join(output_dir, "block_scores.csv"), score_rows, selected_blocks)
        if score_rows:
            importance_scores = AMCImportanceScores.from_block_rows(
                rows=score_rows,
                model=config["model"],
                block_path=block_path,
                score_name=config["score"],
                metadata={
                    "run_id": run_id,
                    "strategy": config["strategy"],
                    "dataset": config["dataset"],
                    "dataset_config": config["dataset_config"],
                    "split": config["split"],
                    "max_samples": config["max_samples"],
                    "seq_len": config["seq_len"],
                    "score_max_batches": config["score_max_batches"],
                    "selection_objective": config.get("selection_objective"),
                    "outlier_metric": config.get("outlier_metric"),
                    "outlier_weight": config.get("outlier_weight"),
                    "seed": config.get("seed"),
                },
            )
            importance_scores_path = os.path.join(output_dir, "importance-scores.pt")
            importance_scores.save(importance_scores_path)
            score_json["importance_scores_path"] = importance_scores_path
            save_json_file(os.path.join(output_dir, "block_scores.json"), score_json)
            print(f"[Scores] saved importance scores: {importance_scores_path}")
        save_json_file(os.path.join(output_dir, "pruning_plan.json"), pruning_plan)
        save_pruning_plan_units_csv(os.path.join(output_dir, "pruning_plan_units.csv"), pruning_plan)
        depth_pruned_blocks = []
        width_candidate_blocks = selected_blocks
        if config.get("pruning_mode") == "depth_width_physical":
            depth_pruned_blocks = list(selected_blocks)
            depth_pruned_set = set(depth_pruned_blocks)
            width_candidate_blocks = [
                index for index in range(len(blocks)) if index not in depth_pruned_set
            ]
            print(
                "[Depth-Width] "
                f"depth_pruned_blocks={depth_pruned_blocks} "
                f"width_candidate_blocks={width_candidate_blocks}"
            )

        unit_score_rows = []
        unit_objective_plan = None
        if config.get("unit_score") in {"hessian_proxy", "hvp"} or config.get("pruning_mode") in {"unit_mask", "unit_physical", "depth_width_physical"}:
            unit_score_method = config.get("unit_score")
            if unit_score_method == "none":
                unit_score_method = "hvp"
            with timing_trace.stage(f"unit_scoring_{unit_score_method}"):
                unit_score_rows = score_candidate_units_by_hessian_proxy(
                    model=model,
                    blocks=blocks,
                    block_path=block_path,
                    selected_blocks=width_candidate_blocks,
                    dataset=dataset,
                    device=device,
                    batch_size=int(config["batch_size"]),
                    max_batches=int(config["unit_score_max_batches"]),
                    method=unit_score_method,
                    k_horizon=int(config["unit_hvp_k_horizon"]),
                    seq_len=int(config["seq_len"]),
                )
            memory_trace.record(f"unit_scoring_{unit_score_method}")
            unit_objective_plan = build_unit_objective_plan(
                unit_rows=unit_score_rows,
                pruning_ratio=float(config["unit_pruning_ratio"]),
                outlier_weight=float(config["outlier_weight"]),
                memory_weight=float(config["memory_weight"]),
            )
            save_json_file(os.path.join(output_dir, "unit_scores.json"), unit_score_rows)
            save_unit_scores_csv(os.path.join(output_dir, "unit_scores.csv"), unit_score_rows)
            save_json_file(os.path.join(output_dir, "unit_objective_plan.json"), unit_objective_plan)
            save_unit_scores_csv(
                os.path.join(output_dir, "unit_objective_plan.csv"),
                unit_objective_plan["units"],
            )
            decision_log_path = save_unit_decision_log(
                os.path.join(output_dir, "unit_decision_log.txt"),
                unit_objective_plan,
            )
            print(
                "[Unit Objective] "
                f"candidates={unit_objective_plan['total_units']} "
                f"selected={unit_objective_plan['selected_units']} "
                f"ratio={unit_objective_plan['actual_unit_pruning_ratio']:.4f}"
            )
            print(f"[Unit Objective] decision log saved: {decision_log_path}")
            print_unit_objective_plan(unit_objective_plan)
        unit_inventory = inspect_block_units(blocks, block_path, width_candidate_blocks)
        save_json_file(os.path.join(output_dir, "unit_inventory.json"), unit_inventory)
        save_unit_inventory_csv(os.path.join(output_dir, "unit_inventory.csv"), unit_inventory)
        print(f"[Unit Inventory] saved {len(unit_inventory)} block unit records")
        print_pruning_plan(pruning_plan)

        dense_parameter_count = model_parameter_count(model)
        dense_parameter_memory_mb = model_parameter_memory_mb(model)

        with timing_trace.stage("baseline_eval"):
            baseline = evaluate_perplexity(
                model,
                dataset,
                device=device,
                batch_size=int(config["batch_size"]),
            )
        memory_trace.record("baseline_eval")

        with timing_trace.stage("preservation_eval"):
            if config.get("pruning_mode") == "block":
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
            elif config.get("pruning_mode") == "depth_width_physical":
                preservation = evaluate_preservation(
                    model,
                    dataset,
                    device=device,
                    apply_pruning=build_depth_width_pruning_context(
                        model,
                        blocks,
                        block_path,
                        depth_pruned_blocks,
                        unit_objective_plan,
                    ),
                    batch_size=int(config["batch_size"]),
                    max_batches=int(config["preservation_max_batches"]),
                )
            else:
                preservation = evaluate_preservation(
                    model,
                    dataset,
                    device=device,
                    apply_pruning=build_unit_pruning_context(
                        blocks,
                        unit_objective_plan,
                    ),
                    batch_size=int(config["batch_size"]),
                    max_batches=int(config["preservation_max_batches"]),
                )
        memory_trace.record("preservation_eval")

        with timing_trace.stage("dense_inference_benchmark"):
            dense_inference = benchmark_generation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=config["inference_prompt"],
                max_new_tokens=int(config["inference_max_new_tokens"]),
            )
        memory_trace.record("dense_inference_benchmark")

        with timing_trace.stage("physical_pruning"):
            if config.get("pruning_mode") in {"unit_mask", "unit_physical", "depth_width_physical"}:
                if not unit_objective_plan:
                    raise ValueError("unit_mask pruning requires unit_objective_plan.")
                if config.get("pruning_mode") in {"unit_physical", "depth_width_physical"}:
                    physical_pruning = apply_unit_physical_pruning(blocks, unit_objective_plan)
                else:
                    physical_pruning = apply_unit_mask_pruning(blocks, unit_objective_plan)
                if config.get("pruning_mode") == "depth_width_physical":
                    depth_pruning = remove_transformer_blocks(
                        model,
                        block_path,
                        depth_pruned_blocks,
                    )
                    physical_pruning.update({
                        "pruning_mode": "depth_width_physical",
                        "depth_pruning": depth_pruning,
                        "block_path": block_path,
                        "original_num_blocks": depth_pruning["original_num_blocks"],
                        "pruned_num_blocks": depth_pruning["pruned_num_blocks"],
                        "remaining_num_blocks": depth_pruning["remaining_num_blocks"],
                        "removed_original_indices": depth_pruning["removed_original_indices"],
                        "kept_original_indices": depth_pruning["kept_original_indices"],
                        "width_candidate_original_indices": width_candidate_blocks,
                    })
                else:
                    physical_pruning.update({
                        "block_path": block_path,
                        "original_num_blocks": len(blocks),
                        "pruned_num_blocks": 0,
                        "remaining_num_blocks": len(blocks),
                        "removed_original_indices": [],
                        "kept_original_indices": list(range(len(blocks))),
                    })
                physical_pruning["planned_memory_cost"] = sum(
                    float(unit.get("resource_cost_effective", unit.get("memory_cost", 0.0)) or 0.0)
                    for unit in unit_objective_plan["units"]
                    if unit.get("selected")
                )
            else:
                physical_pruning = remove_transformer_blocks(
                    model,
                    block_path,
                    selected_blocks,
                )
                del blocks
            gc.collect()
            if device.type == "cuda":
                import torch
                torch.cuda.empty_cache()
        memory_trace.record("physical_pruning")

        pruned_parameter_count = model_parameter_count(model)
        pruned_parameter_memory_mb = model_parameter_memory_mb(model)
        parameter_sparsity = 1.0 - (
            pruned_parameter_count / max(dense_parameter_count, 1)
        )
        print(
            "[Pruning Result] "
            f"mode={physical_pruning.get('pruning_mode')} "
            f"dense_params={dense_parameter_count:,} "
            f"pruned_params={pruned_parameter_count:,} "
            f"removed_params={dense_parameter_count - pruned_parameter_count:,} "
            f"parameter_sparsity={parameter_sparsity * 100:.2f}%"
        )
        if physical_pruning.get("selected_by_type"):
            print(f"[Pruning Result] selected_by_type={physical_pruning['selected_by_type']}")
        if physical_pruning.get("physically_pruned_by_type"):
            print(f"[Pruning Result] physically_pruned_by_type={physical_pruning['physically_pruned_by_type']}")
        if physical_pruning.get("masked_fallback_by_type"):
            print(f"[Pruning Result] masked_fallback_by_type={physical_pruning['masked_fallback_by_type']}")

        with timing_trace.stage("pruned_eval"):
            pruned = evaluate_perplexity(
                model,
                dataset,
                device=device,
                batch_size=int(config["batch_size"]),
            )
        memory_trace.record("pruned_eval")

        with timing_trace.stage("pruned_inference_benchmark"):
            pruned_inference = benchmark_generation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=config["inference_prompt"],
                max_new_tokens=int(config["inference_max_new_tokens"]),
            )
        memory_trace.record("pruned_inference_benchmark")

        result = {
            "run_id": run_id,
            "config": config,
            "model": config["model"],
            "dataset": config["dataset"],
            "dataset_config": config["dataset_config"],
            "split": config["split"],
            "num_blocks": physical_pruning["original_num_blocks"],
            "block_path": block_path,
            "pruning_unit": "transformer_block",
            "pruning_mode": config.get("pruning_mode"),
            "pruning_ratio": float(config["pruning_ratio"]),
            "unit_score": config.get("unit_score"),
            "unit_pruning_ratio": float(config.get("unit_pruning_ratio", 0.0)),
            "score": config["score"],
            "score_cache": config.get("score_cache"),
            "score_max_batches": int(config["score_max_batches"]),
            "preservation_max_batches": int(config["preservation_max_batches"]),
            "block_scores": score_rows,
            "pruning_plan": pruning_plan,
            "selected_blocks": selected_blocks,
            "depth_pruned_blocks": depth_pruned_blocks,
            "width_candidate_blocks": width_candidate_blocks,
            "unit_inventory": unit_inventory,
            "unit_scores": unit_score_rows,
            "unit_objective_plan": unit_objective_plan,
            "outlier_metrics": outlier_metrics,
            "physical_pruning": physical_pruning,
            "dense_parameter_count": dense_parameter_count,
            "pruned_parameter_count": pruned_parameter_count,
            "removed_parameter_count": dense_parameter_count - pruned_parameter_count,
            "parameter_sparsity": parameter_sparsity,
            "dense_parameter_memory_mb": dense_parameter_memory_mb,
            "pruned_parameter_memory_mb": pruned_parameter_memory_mb,
            "parameter_memory_reduction_mb": (
                dense_parameter_memory_mb - pruned_parameter_memory_mb
            ),
            "baseline": baseline,
            "pruned": pruned,
            "preservation": preservation,
            "inference": {
                "dense": dense_inference,
                "pruned": pruned_inference,
                "ttft_delta_seconds": pruned_inference["ttft_seconds"] - dense_inference["ttft_seconds"],
                "tps_delta": pruned_inference["tokens_per_second"] - dense_inference["tokens_per_second"],
                "peak_vram_delta_mb": pruned_inference["peak_vram_mb"] - dense_inference["peak_vram_mb"],
            },
            "perplexity_delta": pruned["perplexity"] - baseline["perplexity"],
            "memory_trace": memory_trace.rows,
            "timing_trace": timing_trace.rows,
            **cuda_memory_mb(),
        }

        export_dir = None
        if config.get("export_pruned_model"):
            with timing_trace.stage("export_pruned_model"):
                export_dir = os.path.join(output_dir, "pruned_model")
                os.makedirs(export_dir, exist_ok=True)
                if config.get("pruning_mode") in {"unit_physical", "depth_width_physical"}:
                    import torch
                    torch.save(
                        model.state_dict(),
                        os.path.join(export_dir, "unit_physical_state_dict.pt"),
                    )
                else:
                    model.save_pretrained(export_dir)
                tokenizer.save_pretrained(export_dir)
                save_json(export_dir, "amcprune_pruning_config.json", {
                    "base_model": config["model"],
                    "pruning_unit": "transformer_block",
                    "pruning_mode": config.get("pruning_mode"),
                    "block_path": block_path,
                    "selected_blocks": selected_blocks,
                    "depth_pruned_blocks": depth_pruned_blocks,
                    "width_candidate_blocks": width_candidate_blocks,
                    "unit_score": config.get("unit_score"),
                    "unit_objective_plan": unit_objective_plan,
                    "physical_pruning": physical_pruning,
                    "remaining_num_blocks": physical_pruning["remaining_num_blocks"],
                    "selected_unit_names": [
                        unit["unit_name"] for unit in pruning_plan["units"] if unit["selected"]
                    ],
                    "score": config["score"],
                    "score_cache": config.get("score_cache"),
                    "pruning_ratio": float(config["pruning_ratio"]),
                    "run_id": run_id,
                    "export_note": (
                        "unit_physical saves a state_dict because per-layer FFN "
                        "dimensions may differ from the base HuggingFace config."
                        if config.get("pruning_mode") in {"unit_physical", "depth_width_physical"}
                        else "save_pretrained-compatible export"
                    ),
                })
            memory_trace.record("export_pruned_model")
            result["exported_pruned_model"] = export_dir
            result["memory_trace"] = memory_trace.rows
            result["timing_trace"] = timing_trace.rows

        plot_paths = plot_run_artifacts(
            output_dir=output_dir,
            run_id=run_id,
            score_rows=score_rows,
            selected_blocks=selected_blocks,
            pruning_plan=pruning_plan,
            baseline=baseline,
            pruned=pruned,
            preservation=preservation,
            memory_trace=memory_trace.rows,
            timing_trace=timing_trace.rows,
            unit_inventory=unit_inventory,
            outlier_metrics=outlier_metrics,
            inference_metrics=result["inference"],
            unit_objective_plan=unit_objective_plan,
        )
        result["plots"] = plot_paths
        path = save_json(output_dir, "result.json", result)
        save_json_file(os.path.join(output_dir, "timing_trace.json"), timing_trace.rows)
        save_json_file(os.path.join(output_dir, "memory_trace.json"), memory_trace.rows)
        save_json_file(os.path.join(output_dir, "inference_metrics.json"), result["inference"])
        save_json_file(os.path.join(output_dir, "plots.json"), plot_paths)

        print(f"[AMCPrune] model={config['model']}")
        print(
            f"[AMCPrune] blocks={physical_pruning['original_num_blocks']} "
            f"path={block_path}"
        )
        print_score_rows(config, score_rows, selected_blocks)
        print(f"[AMCPrune] selected_blocks={selected_blocks}")
        print(
            f"[AMCPrune] physical_blocks="
            f"{physical_pruning['original_num_blocks']}->"
            f"{physical_pruning['remaining_num_blocks']} "
            f"parameter_sparsity={parameter_sparsity:.6f}"
        )
        print(f"[AMCPrune] baseline_ppl={baseline['perplexity']:.4f}")
        print(f"[AMCPrune] pruned_ppl={pruned['perplexity']:.4f}")
        print(
            "[AMCPrune] preservation "
            f"hidden_cos={_fmt_float(preservation.get('hidden_cosine_similarity'))} "
            f"logit_kl={_fmt_float(preservation.get('logit_kl_divergence'))}"
        )
        if export_dir:
            print(f"[AMCPrune] exported_pruned_model={export_dir}")
        print("[AMCPrune] inference benchmark:")
        print(
            f"  dense: ttft={dense_inference['ttft_seconds']:.4f}s "
            f"tps={dense_inference['tokens_per_second']:.2f} "
            f"peak_vram={dense_inference['peak_vram_mb']:.2f}MB"
        )
        print(
            f"  pruned: ttft={pruned_inference['ttft_seconds']:.4f}s "
            f"tps={pruned_inference['tokens_per_second']:.2f} "
            f"peak_vram={pruned_inference['peak_vram_mb']:.2f}MB"
        )
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
        if plot_paths:
            print(f"[Visualization] saved {len(plot_paths)} plots under {os.path.join(output_dir, 'plots')}")
        print(f"[AMCPrune] saved={path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
