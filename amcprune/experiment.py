import csv
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
import yaml


def load_yaml_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_config(args, defaults):
    strategy = args.strategy
    config_path = args.config
    if strategy and not config_path:
        config_path = os.path.join("configs", "strategy", f"{strategy}.yaml")
    config = dict(defaults)
    config_file_values = load_yaml_config(config_path)
    config.update(config_file_values)

    cli_values = vars(args)
    for key, value in cli_values.items():
        if value is not None:
            config[key] = value
    config["config_path"] = config_path
    config["strategy"] = strategy or config.get("strategy") or "manual"
    return config


def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def sanitize(value):
    text = str(value).replace("/", "-").replace("\\", "-")
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in text)


def make_run_id(config):
    model = sanitize(config["model"].split("/")[-1])
    strategy = sanitize(config.get("strategy", config.get("score", "manual")))
    ratio = int(round(float(config["pruning_ratio"]) * 1000))
    return f"{model}__{strategy}__prune-{ratio:03d}p0__{timestamp()}"


def prepare_output_dir(config):
    if config.get("output_dir"):
        output_dir = config["output_dir"]
        run_id = os.path.basename(os.path.normpath(output_dir)) or make_run_id(config)
    else:
        run_id = config.get("run_name") or make_run_id(config)
        output_dir = os.path.join(config.get("output_root", "exp/runs"), run_id)
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir
    config["run_id"] = run_id
    return output_dir, run_id


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TeeLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.stream = None
        self.stdout = None
        self.stderr = None

    def start(self):
        self.stream = open(self.log_path, "a", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.stdout.write(message)
        self.stream.write(message)
        self.stream.flush()

    def flush(self):
        self.stdout.flush()
        self.stream.flush()

    def close(self):
        if self.stream is None:
            return
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.stream.close()
        self.stream = None


class TimingTrace:
    def __init__(self):
        self.rows = []
        self.started_at = time.perf_counter()

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        wall_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Time] stage={name} started_at={wall_start}")
        try:
            yield
        finally:
            end = time.perf_counter()
            elapsed = end - start
            total = end - self.started_at
            row = {
                "stage": name,
                "started_at": wall_start,
                "seconds": elapsed,
                "elapsed_seconds": total,
            }
            self.rows.append(row)
            print(
                f"[Time] stage={name} seconds={elapsed:.2f} "
                f"elapsed={total:.2f}"
            )


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
    return path


def save_command(path):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(" ".join(sys.argv) + "\n")


def save_block_scores(path, score_rows, selected_blocks):
    if not score_rows:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    selected = set(selected_blocks)
    fieldnames = sorted({key for row in score_rows for key in row.keys()} | {"selected"})
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in score_rows:
            payload = dict(row)
            payload["selected"] = row.get("block") in selected
            writer.writerow(payload)
    return path


def build_pruning_plan(
    *,
    model_name,
    block_path,
    num_blocks,
    pruning_unit,
    pruning_ratio,
    score_name,
    score_rows,
    selected_blocks,
):
    selected = set(selected_blocks)
    score_by_block = {row.get("block"): row for row in score_rows}
    units = []
    for index in range(num_blocks):
        row = score_by_block.get(index, {})
        units.append({
            "unit_id": index,
            "unit_name": f"{block_path}.{index}",
            "unit_type": pruning_unit,
            "parent_path": block_path,
            "selected": index in selected,
            "score_name": score_name,
            "score_value": row.get("score"),
            "score_details": row,
            "reason": "selected by lowest score" if index in selected else "kept by score ranking",
        })
    selected_units = [unit for unit in units if unit["selected"]]
    return {
        "schema_version": 1,
        "model": model_name,
        "pruning_unit": pruning_unit,
        "pruning_scope": block_path,
        "score_name": score_name,
        "pruning_ratio_target": float(pruning_ratio),
        "total_units": num_blocks,
        "selected_units": len(selected_units),
        "actual_unit_pruning_ratio": len(selected_units) / max(num_blocks, 1),
        "selected_unit_ids": selected_blocks,
        "units": units,
        "notes": (
            "The schema is unit-agnostic. Future head or FFN-neuron pruning can "
            "reuse unit_type, parent_path, score_value, selected, and reason."
        ),
    }


def save_pruning_plan_units_csv(path, pruning_plan):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "unit_id",
        "unit_name",
        "unit_type",
        "parent_path",
        "selected",
        "score_name",
        "score_value",
        "reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for unit in pruning_plan["units"]:
            writer.writerow({key: unit.get(key) for key in fieldnames})
    return path


def save_unit_decision_log(path, unit_plan):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    units = unit_plan.get("units", [])
    selected = [unit for unit in units if unit.get("selected")]
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("[Unit Objective Summary]\n")
        stream.write(f"objective={unit_plan.get('objective')}\n")
        stream.write(f"total_units={unit_plan.get('total_units')}\n")
        stream.write(f"selected_units={unit_plan.get('selected_units')}\n")
        stream.write(f"target_pruning_ratio={unit_plan.get('pruning_ratio_target')}\n")
        stream.write(f"actual_unit_pruning_ratio={unit_plan.get('actual_unit_pruning_ratio')}\n")
        stream.write(f"actual_memory_pruning_ratio={unit_plan.get('actual_memory_pruning_ratio')}\n")
        stream.write(f"selected_memory_cost={unit_plan.get('selected_memory_cost')}\n")
        stream.write(f"total_memory_cost={unit_plan.get('total_memory_cost')}\n")
        stream.write(f"cost_key={unit_plan.get('cost_key')}\n")
        stream.write("\n[Selected Units]\n")
        for unit in selected:
            stream.write(
                "SELECT "
                f"block={unit.get('block')} "
                f"type={unit.get('unit_type')} "
                f"idx={unit.get('unit_index')} "
                f"name={unit.get('unit_name')} "
                f"sensitivity={unit.get('sensitivity_score')} "
                f"outlier={unit.get('outlier_risk')} "
                f"memory_cost={unit.get('memory_cost')} "
                f"resource_cost={unit.get('resource_cost_effective')} "
                f"parameter_cost={unit.get('parameter_cost')} "
                f"resource_cost_type={unit.get('resource_cost_type')} "
                f"keep_score={unit.get('keep_score')} "
                f"objective={unit.get('objective_score')} "
                f"reason={unit.get('reason')}\n"
            )
        stream.write("\n[All Candidate Units]\n")
        for unit in units:
            marker = "PRUNE" if unit.get("selected") else "KEEP"
            stream.write(
                f"{marker} "
                f"block={unit.get('block')} "
                f"type={unit.get('unit_type')} "
                f"idx={unit.get('unit_index')} "
                f"name={unit.get('unit_name')} "
                f"hessian={unit.get('hessian_score')} "
                f"sensitivity={unit.get('sensitivity_score')} "
                f"sensitivity_norm={unit.get('sensitivity_score_normalized')} "
                f"outlier={unit.get('outlier_risk')} "
                f"outlier_norm={unit.get('outlier_risk_normalized')} "
                f"memory_cost={unit.get('memory_cost')} "
                f"resource_cost={unit.get('resource_cost_effective')} "
                f"parameter_cost={unit.get('parameter_cost')} "
                f"resource_cost_type={unit.get('resource_cost_type')} "
                f"memory_norm={unit.get('memory_cost_normalized')} "
                f"keep_score={unit.get('keep_score')} "
                f"objective={unit.get('objective_score')}\n"
            )
    return path
