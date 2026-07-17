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
