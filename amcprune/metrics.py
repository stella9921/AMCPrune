import json
import os
import time

import torch


def model_parameter_memory_mb(model):
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) / 1024**2


def cuda_memory_mb():
    if not torch.cuda.is_available():
        return {
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_peak_allocated_mb": 0.0,
        }
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "cuda_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def save_json(output_dir, name, payload):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return path


def run_id(prefix):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}__{stamp}"

