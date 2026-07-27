import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name, device=None, dtype="auto"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device)
    torch_dtype = "auto"
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32

    print(f"[Model Load] tokenizer start: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[Model Load] tokenizer done", flush=True)

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if device_obj.type == "cuda":
        load_kwargs["device_map"] = {"": str(device_obj)}

    print(
        f"[Model Load] weights start: dtype={dtype} device={device_obj}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    print("[Model Load] weights done", flush=True)
    if device_obj.type != "cuda":
        print(f"[Model Load] moving model to {device_obj}", flush=True)
        model.to(device_obj)
        print("[Model Load] move done", flush=True)
    model.eval()
    return model, tokenizer, device_obj


def get_transformer_blocks(model):
    candidates = [
        "transformer.h",
        "model.layers",
        "gpt_neox.layers",
        "decoder.layers",
    ]
    for path in candidates:
        current = model
        found = True
        for part in path.split("."):
            if not hasattr(current, part):
                found = False
                break
            current = getattr(current, part)
        if found:
            return list(current), path
    raise ValueError("Could not locate transformer blocks for this model.")
