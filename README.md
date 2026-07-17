# AMCPrune

AMCPrune is a prototype repository for memory-aware structured pruning of
pretrained small language models.

Initial goal:

1. Load a pretrained causal language model.
2. Load a small text calibration/evaluation dataset.
3. Measure baseline perplexity.
4. Apply a simple block-level pruning mask.
5. Measure pruned perplexity and memory metrics.

Text data preprocessing follows the HuggingFace causal language modeling example
style: load a dataset from the Hub, tokenize text with `AutoTokenizer`, concatenate
tokens, and split them into fixed-length `block_size` chunks for causal LM loss.

The first prototype intentionally keeps the scoring simple so that the LLM
pruning pipeline can run end to end before adding Fisher/Hessian refinement.

## Smoke Test

```bash
python main.py \
  --model sshleifer/tiny-gpt2 \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 32 \
  --seq-len 128 \
  --pruning-ratio 0.25 \
  --score block_index \
  --output-dir exp/smoke_tiny_gpt2
```

## Preset-Based Experiment

Most experiment settings can be stored in `configs/strategy/*.yaml`. For example,
`configs/strategy/gpt2_loss_delta.yaml` runs the current block-skip prototype with
loss-delta scoring:

```bash
python main.py --strategy gpt2_loss_delta
```

Each run creates a timestamped directory under `exp/runs/` unless `--output-dir`
is provided. The run directory stores:

- `resolved_config.json`: final config after YAML and CLI overrides
- `command.txt`: exact command used for the run
- `*_run.log`: console log copied to file
- `block_scores.csv` / `block_scores.json`: block score and selected block records
- `timing_trace.json`: per-stage elapsed time
- `memory_trace.json`: per-stage CUDA memory trace
- `result.json`: summary metrics, selected blocks, preservation metrics, and paths
- `pruned_model/amcprune_pruning_config.json`: base model and pruning mask config

CLI arguments override YAML values when explicitly provided, so short commands can
use presets while still allowing quick ablations such as:

```bash
python main.py --strategy gpt2_loss_delta --pruning-ratio 0.2 --seed 123
```

## Activation-Weight Block Scoring

The current cheap structured score can rank transformer blocks by:

```text
score = block activation abs mean x block weight abs mean
```

Run once and store block scores:

```bash
python main.py --strategy gpt2_activation_weight
```

Reuse the saved SnapViT-style `.pt` score file with a different pruning ratio without recomputing scores:

```bash
python main.py \
  --strategy gpt2_activation_weight \
  --score-cache exp/runs/<run_id>/importance-scores.pt \
  --pruning-ratio 0.2
```

Compare dense and pruned downstream task performance:

```bash
python scripts/evaluate_dense_vs_pruned.py \
  --pruning-config exp/runs/<run_id>/pruned_model/amcprune_pruning_config.json \
  --tasks hellaswag,piqa,arc_easy \
  --device cuda:0 \
  --batch-size 4 \
  --limit 100 \
  --output-path exp/lm_eval/<run_id>__dense_vs_pruned.json
```

