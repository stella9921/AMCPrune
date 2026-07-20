import os


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _save_bar_plot(path, labels, values, title, ylabel, selected=None):
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    selected = set(selected or [])
    colors = ["tab:red" if index in selected else "tab:blue" for index in range(len(labels))]
    fig_width = max(8, min(18, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Block index")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_line_plot(path, labels, values, title, ylabel):
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    fig_width = max(8, min(18, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.plot(range(len(labels)), values, marker="o")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Block index")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_metric_bar(path, metrics, title, ylabel):
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4.0))
    ax.bar(labels, values, color="tab:green")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path



def _save_grouped_line_plot(path, labels, series, title, ylabel, selected=None, yscale=None):
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    selected = set(selected or [])
    fig_width = max(8, min(18, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    x_values = list(range(len(labels)))
    for name, values in series.items():
        ax.plot(x_values, values, marker="o", linewidth=1.8, label=name)
    for index in selected:
        ax.axvspan(index - 0.45, index + 0.45, color="tab:red", alpha=0.10)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Block index")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def plot_run_artifacts(
    *,
    output_dir,
    run_id,
    score_rows,
    selected_blocks,
    pruning_plan,
    baseline,
    pruned,
    preservation,
    memory_trace,
    timing_trace,
    unit_inventory=None,
    outlier_metrics=None,
    inference_metrics=None,
):
    plot_dir = _ensure_dir(os.path.join(output_dir, "plots"))
    saved = []
    labels = [str(row.get("block", index)) for index, row in enumerate(score_rows)]
    selected_positions = [
        index for index, row in enumerate(score_rows)
        if row.get("block") in set(selected_blocks)
    ]

    if score_rows:
        saved.append(_save_bar_plot(
            os.path.join(plot_dir, f"{run_id}__block_score.png"),
            labels,
            [float(row["score"]) for row in score_rows],
            "Block score by layer",
            "score",
            selected_positions,
        ))
        if "activation_abs_mean" in score_rows[0]:
            saved.append(_save_line_plot(
                os.path.join(plot_dir, f"{run_id}__activation_abs_mean.png"),
                labels,
                [float(row["activation_abs_mean"]) for row in score_rows],
                "Activation abs mean by block",
                "activation abs mean",
            ))
        if "weight_abs_mean" in score_rows[0]:
            saved.append(_save_line_plot(
                os.path.join(plot_dir, f"{run_id}__weight_abs_mean.png"),
                labels,
                [float(row["weight_abs_mean"]) for row in score_rows],
                "Weight abs mean by block",
                "weight abs mean",
            ))
        if "loss_delta" in score_rows[0]:
            saved.append(_save_bar_plot(
                os.path.join(plot_dir, f"{run_id}__loss_delta.png"),
                labels,
                [float(row["loss_delta"]) for row in score_rows],
                "Loss delta by skipped block",
                "loss delta",
                selected_positions,
            ))

    plan_units = pruning_plan.get("units", []) if pruning_plan else []
    if plan_units:
        saved.append(_save_bar_plot(
            os.path.join(plot_dir, f"{run_id}__pruning_selection.png"),
            [str(unit["unit_id"]) for unit in plan_units],
            [1.0 if unit["selected"] else 0.0 for unit in plan_units],
            "Pruned blocks",
            "selected",
            [index for index, unit in enumerate(plan_units) if unit["selected"]],
        ))

    saved.append(_save_metric_bar(
        os.path.join(plot_dir, f"{run_id}__perplexity.png"),
        {
            "baseline": float(baseline["perplexity"]),
            "pruned": float(pruned["perplexity"]),
        },
        "Perplexity before/after pruning",
        "perplexity",
    ))
    saved.append(_save_metric_bar(
        os.path.join(plot_dir, f"{run_id}__preservation.png"),
        {
            "hidden_cos": float(preservation["hidden_cosine_similarity"]),
            "logit_kl": float(preservation["logit_kl_divergence"]),
        },
        "Representation preservation",
        "metric value",
    ))

    if memory_trace:
        saved.append(_save_line_plot(
            os.path.join(plot_dir, f"{run_id}__memory_peak.png"),
            [row["stage"] for row in memory_trace],
            [float(row["cuda_peak_allocated_mb"]) for row in memory_trace],
            "CUDA peak allocated memory by stage",
            "MB",
        ))
    if timing_trace:
        saved.append(_save_bar_plot(
            os.path.join(plot_dir, f"{run_id}__timing_seconds.png"),
            [row["stage"] for row in timing_trace],
            [float(row["seconds"]) for row in timing_trace],
            "Stage runtime",
            "seconds",
        ))

    if unit_inventory:
        unit_labels = [str(row["block"]) for row in unit_inventory]
        unit_selected = [index for index, row in enumerate(unit_inventory) if row.get("selected_block")]
        metric_specs = [
            ("num_attention_heads", "Attention heads by block", "heads"),
            ("ffn_intermediate_dim", "FFN intermediate dim by block", "neurons"),
            ("attention_params", "Attention params by block", "parameters"),
            ("mlp_params", "MLP params by block", "parameters"),
            ("block_params", "Total block params", "parameters"),
        ]
        for key, title, ylabel in metric_specs:
            values = [row.get(key) for row in unit_inventory]
            if any(value is not None for value in values):
                saved.append(_save_bar_plot(
                    os.path.join(plot_dir, f"{run_id}__unit_{key}.png"),
                    unit_labels,
                    [float(value or 0.0) for value in values],
                    title,
                    ylabel,
                    unit_selected,
                ))

    if inference_metrics:
        dense_inference = inference_metrics.get("dense", {})
        pruned_inference = inference_metrics.get("pruned", {})
        saved.append(_save_metric_bar(
            os.path.join(plot_dir, f"{run_id}__inference_ttft.png"),
            {
                "dense": float(dense_inference.get("ttft_seconds") or 0.0),
                "pruned": float(pruned_inference.get("ttft_seconds") or 0.0),
            },
            "Time to first token",
            "seconds",
        ))
        saved.append(_save_metric_bar(
            os.path.join(plot_dir, f"{run_id}__inference_tps.png"),
            {
                "dense": float(dense_inference.get("tokens_per_second") or 0.0),
                "pruned": float(pruned_inference.get("tokens_per_second") or 0.0),
            },
            "Generation throughput",
            "tokens/sec",
        ))
        saved.append(_save_metric_bar(
            os.path.join(plot_dir, f"{run_id}__inference_peak_vram.png"),
            {
                "dense": float(dense_inference.get("peak_vram_mb") or 0.0),
                "pruned": float(pruned_inference.get("peak_vram_mb") or 0.0),
            },
            "Inference peak VRAM",
            "MB",
        ))

    if outlier_metrics:
        outlier_labels = [str(row["block"]) for row in outlier_metrics]
        outlier_selected = [index for index, row in enumerate(outlier_metrics) if row.get("selected_block")]
        metric_specs = [
            ("second_moment_mean", "OATS-style second moment mean by block", "E[x^2]"),
            ("second_moment_top1pct_mean", "Top-1% second moment mean by block", "E[x^2]"),
            ("second_moment_q99", "Second moment Q99 by block", "E[x^2]"),
            ("outlier_ratio", "Second moment outlier ratio by block", "ratio"),
        ]
        for key, title, ylabel in metric_specs:
            values = [row.get(key) for row in outlier_metrics]
            if any(value is not None for value in values):
                saved.append(_save_bar_plot(
                    os.path.join(plot_dir, f"{run_id}__outlier_{key}.png"),
                    outlier_labels,
                    [float(value or 0.0) for value in values],
                    title,
                    ylabel,
                    outlier_selected,
                ))

        saved.append(_save_grouped_line_plot(
            os.path.join(plot_dir, f"{run_id}__outlier_second_moment_profile.png"),
            outlier_labels,
            {
                "mean": [float(row.get("second_moment_mean") or 0.0) for row in outlier_metrics],
                "Q95": [float(row.get("second_moment_q95") or 0.0) for row in outlier_metrics],
                "Q99": [float(row.get("second_moment_q99") or 0.0) for row in outlier_metrics],
                "max": [float(row.get("second_moment_max") or 0.0) for row in outlier_metrics],
            },
            "Second moment outlier profile by block",
            "E[x^2]",
            outlier_selected,
            yscale="log",
        ))

        eps = 1e-12
        saved.append(_save_grouped_line_plot(
            os.path.join(plot_dir, f"{run_id}__outlier_amplification_ratio.png"),
            outlier_labels,
            {
                "Q99/mean": [
                    float(row.get("second_moment_q99") or 0.0) /
                    max(float(row.get("second_moment_mean") or 0.0), eps)
                    for row in outlier_metrics
                ],
                "max/mean": [
                    float(row.get("second_moment_max") or 0.0) /
                    max(float(row.get("second_moment_mean") or 0.0), eps)
                    for row in outlier_metrics
                ],
                "top1%/mean": [
                    float(row.get("second_moment_top1pct_mean") or 0.0) /
                    max(float(row.get("second_moment_mean") or 0.0), eps)
                    for row in outlier_metrics
                ],
            },
            "Activation outlier amplification by block",
            "ratio to mean",
            outlier_selected,
        ))
    return [path for path in saved if path]

