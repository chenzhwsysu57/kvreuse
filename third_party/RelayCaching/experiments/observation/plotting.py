"""Plotting for observation experiments."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

_GRID_STYLE = {"color": "#E0E0E0", "linestyle": "-", "linewidth": 0.8}

STYLES = {
    "RelayCaching": {
        "c": "#E47B7E",
        "m": "o",
        "l": "-",
        "label": "RelayCaching",
    },
    "KVCOMM": {
        "c": "#A8B4C1",
        "m": "s",
        "l": "--",
        "label": "KVCOMM",
    },
    "EPIC": {
        "c": "#DCA68C",
        "m": "^",
        "l": "-.",
        "label": "EPIC",
    },
    "CacheBlend": {
        "c": "#91B396",
        "m": "D",
        "l": ":",
        "label": "CacheBlend",
    },
    "FULL": {
        "c": "#A8B4C1",
        "m": "*",
        "l": "-",
        "label": "Full Prefill",
    },
}

METRIC_COLORS = [
    STYLES["FULL"]["c"],
    STYLES["CacheBlend"]["c"],
    STYLES["EPIC"]["c"],
    STYLES["RelayCaching"]["c"],
]

MACRO_BAR_COLORS = [
    STYLES["EPIC"]["c"],
    STYLES["RelayCaching"]["c"],
    STYLES["FULL"]["c"],
    STYLES["CacheBlend"]["c"],
]

# Inclusive relay band (L_start, L_end) — layer shading, token averaging, recovery curves.
MODEL_LAYER_BAND = {
    "Mistral-7B-Instruct-v0.3": (3, 20),
    "Llama-3.1-8B-Instruct": (1, 18),
    "Qwen2.5-Coder-7B-Instruct": (3, 22),
    "DeepSeek-R1-Distill-Qwen-32B": (4, 50),
    "Qwen3-0.6B": (2, 19),
    "Qwen3-30B-A3B": (1, 43),
}

OBSERVATION_EXPERIMENT_MODEL = {
    "observation_results_mistral_wiki": "Mistral-7B-Instruct-v0.3",
    "observation_results_deepseek_qwen_32b": "DeepSeek-R1-Distill-Qwen-32B",
    "observation_results_llama3.1_8b": "Llama-3.1-8B-Instruct",
    "observation_results_qwen2.5_coder_7b": "Qwen2.5-Coder-7B-Instruct",
    "observation_results_qwen3_0.6b": "Qwen3-0.6B",
    "observation_results_qwen3_30B": "Qwen3-30B-A3B",
}

_DEFAULT_LAYER_BAND = (3, 20)

MODEL_LAYER_REGIONS = MODEL_LAYER_BAND
MODEL_AVG_LAYER_RANGES = MODEL_LAYER_BAND
MODEL_TOKEN_AVG_LAYER_RANGES = MODEL_LAYER_BAND


def _lookup_layer_band(model_name: str = None, experiment_dir: str = None):
    if experiment_dir:
        model = OBSERVATION_EXPERIMENT_MODEL.get(Path(experiment_dir).name)
        if model and model in MODEL_LAYER_BAND:
            return MODEL_LAYER_BAND[model]
    if model_name:
        short = model_name.split("/")[-1]
        if short in MODEL_LAYER_BAND:
            return MODEL_LAYER_BAND[short]
    return None


def resolve_layer_bounds(
    *,
    model_name: str = None,
    experiment_dir: str = None,
    L_start: int = None,
    L_end: int = None,
    shallow_end: int = None,
    deep_start: int = None,
) -> tuple:
    """Resolve inclusive (L_start, L_end). Used by all layer-index plots."""
    if L_start is not None and L_end is not None:
        return int(L_start), int(L_end)

    if shallow_end is not None:
        ls = int(shallow_end)
        if deep_start is not None:
            ds = int(deep_start)
            return (ls, ds - 1) if ds > ls else (ls, ds)

    band = _lookup_layer_band(model_name=model_name, experiment_dir=experiment_dir)
    if band is not None:
        return band

    return _DEFAULT_LAYER_BAND


def get_token_avg_layer_range(model_name: str) -> tuple:
    """Same band as resolve_layer_bounds (token-wise layer averaging)."""
    band = _lookup_layer_band(model_name=model_name)
    if band is None:
        raise ValueError(
            f"Unknown model_name for layer band: {model_name!r}. "
            f"Expected one of: {sorted(MODEL_LAYER_BAND.keys())}"
        )
    return band


def _apply_grid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which="both", zorder=0, **_GRID_STYLE)


_MACRO_METRICS = ('key_cosine', 'value_cosine', 'key_norm', 'value_norm')


def _macro_block_for_fig1(macro_similarities: dict, primary_key: str) -> dict:
    """Read one macro-similarity column (e.g. decoding_vs_prefill, random_vs_prefill)."""
    block = macro_similarities.get(primary_key) or {}
    if not isinstance(block, dict):
        block = {}
    return {m: float(block.get(m, 0.0) or 0.0) for m in _MACRO_METRICS}


def plot_macro_similarity(
    macro_similarities: dict = None,
    decoding_vs_prefill_cos: float = None,
    decoding_vs_prefill_norm: float = None,
    decoding_vs_random_cos: float = None,
    decoding_vs_random_norm: float = None,
    independent_vs_prefill_cos: float = None,
    independent_vs_prefill_norm: float = None,
    save_path: str = None,
):
    """
    Compare similarity of three KV sources against Prefill KV across
    all four similarity metrics (Key Cosine, Value Cosine, Key Norm, Value Norm):
    - Decoding KV vs Prefill KV (our method)
    - Random KV vs Prefill KV (random baseline)
    - Independent KV vs Prefill KV (independent prefill baseline)

    Args:
        macro_similarities: dict with macro similarity data containing:
            - 'decoding_vs_prefill': vs Prefill (Decoding KV column)
            - 'random_vs_prefill': vs Prefill (Random KV column)
            - 'independent_vs_prefill': vs Prefill (Independent KV column)
        save_path: path to save the figure
    """
    categories = ['Decoding KV', 'Random KV', 'Independent KV']

    if macro_similarities is not None:
        dvp = _macro_block_for_fig1(macro_similarities, 'decoding_vs_prefill')
        rvp = _macro_block_for_fig1(macro_similarities, 'random_vs_prefill')
        ivp = _macro_block_for_fig1(macro_similarities, 'independent_vs_prefill')

        bar_data = {
            'Key Norm': [dvp['key_norm'], rvp['key_norm'], ivp['key_norm']],
            'Value Norm': [dvp['value_norm'], rvp['value_norm'], ivp['value_norm']],
            'Key Cosine': [dvp['key_cosine'], rvp['key_cosine'], ivp['key_cosine']],
            'Value Cosine': [dvp['value_cosine'], rvp['value_cosine'], ivp['value_cosine']],
        }
    else:
        bar_data = {
            'Key Norm': [decoding_vs_prefill_norm or 0, decoding_vs_random_norm or 0, independent_vs_prefill_norm or 0],
            'Value Norm': [decoding_vs_prefill_norm or 0, decoding_vs_random_norm or 0, independent_vs_prefill_norm or 0],
            'Key Cosine': [decoding_vs_prefill_cos or 0, decoding_vs_random_cos or 0, independent_vs_prefill_cos or 0],
            'Value Cosine': [decoding_vs_prefill_cos or 0, decoding_vs_random_cos or 0, independent_vs_prefill_cos or 0],
        }

    fig, ax = plt.subplots(figsize=(7.6, 4), dpi=200)

    x = np.arange(len(categories))
    width = 0.18
    BAR_ALPHA = 0.6

    bar_specs = [
        ('Key Norm', -1.5, MACRO_BAR_COLORS[0]),
        ('Value Norm', -0.5, MACRO_BAR_COLORS[1]),
        ('Key Cosine', 0.5, MACRO_BAR_COLORS[2]),
        ('Value Cosine', 1.5, MACRO_BAR_COLORS[3]),
    ]
    all_rects = []
    for label, offset_mult, color in bar_specs:
        xpos = x + offset_mult * width
        values = bar_data[label]
        ax.bar(xpos, values, width, color='white', edgecolor='none', linewidth=0.0, zorder=2)
        rects = ax.bar(
            xpos, values, width, label=label,
            color=color, edgecolor='black', linewidth=0.8, alpha=BAR_ALPHA, zorder=3,
        )
        all_rects.append(rects)

    for rects in all_rects:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                )

    ax.set_ylabel('Similarity', fontweight='bold', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.15])

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    for _t in ax.get_xticklabels():
        _t.set_fontweight('bold')
    for _t in ax.get_yticklabels():
        _t.set_fontweight('normal')

    _apply_grid(ax)
    ax.set_axisbelow(True)
    for _gl in (ax.get_xgridlines() + ax.get_ygridlines()):
        _gl.set_zorder(0)

    ax.set_title('')
    fig.suptitle('')

    ax.legend(loc='upper right', fontsize=14, ncol=2, framealpha=1.0)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        png_path = str(Path(save_path).with_suffix('.png'))
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"   Macro similarity plot saved to: {save_path} and {png_path}")

    plt.close()


def _shade_layer_regions(
    ax,
    num_layers: int,
    L_start: int,
    L_end: int,
    c_shallow,
    c_middle,
    c_deep,
    region_labels=(
        "Shallow Layers",
        "Middle Layers",
        "Deep Layers",
    ),
):
    """
    Shade shallow / middle / deep behind layer-index curves.

    Layer k is drawn at x=k. Only the middle band [L_start, L_end] is widened by ±0.5
    to [L_start - 0.5, L_end + 0.5]. Shallow/deep use integer layer bounds at the seam.
    """
    if L_start > 0:
        ax.axvspan(
            0, L_start - 0.5, color=c_shallow, alpha=0.2, label=region_labels[0],
        )
    if L_start <= L_end:
        ax.axvspan(
            L_start - 0.5, L_end + 0.5, color=c_middle, alpha=0.2, label=region_labels[1],
        )
    if L_end + 1 < num_layers:
        ax.axvspan(
            L_end + 0.5, num_layers - 1, color=c_deep, alpha=0.2, label=region_labels[2],
        )


def plot_layer_wise_deviation(
    layer_similarities: list,
    model_name: str = "Model",
    L_start: int = None,
    L_end: int = None,
    shallow_end: int = None,
    deep_start: int = None,
    middle_left: int = None,
    middle_right: int = None,
    save_path: str = None,
):
    """
    Per-layer similarity profile (U-shape: high at shallow/deep, low in middle).

    Args:
        layer_similarities: list of dicts with 'key_cosine', 'value_cosine', 'key_norm', 'value_norm'
        model_name: model name (for title); used to look up MODEL_LAYER_BAND when bounds omitted
        L_start, L_end: inclusive middle / relay band (preferred)
        shallow_end, deep_start: deprecated aliases (see resolve_layer_bounds)
        save_path: path to save the figure
    """
    L_start, L_end = resolve_layer_bounds(
        model_name=model_name,
        L_start=L_start,
        L_end=L_end,
        shallow_end=shallow_end,
        deep_start=deep_start,
    )
    num_layers = len(layer_similarities)
    layers = np.arange(num_layers)

    def get_val(d, base_name):
        return d.get(base_name) or d.get(f"{base_name}_similarity") or 0

    key_cos = [get_val(d, 'key_cosine') for d in layer_similarities]
    val_cos = [get_val(d, 'value_cosine') for d in layer_similarities]
    key_norm = [get_val(d, 'key_norm') for d in layer_similarities]
    val_norm = [get_val(d, 'value_norm') for d in layer_similarities]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)

    c_shallow = STYLES["CacheBlend"]["c"]
    c_middle = STYLES["RelayCaching"]["c"]
    c_deep = STYLES["FULL"]["c"]
    _ = middle_left, middle_right
    _shade_layer_regions(ax, num_layers, L_start, L_end, c_shallow, c_middle, c_deep)

    ls = {'linewidth': 2.4, 'markersize': 7.0, 'markerfacecolor': 'white', 'markeredgewidth': 1.4}

    ax.plot(layers, key_cos, label='Key Cosine', color=STYLES["FULL"]["c"], marker='s', **ls)
    ax.plot(layers, val_cos, label='Value Cosine', color=STYLES["CacheBlend"]["c"], marker='^', **ls)
    ax.plot(layers, key_norm, label='Key Norm', color=STYLES["EPIC"]["c"], marker='d', linestyle='--', **ls)
    ax.plot(layers, val_norm, label='Value Norm', color=STYLES["RelayCaching"]["c"], marker='v', linestyle='--', **ls)

    ax.set_xlabel("Layer Index", fontsize=18, fontweight='bold')
    ax.set_ylabel("Similarity", fontsize=18, fontweight='bold')

    ax.set_xlim(-0.5, num_layers - 0.5)

    tick_step = 1
    if num_layers >= 48:
        tick_step = 2
    ax.set_xticks(np.arange(0, num_layers, tick_step))

    all_sims = np.array(key_cos + val_cos)
    ax.set_ylim(min(all_sims[all_sims > 0]) * 0.95, 1.01)

    _apply_grid(ax)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(loc='lower left', ncol=1, frameon=True, fontsize=14, facecolor='white', framealpha=1.0)

    ax.set_title("")
    fig.suptitle("")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        png_path = str(Path(save_path).with_suffix('.png'))
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"   Layer-wise deviation plot saved to: {save_path} and {png_path}")

    plt.close()


def plot_layer_profile_selection(
    layer_similarities: list,
    L_start: int,
    L_end: int,
    L_det: int = None,
    token_deviations: dict = None,
    correlations: dict = None,
    second_order_diffs: dict = None,
    metric: str = "value_cosine",
    lower_threshold: float = 0.99,
    topk_ratio: float = 0.2,
    model_name: str = "Model",
    save_path: str = None,
):
    """
    Two-panel layer profiling figure: U-shape similarity + rho/alpha for L_det.

    Pass ``correlations`` / ``second_order_diffs`` from the profiler, or derive from
    ``token_deviations``.
    """
    try:
        from experiments.observation.layer_selection import (
            compute_spearman_correlations,
            compute_second_order_diffs,
        )
    except ImportError:
        from layer_selection import (
            compute_spearman_correlations,
            compute_second_order_diffs,
        )

    num_layers = len(layer_similarities)
    layers = np.arange(num_layers)

    def get_val(d, base_name):
        return d.get(base_name) or d.get(f"{base_name}_similarity") or 0.0

    metric_key = metric if metric in ("key_cosine", "value_cosine", "key_norm", "value_norm") else "value_cosine"
    sims = np.asarray([get_val(d, metric_key) for d in layer_similarities], dtype=float)

    has_det_panel = (
        (correlations is not None and len(correlations) >= 2)
        or (token_deviations is not None and len(token_deviations) >= 3)
    )
    nrows = 2 if has_det_panel else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4.5 * nrows), dpi=200, sharex=False)
    if nrows == 1:
        axes = [axes]

    ax0 = axes[0]
    c_shallow = STYLES["CacheBlend"]["c"]
    c_middle = STYLES["RelayCaching"]["c"]
    c_deep = STYLES["FULL"]["c"]

    _shade_layer_regions(
        ax0, num_layers, L_start, L_end, c_shallow, c_middle, c_deep,
        region_labels=("Shallow (high sim)", "Middle (relay band)", "Deep (high sim)"),
    )

    ls = {"linewidth": 2.4, "markersize": 7.0, "markerfacecolor": "white", "markeredgewidth": 1.4}
    ax0.plot(
        layers, sims,
        label=f"Dec vs Prefill ({metric_key})",
        color=STYLES["RelayCaching"]["c"],
        marker="o",
        **ls,
    )
    ax0.axhline(lower_threshold, color="gray", linestyle=":", linewidth=1.8, label=f"τ = {lower_threshold}")

    ax0.axvline(L_start, color=c_middle, linestyle="--", linewidth=2.0, label=f"$L_{{start}}$={L_start}")
    ax0.axvline(L_end, color=c_middle, linestyle="-.", linewidth=2.0, label=f"$L_{{end}}$={L_end}")
    if L_det is not None:
        ax0.axvline(L_det, color=STYLES["EPIC"]["c"], linestyle="-", linewidth=2.2, label=f"$L_{{det}}$={L_det}")

    ax0.set_ylabel("Similarity", fontsize=16, fontweight="bold")
    ax0.set_xlim(-0.5, num_layers - 0.5)
    tick_step = 2 if num_layers >= 48 else 1
    ax0.set_xticks(np.arange(0, num_layers, tick_step))
    if np.any(sims > 0):
        ax0.set_ylim(float(np.min(sims[sims > 0]) * 0.95), 1.01)
    _apply_grid(ax0)
    ax0.tick_params(axis="both", labelsize=12)
    ax0.legend(loc="lower left", fontsize=11, frameon=True, facecolor="white")
    ax0.set_title("(a) U-shape profile → $L_{start}$, $L_{end}$", fontsize=14, fontweight="bold", loc="left")

    if has_det_panel:
        if correlations is None and token_deviations is not None:
            correlations = compute_spearman_correlations(
                token_deviations, topk_ratio=topk_ratio,
            )
        correlations = correlations or {}
        if second_order_diffs is None and correlations:
            second_order_diffs = compute_second_order_diffs(correlations)
        second_order_diffs = second_order_diffs or {}
        ax1 = axes[1]

        rho_layers = sorted(correlations.keys())
        rho_vals = [correlations[l] for l in rho_layers]
        ax1.plot(
            rho_layers, rho_vals,
            color=STYLES["FULL"]["c"],
            marker="s",
            linewidth=2.2,
            markersize=6,
            markerfacecolor="white",
            label=r"$\rho_L$ (pair $L$–$L{+}1$, at $L$)",
        )

        ax1b = ax1.twinx()
        alpha_layers = sorted(second_order_diffs.keys())
        alpha_vals = [second_order_diffs[l] for l in alpha_layers]
        ax1b.plot(
            alpha_layers, alpha_vals,
            color=STYLES["EPIC"]["c"],
            marker="d",
            linewidth=2.0,
            linestyle="--",
            markersize=6,
            markerfacecolor="white",
            label=r"$\alpha_L$ (2nd-order diff at pair start $L$)",
        )
        ax1b.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
        ax1b.set_ylabel(r"$\alpha_\ell$", fontsize=16, fontweight="bold", color=STYLES["EPIC"]["c"])
        ax1b.tick_params(axis="y", labelcolor=STYLES["EPIC"]["c"], labelsize=12)

        if L_start <= L_end:
            ax1.axvspan(L_start - 0.5, L_end + 0.5, color=c_middle, alpha=0.12)
        if L_det is not None:
            ax1.axvline(L_det, color=STYLES["EPIC"]["c"], linestyle="-", linewidth=2.2, label=f"$L_{{det}}$={L_det}")

        cand = sorted(l for l in second_order_diffs if l <= L_end)
        cross_det = None
        for i in range(len(cand) - 1):
            if second_order_diffs[cand[i]] >= 0 and second_order_diffs[cand[i + 1]] < 0:
                cross_det = cand[i + 1]
                break
        if cross_det is not None:
            if cross_det <= L_start:
                cross_det = L_start + 1
            ax1.axvline(
                cross_det, color="orange", linestyle=":", linewidth=1.8, alpha=0.9,
                label="α sign change (clamped)",
            )

        ax1.set_xlabel("Pair start layer $L$ ($L$–$L{+}1$)", fontsize=16, fontweight="bold")
        ax1.set_ylabel(r"$\rho_L$", fontsize=16, fontweight="bold")
        ax1.set_xlim(-0.5, num_layers - 0.5)
        ax1.set_xticks(np.arange(0, num_layers, tick_step))
        ax1.set_ylim(0.0, 1.05)
        _apply_grid(ax1)
        ax1.tick_params(axis="x", labelsize=12)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11, frameon=True)
        ax1.set_title("(b) Token-rank stability → $L_{det}$", fontsize=14, fontweight="bold", loc="left")
    else:
        ax0.set_xlabel("Layer Index", fontsize=16, fontweight="bold")

    if has_det_panel:
        axes[-1].set_xlabel("Layer Index", fontsize=16, fontweight="bold")

    fig.suptitle(f"Layer profile selection — {model_name}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        png_path = str(Path(save_path).with_suffix(".png"))
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"   Layer profile selection plot saved to: {save_path} and {png_path}")

    plt.close()


def plot_relative_recovery(
    lower_sims,
    upper_sims,
    model_name: str = None,
    L_start: int = None,
    L_end: int = None,
    shallow_end: int = None,
    deep_start: int = None,
    save_path: str = None,
):
    """Lower/upper bound relative recovery (%), normalized to global min/max."""
    L_start, L_end = resolve_layer_bounds(
        model_name=model_name,
        L_start=L_start,
        L_end=L_end,
        shallow_end=shallow_end,
        deep_start=deep_start,
    )

    lower_sims = np.asarray(lower_sims, dtype=float)
    upper_sims = np.asarray(upper_sims, dtype=float)
    num_layers = len(lower_sims)
    layers = np.arange(num_layers)

    baseline_sim = float(np.min([np.min(lower_sims), np.min(upper_sims)]))
    top_sim = float(np.max([np.max(lower_sims), np.max(upper_sims)]))
    denom = top_sim - baseline_sim
    if denom <= 1e-12:
        lower_rec = np.zeros_like(lower_sims, dtype=float)
        upper_rec = np.zeros_like(upper_sims, dtype=float)
    else:
        lower_rec = (lower_sims - baseline_sim) / denom * 100
        upper_rec = (upper_sims - baseline_sim) / denom * 100

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)

    c_shallow = STYLES["CacheBlend"]["c"]
    c_middle = STYLES["RelayCaching"]["c"]
    c_deep = STYLES["FULL"]["c"]
    _shade_layer_regions(ax, num_layers, L_start, L_end, c_shallow, c_middle, c_deep)

    c_lower = STYLES["CacheBlend"]["c"]
    c_upper = STYLES["RelayCaching"]["c"]
    ms = {'markersize': 6.0, 'markerfacecolor': 'white', 'markeredgewidth': 1.4}

    ax.plot(
        layers, lower_rec, linewidth=2.4, color=c_lower, marker='^',
        markeredgecolor=c_lower, label='Lower Bound Recovery (Rec[0, L_end])', **ms,
    )
    ax.plot(
        layers, upper_rec, linewidth=2.4, color=c_upper, marker='s',
        markeredgecolor=c_upper, label='Upper Bound Recovery (Rec[L_start, Max])', **ms,
    )

    ax.set_xlabel("Layer Index", fontweight='bold', fontsize=18)
    ax.set_ylabel("Relative Recovery (%)", fontweight='bold', fontsize=18)
    ax.set_xlim(-0.5, num_layers - 0.5)

    tick_step = 1
    if num_layers >= 48:
        tick_step = 2
    ax.set_xticks(np.arange(0, num_layers, tick_step))
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.legend(
        loc='center right', bbox_to_anchor=(0.98, 0.5),
        framealpha=1.0, fontsize=14, ncol=1,
    )
    _apply_grid(ax)
    ax.set_title('')
    fig.suptitle('')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        png_path = str(Path(save_path).with_suffix('.png'))
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"   Relative recovery plot saved to: {save_path} and {png_path}")

    plt.close()


def _aggregate_token_metrics_by_layers(
    token_sims_by_layer: dict,
    model_name: str = None,
    avg_layer_range=None,
) -> dict:
    """Average per-token metrics over a layer index range."""
    metrics = ['key_cosine', 'value_cosine', 'key_norm', 'value_norm']
    all_layer_ints = sorted(int(k) for k in token_sims_by_layer.keys())

    if avg_layer_range is not None:
        s, e = avg_layer_range
    elif model_name:
        s, e = get_token_avg_layer_range(model_name)
    else:
        s, e = all_layer_ints[0], all_layer_ints[-1]

    layers_used = [i for i in all_layer_ints if s <= i <= e]
    if not layers_used:
        raise ValueError(
            f"No layers in range ({s}, {e}) for model {model_name!r}. "
            f"Available: {all_layer_ints[:5]}...{all_layer_ints[-5:]}"
        )

    stacked = {m: [] for m in metrics}
    for li in layers_used:
        layer_data = token_sims_by_layer.get(li, token_sims_by_layer.get(str(li), {}))
        for m in metrics:
            stacked[m].append(np.asarray(layer_data[m], dtype=float))
    return {m: np.mean(stacked[m], axis=0) for m in metrics}


def plot_token_wise_all_metrics(
    token_similarities: dict = None,
    token_similarities_by_layer: dict = None,
    model_name: str = None,
    avg_layer_range=None,
    layer_idx=0,
    prefix_len: int = 0,
    suffix_start: int = None,
    num_layers: int = None,
    save_path: str = None,
):
    """
    Token-level curves for key/value cosine and norm.

    Provide ``token_similarities`` or ``token_similarities_by_layer`` (with optional
    ``model_name`` / ``avg_layer_range`` for layer averaging).
    """
    if token_similarities_by_layer is not None:
        token_similarities = _aggregate_token_metrics_by_layers(
            token_similarities_by_layer,
            model_name=model_name,
            avg_layer_range=avg_layer_range,
        )
    elif token_similarities is None:
        raise ValueError("Provide token_similarities or token_similarities_by_layer")

    metrics = ['key_cosine', 'value_cosine', 'key_norm', 'value_norm']

    key_cosine = np.array(token_similarities.get('key_cosine', []), dtype=float)
    value_cosine = np.array(token_similarities.get('value_cosine', []), dtype=float)
    key_norm = np.array(token_similarities.get('key_norm', []), dtype=float)
    value_norm = np.array(token_similarities.get('value_norm', []), dtype=float)

    seq_len = len(key_cosine)
    token_indices = np.arange(seq_len)

    deviation_percentile = 20
    deviation_threshold = float(np.percentile(value_cosine, deviation_percentile))
    high_deviation_mask = value_cosine <= deviation_threshold

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)

    c_shallow = STYLES["CacheBlend"]["c"]
    if prefix_len > 0:
        prefix_right = min(prefix_len + 0.5, seq_len - 0.5)
        ax.axvspan(-0.5, prefix_right, color=c_shallow, alpha=0.2, label='Prefix Tokens')

    if suffix_start is not None and suffix_start < seq_len:
        ax.axvspan(suffix_start - 0.5, seq_len - 0.5, alpha=0.15, color='#80D4FF', label='Suffix Tokens')

    styles = [
        ('key_cosine', STYLES["FULL"]["c"], 's', '-', 'Key Cosine'),
        ('value_cosine', STYLES["CacheBlend"]["c"], '^', '-', 'Value Cosine'),
        ('key_norm', STYLES["EPIC"]["c"], 'd', '--', 'Key Norm'),
        ('value_norm', STYLES["RelayCaching"]["c"], 'v', '--', 'Value Norm'),
    ]
    for m_key, color, marker, ls, label in styles:
        ax.plot(
            token_indices,
            token_similarities[m_key],
            linewidth=2.4,
            linestyle=ls,
            color=color,
            marker=marker,
            markersize=6,
            markerfacecolor='white',
            markeredgecolor=color,
            markeredgewidth=1.2,
            markevery=max(1, seq_len // 22),
            label=label,
        )

    if np.any(high_deviation_mask):
        ax.scatter(
            token_indices[high_deviation_mask],
            value_cosine[high_deviation_mask],
            marker='x',
            s=45,
            color=STYLES["RelayCaching"]["c"],
            linewidths=1.3,
            zorder=5,
            label=f'High Deviation (Bottom {deviation_percentile}%)',
        )

    ax.set_xlabel('Token Position', fontweight='bold', fontsize=18)
    ax.set_ylabel('Similarity Score', fontweight='bold', fontsize=18)
    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim([max(0, np.min(value_cosine) * 0.9), 1.02])

    _apply_grid(ax)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(loc='lower right', ncol=2, frameon=True, fontsize=14, facecolor='white', framealpha=1.0)

    ax.set_title('')
    fig.suptitle('')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        png_path = str(Path(save_path).with_suffix('.png'))
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"   Token-wise all metrics plot saved to: {save_path} and {png_path}")

    plt.close()


def plot_inter_layer_correlation(
    layer_deviations: dict,
    reference_layer: int = None,
    save_path: str = None,
):
    """
    Adjacent-layer Spearman rho on per-token value_cosine vectors.

    Args:
        layer_deviations: {layer_idx: 1d array} per-layer token similarities
        reference_layer: deprecated, ignored
        save_path: path to save the figure
    """
    _ = reference_layer

    layer_indices = sorted(int(k) for k in layer_deviations.keys())
    if len(layer_indices) < 2:
        print("   Not enough layers for inter-layer correlation plot.")
        return

    pair_starts = []
    correlations = []
    for i in range(len(layer_indices) - 1):
        curr_idx = layer_indices[i]
        next_idx = layer_indices[i + 1]
        curr_sims = np.asarray(layer_deviations[curr_idx], dtype=float)
        next_sims = np.asarray(layer_deviations[next_idx], dtype=float)
        corr, _ = spearmanr(curr_sims, next_sims)
        pair_starts.append(curr_idx)
        correlations.append(float(corr) if not np.isnan(corr) else 0.0)

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
    x_positions = np.asarray(pair_starts, dtype=int)
    line_color = STYLES["FULL"]["c"]

    ax.plot(
        x_positions,
        correlations,
        linewidth=2.4,
        color=line_color,
        marker='o',
        markersize=6.0,
        markerfacecolor='white',
        markeredgecolor=line_color,
        markeredgewidth=1.4,
        label='Adjacent Layer Correlation',
    )

    n = len(correlations)
    tick_step = 1
    if len(layer_indices) >= 48:
        tick_step = 2
    tick_idx = list(range(0, n, tick_step))
    ax.set_xticks([x_positions[i] for i in tick_idx])
    ax.set_xticklabels([str(pair_starts[i]) for i in tick_idx], rotation=0, ha='center')
    num_layers = max(layer_indices) + 1
    ax.set_xlim(-0.5, num_layers - 0.5)

    ax.set_xlabel('Pair start layer L (L–L+1)', fontweight='bold', fontsize=18)
    ax.set_ylabel('Spearman Rank Correlation', fontweight='bold', fontsize=18)
    ax.set_title('')
    fig.suptitle('')

    _apply_grid(ax)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0.0, 1.05)

    for _t in ax.get_xticklabels():
        _t.set_fontweight('bold')
    for _t in ax.get_yticklabels():
        _t.set_fontweight('normal')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        png_path = str(Path(save_path).with_suffix('.png'))
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"   Inter-layer correlation plot saved to: {save_path} and {png_path}")

    plt.close()
