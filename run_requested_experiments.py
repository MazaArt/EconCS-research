"""
Run the requested experiment suite and save plots.

This script executes all requested cases from experiment_suite.py and writes
plots to plots/requested_experiments/. Files are overwritten on reruns.
"""

import os
import sys
import json
from typing import Callable, Dict, List, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from experiment_suite import (
    define_experiments,
    run_alpha_constant_m_budget_increase_case,
    run_alpha_increase_constant_ratio_case,
    run_alpha_increase_fixed_budget_case,
    run_base_case_n_scaling,
    run_budget_increase_case,
    run_signal_type_case,
)


OUTPUT_DIR = "plots/requested_experiments"
DATA_DIR = "data/requested_experiments"
PREFERENCES_SUBDIR = "agent_preferences"


def _paths_for_utility(utility_type: str) -> Tuple[str, str]:
    """Separate folders for cost-proportional runs so plots are not overwritten."""
    if utility_type == "cost_proportional":
        return (
            "plots/requested_experiments_cost_proportional",
            "data/requested_experiments_cost_proportional",
        )
    return "plots/requested_experiments", "data/requested_experiments"


CASE_1C_N_VALUES = list(range(10, 201, 10))
CASE_1C_NUM_SAMPLES = 100
CASE_1C_NUM_TRIALS = 100
DEFAULT_NUM_SAMPLES = 100
DEFAULT_NUM_TRIALS = 100
# Case 5 parallel ids zip to alpha_values in experiment_suite (same length required).
CASE_5_SUBIDS = ("5a", "5b", "5c")
VALID_EXPERIMENT_IDS = {"1a", "1b", "1c", "2", "3a", "3b", "4", "5", *CASE_5_SUBIDS}
# `all` runs 5a–5c instead of `5` so Case 5 is not executed four times.
ALL_EXPERIMENT_IDS = VALID_EXPERIMENT_IDS - {"5"}

RULE_COLORS = {
    "AV": "#1f77b4",
    "AV/Cost": "#17becf",
    "Bucket": "#7f7f7f",
    "GoB": "#e377c2",
    "GC": "#ff7f0e",
    "GC + AV": "#e377c2",
    "GC+AV": "#e377c2",
    "MES": "#2ca02c",
    "MES + AV": "#d62728",
    "MES+AV": "#d62728",
    "MES+Phragmen": "#bcbd22",
    "Phragmen": "#9467bd",
    "seq-PAV": "#8c564b",
    "ls-PAV": "#7f3c8d",
}


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, PREFERENCES_SUBDIR), exist_ok=True)


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _case_data_filename(case_id: str, params: dict) -> str:
    """
    Build a descriptive JSON filename for cases that change parameter sweeps.
    Falls back to legacy naming for other cases.
    """
    if case_id == "case2":
        m = _format_num_for_filename(params["m"])
        alpha = _format_num_for_filename(params["alpha"])
        budgets = params.get("budget_values", [])
        if budgets:
            b_start = _format_num_for_filename(min(budgets))
            b_end = _format_num_for_filename(max(budgets))
            return f"{case_id}_m{m}_a{alpha}_b{b_start}_to_{b_end}.json"
    if case_id == "case3a":
        m = _format_num_for_filename(params["m"])
        budget = _format_num_for_filename(params["budget"])
        alphas = params.get("alpha_values", [])
        if alphas:
            a_start = _format_num_for_filename(min(alphas))
            a_end = _format_num_for_filename(max(alphas))
            return f"{case_id}_m{m}_b{budget}_a{a_start}_to_{a_end}.json"
    return f"{case_id}.json"


def _case_preferences_filename(case_id: str, params: dict | None = None) -> str:
    """
    Build a descriptive JSONL filename for agent preference dumps.
    Falls back to legacy naming for other cases.
    """
    if params is not None:
        if case_id == "case2":
            m = _format_num_for_filename(params["m"])
            alpha = _format_num_for_filename(params["alpha"])
            budgets = params.get("budget_values", [])
            if budgets:
                b_start = _format_num_for_filename(min(budgets))
                b_end = _format_num_for_filename(max(budgets))
                return f"{case_id}_m{m}_a{alpha}_b{b_start}_to_{b_end}_agent_preferences.jsonl"
        if case_id == "case3a":
            m = _format_num_for_filename(params["m"])
            budget = _format_num_for_filename(params["budget"])
            alphas = params.get("alpha_values", [])
            if alphas:
                a_start = _format_num_for_filename(min(alphas))
                a_end = _format_num_for_filename(max(alphas))
                return f"{case_id}_m{m}_b{budget}_a{a_start}_to_{a_end}_agent_preferences.jsonl"
    return f"{case_id}_agent_preferences.jsonl"


def _save_simulation_data(
    case_id: str,
    params: dict,
    raw_data: dict,
    utility_type: str,
    num_samples: int,
    num_trials: int,
) -> None:
    case_label = case_id[4:] if case_id.startswith("case") else case_id
    payload = {
        "case": case_label,
        "case_id": case_id,
        "utility_type": utility_type,
        "num_samples": int(num_samples),
        "num_trials": int(num_trials),
        "num_trials_x_samples": int(num_samples) * int(num_trials),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "params": _to_builtin(params),
        "raw_data": _to_builtin(raw_data),
    }
    path = os.path.join(DATA_DIR, _case_data_filename(case_id, params))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved simulation data: {path}")


def _append_agent_preferences(
    *,
    case_id: str,
    preferences_file_name: str,
    utility_type: str,
    num_samples: int,
    num_trials: int,
    run_id: str,
    trial: int,
    context: dict,
    instance: dict,
) -> None:
    case_label = case_id[4:] if case_id.startswith("case") else case_id
    prefs_path = os.path.join(DATA_DIR, PREFERENCES_SUBDIR, preferences_file_name)
    record = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "case": case_label,
        "case_id": case_id,
        "utility_type": utility_type,
        "num_samples": int(num_samples),
        "num_trials": int(num_trials),
        "num_trials_x_samples": int(num_samples) * int(num_trials),
        "trial": int(trial),
        "context": _to_builtin(context),
        # Keep instance generation artifacts independent from voting-rule outputs,
        # so additional rules can be evaluated later on the exact same instances.
        "instance": _to_builtin(instance),
    }
    with open(prefs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record))
        f.write("\n")


def _build_trial_instance_callback(
    *,
    enabled: bool,
    case_id: str,
    preferences_file_name: str,
    utility_type: str,
    num_samples: int,
    num_trials: int,
    run_id: str,
) -> Callable[[dict, int, dict], None] | None:
    if not enabled:
        return None

    def _callback(context: dict, trial: int, instance: dict) -> None:
        _append_agent_preferences(
            case_id=case_id,
            preferences_file_name=preferences_file_name,
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
            run_id=run_id,
            trial=trial,
            context=context,
            instance=instance,
        )

    return _callback


def _summarize(per_rule: Dict[str, List[float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    means = {rule: float(np.mean(values)) for rule, values in per_rule.items()}
    stds = {rule: float(np.std(values)) for rule, values in per_rule.items()}
    return means, stds


def _plot_curve(
    x_values: List[float],
    y_by_rule: Dict[str, List[float]],
    x_label: str,
    title: str,
    filename: str,
    show_std: bool = False,
    std_by_rule: Dict[str, List[float]] | None = None,
) -> None:
    plt.figure(figsize=(11, 7))
    for rule, y_values in y_by_rule.items():
        color = RULE_COLORS.get(rule, "black")
        if show_std and std_by_rule is not None:
            plt.errorbar(
                x_values,
                y_values,
                yerr=std_by_rule[rule],
                marker="o",
                linewidth=2,
                capsize=4,
                color=color,
                label=rule,
            )
        else:
            plt.plot(x_values, y_values, marker="o", linewidth=2, color=color, label=rule)

    plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel("Performance")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def _plot_case_3a_rule_panels(
    alpha_values: List[float],
    mean_by_rule: Dict[str, List[float]],
    std_by_rule: Dict[str, List[float]],
    budget: float,
    filename: str,
) -> None:
    """
    Plot Case 3a as rule-wise panels (similar style to unit-vs-general figure).
    """
    rules = list(mean_by_rule.keys())
    n_plots = len(rules)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, rule in enumerate(rules):
        ax = axes[idx]
        means = mean_by_rule[rule]
        stds = std_by_rule[rule]
        color = RULE_COLORS.get(rule, "black")
        ax.errorbar(
            alpha_values,
            means,
            yerr=stds,
            marker="o",
            linewidth=3,
            markersize=8,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.4,
            color=color,
            label=rule,
        )
        ax.lines[-1].set_alpha(1.0)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, linewidth=2)
        ax.set_xlabel("Alpha", fontsize=20)
        ax.set_ylabel("Performance", fontsize=20)
        ax.set_title(rule, fontsize=22, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, loc="best")
        all_vals = means + [1.0]
        y_min = max(0.0, min(all_vals) - 0.05)
        y_max = min(1.05, max(all_vals) + 0.05)
        if y_min >= y_max:
            y_min = max(0.0, y_max - 0.1)
        ax.set_ylim([y_min, y_max])
        ax.tick_params(labelsize=14)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Rule-wise Performance vs Alpha (fixed budget={budget})",
        fontsize=26,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def _set_request_paths(utility_type: str) -> None:
    global OUTPUT_DIR, DATA_DIR
    OUTPUT_DIR, DATA_DIR = _paths_for_utility(utility_type)


def _format_num_for_filename(value: float | int) -> str:
    """Return compact numeric tags for filenames (e.g., 7.0 -> '7', 2.5 -> '2p5')."""
    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return str(as_float).replace(".", "p")


def run_all(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_trials: int = DEFAULT_NUM_TRIALS,
    utility_type: str = "normal",
    save_agent_preferences: bool = False,
) -> None:
    _set_request_paths(utility_type)
    cfg = define_experiments(utility_type)
    _ensure_output_dir()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Base case: n-scaling for each (alpha, budget) setting
    base = cfg["base_case_n_scaling"]
    n_values = base["n_values"]
    for setting in base["settings"]:
        alpha = setting["alpha"]
        budget = setting["budget"]
        label = setting["label"]
        is_case_1c = setting.get("fixed_costs") is not None
        case_n_values = CASE_1C_N_VALUES if is_case_1c else n_values
        case_num_samples = CASE_1C_NUM_SAMPLES if is_case_1c else num_samples
        case_num_trials = CASE_1C_NUM_TRIALS if is_case_1c else num_trials
        print(f"[Case 1{label}] alpha={alpha}, budget={budget}")
        raw = run_base_case_n_scaling(
            n_values=case_n_values,
            m=base["m"],
            alpha=alpha,
            budget=budget,
            fixed_costs=setting.get("fixed_costs"),
            fixed_cost_scale=setting.get("fixed_cost_scale"),
            quality_range=base["quality_range"],
            utility_type=base["utility_type"],
            num_samples=case_num_samples,
            num_trials=case_num_trials,
            trial_instance_callback=_build_trial_instance_callback(
                enabled=save_agent_preferences,
                case_id=f"case1{label}",
                preferences_file_name=_case_preferences_filename(f"case1{label}"),
                utility_type=utility_type,
                num_samples=case_num_samples,
                num_trials=case_num_trials,
                run_id=run_id,
            ),
        )
        _save_simulation_data(
            case_id=f"case1{label}",
            params=setting,
            raw_data=raw,
            utility_type=utility_type,
            num_samples=case_num_samples,
            num_trials=case_num_trials,
        )
        rules = list(next(iter(raw.values())).keys())
        means = {rule: [] for rule in rules}
        stds = {rule: [] for rule in rules}
        for n in case_n_values:
            mean_n, std_n = _summarize(raw[n])
            for rule in rules:
                means[rule].append(mean_n[rule])
                stds[rule].append(std_n[rule])
        _plot_curve(
            x_values=case_n_values,
            y_by_rule=means,
            std_by_rule=stds,
            show_std=True,
            x_label="Number of Agents (n)",
            title=(
                f"Performance vs n (Cambridge PB11 costs, budget={budget}, "
                f"samples={case_num_samples}, trials={case_num_trials})"
                if is_case_1c
                else f"Performance vs n (alpha={alpha}, budget={budget})"
            ),
            filename=(
                f"case1{label}_cambridge_pb11_budget{int(budget)}_s{case_num_samples}_t{case_num_trials}.png"
                if is_case_1c
                else f"case1{label}_alpha{int(alpha)}_budget{int(budget)}_s{case_num_samples}_t{case_num_trials}.png"
            ),
        )

    # 2) Budget increase
    bcfg = cfg["budget_increase"]
    print("[Case 2] Budget increase")
    raw = run_budget_increase_case(
        n=bcfg["n"],
        m=bcfg["m"],
        alpha=bcfg["alpha"],
        budget_values=bcfg["budget_values"],
        quality_range=bcfg["quality_range"],
        utility_type=bcfg["utility_type"],
        num_samples=num_samples,
        num_trials=num_trials,
        trial_instance_callback=_build_trial_instance_callback(
            enabled=save_agent_preferences,
            case_id="case2",
            preferences_file_name=_case_preferences_filename("case2", bcfg),
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
            run_id=run_id,
        ),
    )
    _save_simulation_data(
        case_id="case2",
        params=bcfg,
        raw_data=raw,
        utility_type=utility_type,
        num_samples=num_samples,
        num_trials=num_trials,
    )
    budgets = bcfg["budget_values"]
    rules = list(next(iter(raw.values())).keys())
    means = {rule: [] for rule in rules}
    stds = {rule: [] for rule in rules}
    for budget in budgets:
        mean_b, std_b = _summarize(raw[budget])
        for rule in rules:
            means[rule].append(mean_b[rule])
            stds[rule].append(std_b[rule])
    _plot_curve(
        x_values=budgets,
        y_by_rule=means,
        std_by_rule=stds,
        show_std=True,
        x_label="Budget",
        title=f"Performance vs Budget (alpha={bcfg['alpha']})",
        filename=(
            f"case2_budget_increase_alpha{_format_num_for_filename(bcfg['alpha'])}"
            f"_s{num_samples}_t{num_trials}.png"
        ),
    )

    # 3a) Alpha increase, fixed budget
    acfg = cfg["alpha_increase_fixed_budget"]
    print("[Case 3a] Alpha increase with fixed budget")
    raw = run_alpha_increase_fixed_budget_case(
        n=acfg["n"],
        m=acfg["m"],
        budget=acfg["budget"],
        alpha_values=acfg["alpha_values"],
        quality_range=acfg["quality_range"],
        utility_type=acfg["utility_type"],
        num_samples=num_samples,
        num_trials=num_trials,
        trial_instance_callback=_build_trial_instance_callback(
            enabled=save_agent_preferences,
            case_id="case3a",
            preferences_file_name=_case_preferences_filename("case3a", acfg),
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
            run_id=run_id,
        ),
    )
    _save_simulation_data(
        case_id="case3a",
        params=acfg,
        raw_data=raw,
        utility_type=utility_type,
        num_samples=num_samples,
        num_trials=num_trials,
    )
    alphas = acfg["alpha_values"]
    rules = list(next(iter(raw.values())).keys())
    means = {rule: [] for rule in rules}
    stds = {rule: [] for rule in rules}
    for alpha in alphas:
        mean_a, std_a = _summarize(raw[alpha])
        for rule in rules:
            means[rule].append(mean_a[rule])
            stds[rule].append(std_a[rule])
    _plot_curve(
        x_values=alphas,
        y_by_rule=means,
        std_by_rule=stds,
        show_std=True,
        x_label="Alpha",
        title=f"Performance vs Alpha (fixed budget={acfg['budget']})",
        filename=(
            f"case3a_alpha_increase_fixed_budget{_format_num_for_filename(acfg['budget'])}"
            f"_s{num_samples}_t{num_trials}.png"
        ),
    )

    # 3b) Alpha increase, constant ratio budget/(alpha+1)
    rcfg = cfg["alpha_increase_constant_ratio"]
    print("[Case 3b] Alpha increase with constant budget/(alpha+1)")
    for ratio in rcfg["ratios_budget_over_alpha_plus_one"]:
        raw_nested = run_alpha_increase_constant_ratio_case(
            n=rcfg["n"],
            m=rcfg["m"],
            alpha_values=rcfg["alpha_values"],
            ratios_budget_over_alpha_plus_one=[ratio],
            quality_range=rcfg["quality_range"],
            utility_type=rcfg["utility_type"],
            num_samples=num_samples,
            num_trials=num_trials,
            trial_instance_callback=_build_trial_instance_callback(
                enabled=save_agent_preferences,
                case_id=f"case3b_ratio_{str(ratio).replace('.', 'p')}",
                preferences_file_name=_case_preferences_filename(
                    f"case3b_ratio_{str(ratio).replace('.', 'p')}"
                ),
                utility_type=utility_type,
                num_samples=num_samples,
                num_trials=num_trials,
                run_id=run_id,
            ),
        )
        raw = raw_nested[ratio]
        _save_simulation_data(
            case_id=f"case3b_ratio_{str(ratio).replace('.', 'p')}",
            params={"ratio": ratio, **rcfg},
            raw_data=raw,
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
        )
        alphas = rcfg["alpha_values"]
        rules = list(next(iter(raw.values())).keys())
        means = {rule: [] for rule in rules}
        stds = {rule: [] for rule in rules}
        for alpha in alphas:
            mean_a, std_a = _summarize(raw[alpha])
            for rule in rules:
                means[rule].append(mean_a[rule])
                stds[rule].append(std_a[rule])
        ratio_tag = str(ratio).replace(".", "p")
        _plot_curve(
            x_values=alphas,
            y_by_rule=means,
            std_by_rule=stds,
            show_std=True,
            x_label="Alpha",
            title=f"Performance vs Alpha (budget/(alpha+1)={ratio:.4f})",
            filename=f"case3b_alpha_increase_ratio_{ratio_tag}_s{num_samples}_t{num_trials}.png",
        )

    # 4) Signal type case
    scfg = cfg["signal_type_case"]
    print("[Case 4] Signal type case")
    raw = run_signal_type_case(
        n=scfg["n"],
        m=scfg["m"],
        alpha=scfg["alpha"],
        budget=scfg["budget"],
        num_types_values=scfg["num_types_values"],
        quality_range=scfg["quality_range"],
        utility_type=scfg["utility_type"],
        num_samples=num_samples,
        num_trials=num_trials,
        trial_instance_callback=_build_trial_instance_callback(
            enabled=save_agent_preferences,
            case_id="case4",
            preferences_file_name=_case_preferences_filename("case4"),
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
            run_id=run_id,
        ),
    )
    _save_simulation_data(
        case_id="case4",
        params=scfg,
        raw_data=raw,
        utility_type=utility_type,
        num_samples=num_samples,
        num_trials=num_trials,
    )
    type_counts = scfg["num_types_values"]
    rules = list(next(iter(raw.values())).keys())
    means = {rule: [] for rule in rules}
    stds = {rule: [] for rule in rules}
    for k in type_counts:
        mean_k, std_k = _summarize(raw[k])
        for rule in rules:
            means[rule].append(mean_k[rule])
            stds[rule].append(std_k[rule])
    _plot_curve(
        x_values=type_counts,
        y_by_rule=means,
        std_by_rule=stds,
        show_std=True,
        x_label="Number of Agent Types",
        title=f"Signal Types (alpha={scfg['alpha']}, budget={scfg['budget']})",
        filename=f"case4_signal_types_s{num_samples}_t{num_trials}.png",
    )

    # 5) Alpha constant, m and budget increase
    mcfg = cfg["alpha_constant_m_budget_increase"]
    print("[Case 5] Alpha constant with increasing m and budget")
    for alpha in mcfg["alpha_values"]:
        raw_nested = run_alpha_constant_m_budget_increase_case(
            n=mcfg["n"],
            alpha_values=[alpha],
            m_values=mcfg["m_values"],
            budget_values=mcfg["budget_values"],
            quality_range=mcfg["quality_range"],
            utility_type=mcfg["utility_type"],
            num_samples=num_samples,
            num_trials=num_trials,
            trial_instance_callback=_build_trial_instance_callback(
                enabled=save_agent_preferences,
                case_id=f"case5_alpha_{int(alpha)}",
                preferences_file_name=_case_preferences_filename(f"case5_alpha_{int(alpha)}"),
                utility_type=utility_type,
                num_samples=num_samples,
                num_trials=num_trials,
                run_id=run_id,
            ),
        )
        raw = raw_nested[alpha]
        _save_simulation_data(
            case_id=f"case5_alpha_{int(alpha)}",
            params={"alpha": alpha, **mcfg},
            raw_data=raw,
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
        )
        # m crosses the PAV inclusion threshold (<=12), so keep only rules
        # that are present for every (m, budget) point to avoid KeyError.
        rules = sorted(set.intersection(*(set(point.keys()) for point in raw.values())))
        means = {rule: [] for rule in rules}
        stds = {rule: [] for rule in rules}
        for m, budget in zip(mcfg["m_values"], mcfg["budget_values"]):
            mean_mb, std_mb = _summarize(raw[(m, budget)])
            for rule in rules:
                means[rule].append(mean_mb[rule])
                stds[rule].append(std_mb[rule])
        _plot_curve(
            x_values=mcfg["m_values"],
            y_by_rule=means,
            std_by_rule=stds,
            show_std=True,
            x_label="Number of Projects (m)",
            title=f"Performance vs m (alpha={alpha}, budget scales with m)",
            filename=f"case5_alpha{int(alpha)}_m_budget_scaling_s{num_samples}_t{num_trials}.png",
        )

    print(f"All requested plots saved in: {OUTPUT_DIR}")


def _parse_run_requested_argv(argv: List[str]) -> Tuple[str, bool, List[str]]:
    """
    Strip known flags and return (utility_type, save_agent_preferences, experiment ids).

    Examples:
      python3 run_requested_experiments.py 1a --utility cost_proportional
      python3 run_requested_experiments.py all --utility cost_proportional
    """
    utility_type = "normal"
    save_agent_preferences = False
    rest: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--utility":
            if i + 1 >= len(argv):
                raise ValueError("--utility requires a value: normal or cost_proportional")
            utility_type = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--save-agent-preferences":
            save_agent_preferences = True
            i += 1
            continue
        rest.append(argv[i])
        i += 1
    if utility_type not in ("normal", "cost_proportional"):
        raise ValueError(
            f"Invalid --utility {utility_type!r}; use normal or cost_proportional."
        )
    return utility_type, save_agent_preferences, rest


def _case_5_alphas_to_run(experiment_ids: set[str], alpha_values: List[float]) -> List[float]:
    """
    Which Case 5 alphas to run: `5` runs all; `5a`/`5b`/`5c` run one each (dominant `5`
    ignores sub-ids so nothing is duplicated).
    """
    if len(alpha_values) != len(CASE_5_SUBIDS):
        raise ValueError(
            f"Case 5 expects {len(CASE_5_SUBIDS)} alpha values in config, got {len(alpha_values)}"
        )
    sub_to_alpha = dict(zip(CASE_5_SUBIDS, alpha_values))
    if "5" in experiment_ids:
        return list(alpha_values)
    return [sub_to_alpha[sid] for sid in CASE_5_SUBIDS if sid in experiment_ids]


def _parse_experiment_selection(argv: List[str]) -> set[str]:
    """
    Parse selected experiment ids from CLI args.

    Examples:
      python3 run_requested_experiments.py 1a 1b 3b
      python3 run_requested_experiments.py all
    """
    if not argv or "all" in {arg.lower() for arg in argv}:
        return set(ALL_EXPERIMENT_IDS)

    selected = {arg.lower() for arg in argv}
    unknown = sorted(selected - VALID_EXPERIMENT_IDS)
    if unknown:
        raise ValueError(
            f"Unknown experiment ids: {', '.join(unknown)}. "
            f"Valid ids: {', '.join(sorted(VALID_EXPERIMENT_IDS))}, all"
        )
    return selected


def run_selected(
    experiment_ids: set[str],
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_trials: int = DEFAULT_NUM_TRIALS,
    utility_type: str = "normal",
    save_agent_preferences: bool = False,
) -> None:
    """Run only selected experiments and save plots."""
    _set_request_paths(utility_type)
    cfg = define_experiments(utility_type)
    _ensure_output_dir()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Base case variants
    if any(case in experiment_ids for case in {"1a", "1b", "1c"}):
        base = cfg["base_case_n_scaling"]
        n_values = base["n_values"]
        for setting in base["settings"]:
            label = setting["label"]
            case_id = f"1{label}"
            if case_id not in experiment_ids:
                continue
            alpha = setting["alpha"]
            budget = setting["budget"]
            is_case_1c = setting.get("fixed_costs") is not None
            case_n_values = CASE_1C_N_VALUES if is_case_1c else n_values
            case_num_samples = CASE_1C_NUM_SAMPLES if is_case_1c else num_samples
            case_num_trials = CASE_1C_NUM_TRIALS if is_case_1c else num_trials
            print(f"[Case {case_id}] alpha={alpha}, budget={budget}")
            raw = run_base_case_n_scaling(
                n_values=case_n_values,
                m=base["m"],
                alpha=alpha,
                budget=budget,
                fixed_costs=setting.get("fixed_costs"),
                fixed_cost_scale=setting.get("fixed_cost_scale"),
                quality_range=base["quality_range"],
                utility_type=base["utility_type"],
                num_samples=case_num_samples,
                num_trials=case_num_trials,
                trial_instance_callback=_build_trial_instance_callback(
                    enabled=save_agent_preferences,
                    case_id=f"case{case_id}",
                    preferences_file_name=_case_preferences_filename(f"case{case_id}"),
                    utility_type=utility_type,
                    num_samples=case_num_samples,
                    num_trials=case_num_trials,
                    run_id=run_id,
                ),
            )
            _save_simulation_data(
                case_id=f"case{case_id}",
                params=setting,
                raw_data=raw,
                utility_type=utility_type,
                num_samples=case_num_samples,
                num_trials=case_num_trials,
            )
            rules = list(next(iter(raw.values())).keys())
            means = {rule: [] for rule in rules}
            stds = {rule: [] for rule in rules}
            for n in case_n_values:
                mean_n, std_n = _summarize(raw[n])
                for rule in rules:
                    means[rule].append(mean_n[rule])
                    stds[rule].append(std_n[rule])
            _plot_curve(
                x_values=case_n_values,
                y_by_rule=means,
                std_by_rule=stds,
                show_std=True,
                x_label="Number of Agents (n)",
                title=(
                    f"Performance vs n (Cambridge PB11 costs, budget={budget}, "
                    f"samples={case_num_samples}, trials={case_num_trials})"
                    if is_case_1c
                    else f"Performance vs n (alpha={alpha}, budget={budget})"
                ),
                filename=(
                    f"case1{label}_cambridge_pb11_budget{int(budget)}_s{case_num_samples}_t{case_num_trials}.png"
                    if is_case_1c
                    else f"case1{label}_alpha{int(alpha)}_budget{int(budget)}_s{case_num_samples}_t{case_num_trials}.png"
                ),
            )

    # 2) Budget increase
    if "2" in experiment_ids:
        bcfg = cfg["budget_increase"]
        print("[Case 2] Budget increase")
        raw = run_budget_increase_case(
            n=bcfg["n"],
            m=bcfg["m"],
            alpha=bcfg["alpha"],
            budget_values=bcfg["budget_values"],
            quality_range=bcfg["quality_range"],
            utility_type=bcfg["utility_type"],
            num_samples=num_samples,
            num_trials=num_trials,
            trial_instance_callback=_build_trial_instance_callback(
                enabled=save_agent_preferences,
                case_id="case2",
                preferences_file_name=_case_preferences_filename("case2", bcfg),
                utility_type=utility_type,
                num_samples=num_samples,
                num_trials=num_trials,
                run_id=run_id,
            ),
        )
        _save_simulation_data(
            case_id="case2",
            params=bcfg,
            raw_data=raw,
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
        )
        budgets = bcfg["budget_values"]
        rules = list(next(iter(raw.values())).keys())
        means = {rule: [] for rule in rules}
        stds = {rule: [] for rule in rules}
        for budget in budgets:
            mean_b, std_b = _summarize(raw[budget])
            for rule in rules:
                means[rule].append(mean_b[rule])
                stds[rule].append(std_b[rule])
        _plot_curve(
            x_values=budgets,
            y_by_rule=means,
            std_by_rule=stds,
            show_std=True,
            x_label="Budget",
            title=f"Performance vs Budget (alpha={bcfg['alpha']})",
            filename=(
                f"case2_budget_increase_alpha{_format_num_for_filename(bcfg['alpha'])}"
                f"_s{num_samples}_t{num_trials}.png"
            ),
        )

    # 3a) Alpha increase, fixed budget
    if "3a" in experiment_ids:
        acfg = cfg["alpha_increase_fixed_budget"]
        print("[Case 3a] Alpha increase with fixed budget")
        raw = run_alpha_increase_fixed_budget_case(
            n=acfg["n"],
            m=acfg["m"],
            budget=acfg["budget"],
            alpha_values=acfg["alpha_values"],
            quality_range=acfg["quality_range"],
            utility_type=acfg["utility_type"],
            num_samples=num_samples,
            num_trials=num_trials,
            trial_instance_callback=_build_trial_instance_callback(
                enabled=save_agent_preferences,
                case_id="case3a",
                preferences_file_name=_case_preferences_filename("case3a", acfg),
                utility_type=utility_type,
                num_samples=num_samples,
                num_trials=num_trials,
                run_id=run_id,
            ),
        )
        _save_simulation_data(
            case_id="case3a",
            params=acfg,
            raw_data=raw,
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
        )
        alphas = acfg["alpha_values"]
        rules = list(next(iter(raw.values())).keys())
        means = {rule: [] for rule in rules}
        stds = {rule: [] for rule in rules}
        for alpha in alphas:
            mean_a, std_a = _summarize(raw[alpha])
            for rule in rules:
                means[rule].append(mean_a[rule])
                stds[rule].append(std_a[rule])
        _plot_curve(
            x_values=alphas,
            y_by_rule=means,
            std_by_rule=stds,
            show_std=True,
            x_label="Alpha",
            title=f"Performance vs Alpha (fixed budget={acfg['budget']})",
            filename=(
                f"case3a_alpha_increase_fixed_budget{_format_num_for_filename(acfg['budget'])}"
                f"_s{num_samples}_t{num_trials}.png"
            ),
        )

    # 3b) Alpha increase, constant ratio budget/(alpha+1)
    if "3b" in experiment_ids:
        rcfg = cfg["alpha_increase_constant_ratio"]
        print("[Case 3b] Alpha increase with constant budget/(alpha+1)")
        for ratio in rcfg["ratios_budget_over_alpha_plus_one"]:
            raw_nested = run_alpha_increase_constant_ratio_case(
                n=rcfg["n"],
                m=rcfg["m"],
                alpha_values=rcfg["alpha_values"],
                ratios_budget_over_alpha_plus_one=[ratio],
                quality_range=rcfg["quality_range"],
                utility_type=rcfg["utility_type"],
                num_samples=num_samples,
                num_trials=num_trials,
                trial_instance_callback=_build_trial_instance_callback(
                    enabled=save_agent_preferences,
                    case_id=f"case3b_ratio_{str(ratio).replace('.', 'p')}",
                    preferences_file_name=_case_preferences_filename(
                        f"case3b_ratio_{str(ratio).replace('.', 'p')}"
                    ),
                    utility_type=utility_type,
                    num_samples=num_samples,
                    num_trials=num_trials,
                    run_id=run_id,
                ),
            )
            raw = raw_nested[ratio]
            _save_simulation_data(
                case_id=f"case3b_ratio_{str(ratio).replace('.', 'p')}",
                params={"ratio": ratio, **rcfg},
                raw_data=raw,
                utility_type=utility_type,
                num_samples=num_samples,
                num_trials=num_trials,
            )
            alphas = rcfg["alpha_values"]
            rules = list(next(iter(raw.values())).keys())
            means = {rule: [] for rule in rules}
            stds = {rule: [] for rule in rules}
            for alpha in alphas:
                mean_a, std_a = _summarize(raw[alpha])
                for rule in rules:
                    means[rule].append(mean_a[rule])
                    stds[rule].append(std_a[rule])
            ratio_tag = str(ratio).replace(".", "p")
            _plot_curve(
                x_values=alphas,
                y_by_rule=means,
                std_by_rule=stds,
                show_std=True,
                x_label="Alpha",
                title=f"Performance vs Alpha (budget/(alpha+1)={ratio:.4f})",
                filename=f"case3b_alpha_increase_ratio_{ratio_tag}_s{num_samples}_t{num_trials}.png",
            )

    # 4) Signal type case
    if "4" in experiment_ids:
        scfg = cfg["signal_type_case"]
        print("[Case 4] Signal type case")
        raw = run_signal_type_case(
            n=scfg["n"],
            m=scfg["m"],
            alpha=scfg["alpha"],
            budget=scfg["budget"],
            num_types_values=scfg["num_types_values"],
            quality_range=scfg["quality_range"],
            utility_type=scfg["utility_type"],
            num_samples=num_samples,
            num_trials=num_trials,
            trial_instance_callback=_build_trial_instance_callback(
                enabled=save_agent_preferences,
                case_id="case4",
                preferences_file_name=_case_preferences_filename("case4"),
                utility_type=utility_type,
                num_samples=num_samples,
                num_trials=num_trials,
                run_id=run_id,
            ),
        )
        _save_simulation_data(
            case_id="case4",
            params=scfg,
            raw_data=raw,
            utility_type=utility_type,
            num_samples=num_samples,
            num_trials=num_trials,
        )
        type_counts = scfg["num_types_values"]
        rules = list(next(iter(raw.values())).keys())
        means = {rule: [] for rule in rules}
        stds = {rule: [] for rule in rules}
        for k in type_counts:
            mean_k, std_k = _summarize(raw[k])
            for rule in rules:
                means[rule].append(mean_k[rule])
                stds[rule].append(std_k[rule])
        _plot_curve(
            x_values=type_counts,
            y_by_rule=means,
            std_by_rule=stds,
            show_std=True,
            x_label="Number of Agent Types",
            title=f"Signal Types (alpha={scfg['alpha']}, budget={scfg['budget']})",
            filename=f"case4_signal_types_s{num_samples}_t{num_trials}.png",
        )

    # 5) Alpha constant, m and budget increase
    mcfg = cfg["alpha_constant_m_budget_increase"]
    case5_alphas = _case_5_alphas_to_run(experiment_ids, mcfg["alpha_values"])
    if case5_alphas:
        print("[Case 5] Alpha constant with increasing m and budget")
        for alpha in case5_alphas:
            raw_nested = run_alpha_constant_m_budget_increase_case(
                n=mcfg["n"],
                alpha_values=[alpha],
                m_values=mcfg["m_values"],
                budget_values=mcfg["budget_values"],
                quality_range=mcfg["quality_range"],
                utility_type=mcfg["utility_type"],
                num_samples=num_samples,
                num_trials=num_trials,
                trial_instance_callback=_build_trial_instance_callback(
                    enabled=save_agent_preferences,
                    case_id=f"case5_alpha_{int(alpha)}",
                    preferences_file_name=_case_preferences_filename(f"case5_alpha_{int(alpha)}"),
                    utility_type=utility_type,
                    num_samples=num_samples,
                    num_trials=num_trials,
                    run_id=run_id,
                ),
            )
            raw = raw_nested[alpha]
            _save_simulation_data(
                case_id=f"case5_alpha_{int(alpha)}",
                params={"alpha": alpha, **mcfg},
                raw_data=raw,
                utility_type=utility_type,
                num_samples=num_samples,
                num_trials=num_trials,
            )
            # m crosses the PAV inclusion threshold (<=12), so keep only rules
            # that are present for every (m, budget) point to avoid KeyError.
            rules = sorted(set.intersection(*(set(point.keys()) for point in raw.values())))
            means = {rule: [] for rule in rules}
            stds = {rule: [] for rule in rules}
            for m, budget in zip(mcfg["m_values"], mcfg["budget_values"]):
                mean_mb, std_mb = _summarize(raw[(m, budget)])
                for rule in rules:
                    means[rule].append(mean_mb[rule])
                    stds[rule].append(std_mb[rule])
            _plot_curve(
                x_values=mcfg["m_values"],
                y_by_rule=means,
                std_by_rule=stds,
                show_std=True,
                x_label="Number of Projects (m)",
                title=f"Performance vs m (alpha={alpha}, budget scales with m)",
                filename=f"case5_alpha{int(alpha)}_m_budget_scaling_s{num_samples}_t{num_trials}.png",
            )

    print(f"Selected plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Runtime-oriented defaults; increase for higher statistical precision.
    try:
        utility_type, save_agent_preferences, argv_rest = _parse_run_requested_argv(sys.argv[1:])
        selected = _parse_experiment_selection(argv_rest)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)
    print(f"Utility setting: {utility_type}")
    print(f"Save agent preferences: {save_agent_preferences}")
    run_selected(
        selected,
        num_samples=DEFAULT_NUM_SAMPLES,
        num_trials=DEFAULT_NUM_TRIALS,
        utility_type=utility_type,
        save_agent_preferences=save_agent_preferences,
    )
