#!/usr/bin/env python3
"""
Rebuild requested experiment plots from saved JSON under data/, without rerunning simulations.

Uses the same summaries and filenames as run_requested_experiments.py, but assigns each voting
rule a distinct marker shape (and optionally a distinct linestyle; use --vary-linestyles to enable).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Match run_requested_experiments.py for colors.
RULE_COLORS: Dict[str, str] = {
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

RULE_ALIASES = {"GC + AV": "GC+AV", "MES + AV": "MES+AV"}

# Stable preference order for style assignment (aligned with experiment_suite._voting_rules).
CANONICAL_RULE_ORDER: List[str] = [
    "AV",
    "AV/Cost",
    "Bucket",
    "GoB",
    "GC",
    "MES",
    "MES+AV",
    "MES+Phragmen",
    "Phragmen",
    "seq-PAV",
    "ls-PAV",
]

MARKERS = ["o", "s", "^", "v", "D", "P", "*", "X", "<", ">", "p", "h", "8"]
# Include dash tuples so many rules still get visually distinct curves when printed grayscale.
LINESTYLES: List[Any] = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 5)),
    (0, (3, 3, 1, 3)),
    (0, (5, 2, 1, 2)),
    (0, (1, 1)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (5, 1)),
    (0, (2, 2)),
    (0, (5, 3, 1, 3, 1, 3)),
]


def _normalize_rule(rule: str) -> str:
    return RULE_ALIASES.get(rule, rule)


def _collect_rules_from_dir(data_dir: str) -> List[str]:
    seen_raw: set[str] = set()
    for name in os.listdir(data_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(data_dir, name)
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload.get("raw_data") or {}
        for point in raw.values():
            if isinstance(point, Mapping):
                seen_raw.update(point.keys())
    seen = {_normalize_rule(r) for r in seen_raw}
    ordered = [r for r in CANONICAL_RULE_ORDER if r in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def build_rule_styles(rule_order: Sequence[str]) -> Dict[str, Tuple[str, Any]]:
    out: Dict[str, Tuple[str, Any]] = {}
    for i, rule in enumerate(rule_order):
        out[rule] = (MARKERS[i % len(MARKERS)], LINESTYLES[i % len(LINESTYLES)])
    return out


def _summarize(per_rule: Mapping[str, Sequence[float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    means = {rule: float(np.mean(values)) for rule, values in per_rule.items()}
    stds = {rule: float(np.std(values)) for rule, values in per_rule.items()}
    return means, stds


def _resolve_raw_key(raw_data: Mapping[str, Any], x: Any) -> str:
    """Find the JSON object key used in raw_data for scalar coordinate x."""
    candidates: List[str] = []
    if isinstance(x, tuple):
        candidates.append(str(tuple(float(v) if isinstance(v, int) else v for v in x)))
        candidates.append(str(x))
    else:
        candidates.append(str(x))
        if isinstance(x, bool):
            pass
        elif isinstance(x, int):
            candidates.append(str(float(x)))
        elif isinstance(x, float):
            candidates.extend([str(int(x)) if x.is_integer() else str(x), repr(x)])
    seen_c = []
    for c in candidates:
        if c not in seen_c:
            seen_c.append(c)
    for c in seen_c:
        if c in raw_data:
            return c
    raise KeyError(f"No raw_data key for x={x!r} among attempts {seen_c}")


def _case5_keys(params: Mapping[str, Any], raw_data: Mapping[str, Any]) -> List[str]:
    keys = []
    for m, budget in zip(params["m_values"], params["budget_values"]):
        key = str((int(m), float(budget)))
        if key not in raw_data:
            key = _resolve_raw_key(raw_data, (int(m), float(budget)))
        keys.append(key)
    return keys


def ordered_point_keys(case_id: str, params: Mapping[str, Any], raw_data: Mapping[str, Any]) -> List[str]:
    if case_id.startswith("case5"):
        return _case5_keys(params, raw_data)
    if case_id == "case2":
        return [_resolve_raw_key(raw_data, float(b)) for b in params["budget_values"]]
    if case_id.startswith("case3"):
        return [_resolve_raw_key(raw_data, float(a)) for a in params["alpha_values"]]
    if case_id == "case4":
        return [_resolve_raw_key(raw_data, int(k)) for k in params["num_types_values"]]
    # case1*: n on x-axis — keys may be absent from params; sort numerically.
    numeric_keys = sorted(raw_data.keys(), key=lambda s: float(s))
    return numeric_keys


def x_values_for_keys(case_id: str, params: Mapping[str, Any], ordered_keys: Sequence[str]) -> List[float]:
    if case_id.startswith("case5"):
        return [float(ast.literal_eval(k)[0]) for k in ordered_keys]
    if case_id == "case2":
        return [float(ast.literal_eval(k)) if k.startswith("(") else float(k) for k in ordered_keys]
    if case_id.startswith("case3"):
        return [float(k) for k in ordered_keys]
    if case_id == "case4":
        return [float(k) for k in ordered_keys]
    return [float(k) for k in ordered_keys]


def infer_rules(case_id: str, raw_data: Mapping[str, Any], ordered_keys: Sequence[str]) -> List[str]:
    if case_id.startswith("case5"):
        return sorted(set.intersection(*(set(raw_data[k].keys()) for k in ordered_keys)))
    first = raw_data[ordered_keys[0]]
    return list(first.keys())


def build_series(
    case_id: str,
    params: Mapping[str, Any],
    raw_data: Mapping[str, Any],
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, List[float]], List[str]]:
    keys = ordered_point_keys(case_id, params, raw_data)
    rules = infer_rules(case_id, raw_data, keys)
    means = {rule: [] for rule in rules}
    stds = {rule: [] for rule in rules}
    for k in keys:
        mean_pt, std_pt = _summarize(raw_data[k])
        for rule in rules:
            means[rule].append(mean_pt[rule])
            stds[rule].append(std_pt[rule])
    xs = x_values_for_keys(case_id, params, keys)
    return xs, means, stds, keys


def plot_filename(payload: Mapping[str, Any]) -> str:
    case_id = payload["case_id"]
    params = payload["params"]
    ns = int(payload.get("num_samples") or 100)
    nt = int(payload.get("num_trials") or 100)

    if case_id.startswith("case1"):
        label = case_id.replace("case1", "")
        is_c = params.get("fixed_costs") is not None
        if is_c:
            return f"case1{label}_cambridge_pb11_budget{int(params['budget'])}_s{ns}_t{nt}.png"
        alpha = params["alpha"]
        budget = params["budget"]
        return f"case1{label}_alpha{int(alpha)}_budget{int(budget)}_s{ns}_t{nt}.png"

    if case_id == "case2":
        alpha = int(params["alpha"])
        return f"case2_budget_increase_alpha{alpha}_s{ns}_t{nt}.png"

    if case_id == "case3a":
        budget = int(params["budget"])
        return f"case3a_alpha_increase_fixed_budget{budget}_s{ns}_t{nt}.png"

    if case_id.startswith("case3b_ratio_"):
        ratio = params["ratio"]
        ratio_tag = str(ratio).replace(".", "p")
        return f"case3b_alpha_increase_ratio_{ratio_tag}_s{ns}_t{nt}.png"

    if case_id == "case4":
        return f"case4_signal_types_s{ns}_t{nt}.png"

    if case_id.startswith("case5_alpha_"):
        alpha = params["alpha"]
        return f"case5_alpha{int(alpha)}_m_budget_scaling_s{ns}_t{nt}.png"

    raise ValueError(f"Unhandled case_id for filename: {case_id}")


def title_and_xlabel(case_id: str, params: Mapping[str, Any], payload: Mapping[str, Any]) -> Tuple[str, str]:
    ns = int(payload.get("num_samples") or 100)
    nt = int(payload.get("num_trials") or 100)

    if case_id.startswith("case1"):
        label = case_id.replace("case1", "")
        budget = params["budget"]
        if params.get("fixed_costs") is not None:
            return (
                f"Performance vs n (Cambridge PB11 costs, budget={budget}, samples={ns}, trials={nt})",
                "Number of Agents (n)",
            )
        alpha = params["alpha"]
        return (f"Performance vs n (alpha={alpha}, budget={budget})", "Number of Agents (n)")

    if case_id == "case2":
        return (f"Performance vs Budget (alpha={params['alpha']})", "Budget")

    if case_id == "case3a":
        return (f"Performance vs Alpha (fixed budget={params['budget']})", "Alpha")

    if case_id.startswith("case3b_ratio_"):
        return (
            f"Performance vs Alpha (budget/(alpha+1)={params['ratio']:.4f})",
            "Alpha",
        )

    if case_id == "case4":
        return (
            f"Signal Types (alpha={params['alpha']}, budget={params['budget']})",
            "Number of Agent Types",
        )

    if case_id.startswith("case5_alpha_"):
        alpha = params["alpha"]
        return (
            f"Performance vs m (alpha={alpha}, budget scales with m)",
            "Number of Projects (m)",
        )

    raise ValueError(f"Unhandled case_id for title: {case_id}")


def plot_curve(
    *,
    x_values: List[float],
    y_by_rule: Dict[str, List[float]],
    std_by_rule: Dict[str, List[float]],
    x_label: str,
    title: str,
    outfile: str,
    rule_styles: Mapping[str, Tuple[str, Any]],
    vary_linestyles: bool,
) -> None:
    plt.figure(figsize=(11, 7))
    for rule in y_by_rule:
        color = RULE_COLORS.get(rule, RULE_COLORS.get(_normalize_rule(rule), "black"))
        marker, linestyle = rule_styles[_normalize_rule(rule)]
        if not vary_linestyles:
            linestyle = "-"
        y_values = y_by_rule[rule]
        plt.errorbar(
            x_values,
            y_values,
            yerr=std_by_rule[rule],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            capsize=4,
            color=color,
            label=rule,
        )

    plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel("Performance")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def process_payload(
    payload: Mapping[str, Any],
    *,
    out_dir: str,
    rule_styles: Mapping[str, Tuple[str, Any]],
    vary_linestyles: bool,
) -> None:
    case_id = payload["case_id"]
    params = payload["params"]
    raw_data = payload["raw_data"]

    xs, means, stds, _keys = build_series(case_id, params, raw_data)
    fname = plot_filename(payload)
    title, x_label = title_and_xlabel(case_id, params, payload)
    os.makedirs(out_dir, exist_ok=True)
    plot_curve(
        x_values=xs,
        y_by_rule=means,
        std_by_rule=stds,
        x_label=x_label,
        title=title,
        outfile=os.path.join(out_dir, fname),
        rule_styles=rule_styles,
        vary_linestyles=vary_linestyles,
    )


def plot_directory(*, data_dir: str, plots_dir: str, vary_linestyles: bool) -> None:
    rule_order = _collect_rules_from_dir(data_dir)
    styles = build_rule_styles(rule_order)

    json_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    for name in json_files:
        path = os.path.join(data_dir, name)
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        process_payload(
            payload,
            out_dir=plots_dir,
            rule_styles=styles,
            vary_linestyles=vary_linestyles,
        )
        print(f"Wrote plot from {path} -> {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--utility",
        choices=("normal", "cost_proportional", "both"),
        default="both",
        help="Which saved dataset folder under data/ to plot.",
    )
    parser.add_argument(
        "--vary-linestyles",
        action="store_true",
        help=(
            "Give each voting rule a different linestyle (default: solid '-' for every rule; "
            "markers and colors still differ by rule)."
        ),
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    pairs: List[Tuple[str, str]] = []
    if args.utility in ("normal", "both"):
        pairs.append(
            (
                os.path.join(root, "data", "requested_experiments"),
                os.path.join(root, "plots", "requested_experiments_rule_styles"),
            )
        )
    if args.utility in ("cost_proportional", "both"):
        pairs.append(
            (
                os.path.join(root, "data", "requested_experiments_cost_proportional"),
                os.path.join(root, "plots", "requested_experiments_cost_proportional_rule_styles"),
            )
        )

    for data_dir, plots_dir in pairs:
        if not os.path.isdir(data_dir):
            print(f"Skip missing data dir: {data_dir}")
            continue
        plot_directory(
            data_dir=data_dir,
            plots_dir=plots_dir,
            vary_linestyles=args.vary_linestyles,
        )


if __name__ == "__main__":
    main()
