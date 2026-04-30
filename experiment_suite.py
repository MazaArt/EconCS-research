"""
Experiment suite definitions for participatory budgeting research.

This module defines the requested experiment families and provides helper
functions to run them with the existing simulation engine.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from simulation import (
    approval_voting,
    approval_voting_per_cost,
    greedy_cover,
    method_of_equal_shares,
    mes_plus_av,
    mes_plus_phragmen,
    phragmen,
    proportional_approval_voting,
    calculate_informed_ratio,
    generate_instance,
    knapsack_optimal,
)


def _voting_rules(m: int):
    rules = {
        "AV": approval_voting,
        "AV/Cost": approval_voting_per_cost,
        "GC": greedy_cover,
        "MES": method_of_equal_shares,
        "MES+AV": mes_plus_av,
        "MES+Phragmen": mes_plus_phragmen,
        "Phragmen": phragmen,
    }
    if m <= 12:
        rules["PAV"] = proportional_approval_voting
    return rules


def define_experiments() -> Dict[str, dict]:
    """Return all requested experiments as a structured dictionary."""
    return {
        # 1) Base Case: increase n at fixed alpha/budget pairs
        "base_case_n_scaling": {
            "n_values": list(range(10, 201, 10)),
            "m": 8,
            "quality_range": (0, 2),
            "utility_type": "normal",
            "settings": [
                {"label": "a", "alpha": 1.0, "budget": 4.0},
                {"label": "b", "alpha": 5.0, "budget": 8.0},
                # Cambridge PB Cycle 11 ballot costs and winning budget.
                {
                    "label": "c",
                    "alpha": None,
                    "budget": 1_060_000.0,
                    "fixed_cost_scale": 1000.0,
                    "fixed_costs": [
                        250000.0, 75000.0, 150000.0, 250000.0, 75000.0,
                        60000.0, 200000.0, 350000.0, 150000.0, 200000.0,
                        45000.0, 15000.0, 30000.0, 12000.0, 150000.0,
                        20000.0, 250000.0, 60000.0, 110000.0, 20000.0,
                    ],
                },
            ],
        },
        # 2) Budget Increase case
        "budget_increase": {
            "n": 100,
            "m": 8,
            "alpha": 20.0,
            "budget_values": [20.0, 30.0, 40.0, 50.0],
            "quality_range": (0, 2),
            "utility_type": "normal",
        },
        # 3a) Alpha Increase with fixed budget
        "alpha_increase_fixed_budget": {
            "n": 100,
            "m": 8,
            "budget": 40.0,
            "alpha_values": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            "quality_range": (0, 2),
            "utility_type": "normal",
        },
        # 3b) Alpha Increase with fixed budget/(alpha+1) ratio
        "alpha_increase_constant_ratio": {
            "n": 100,
            "m": 8,
            "alpha_values": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            "ratios_budget_over_alpha_plus_one": [0.5, 1 / 3, 2 / 3, 0.25, 0.75],
            "quality_range": (0, 2),
            "utility_type": "normal",
        },
        # 4) Signal Type case
        "signal_type_case": {
            "n": 100,
            "m": 8,
            "alpha": 5.0,
            "budget": 8.0,
            "num_types_values": [1, 2, 3, 4, 5],
            "quality_range": (0, 2),
            "utility_type": "normal",
        },
        # 5) Alpha constant; projects and budget increase
        "alpha_constant_m_budget_increase": {
            "n": 100,
            "alpha_values": [5.0, 10.0, 20.0],
            "m_values": [4, 8, 12, 16, 20],
            "budget_values": [8.0, 16.0, 24.0, 32.0, 40.0],
            "quality_range": (0, 2),
            "utility_type": "normal",
        },
    }


def _run_single_instance(instance: dict, utility_type: str, num_samples: int, m: int) -> Dict[str, float]:
    """Evaluate all voting rules on one fixed generated instance."""
    use_cost_proportional = utility_type == "cost_proportional"
    rules = _voting_rules(m)
    row = {}
    for name, func in rules.items():
        if "agent_types" in instance and "type_signal_dists" in instance:
            row[name] = _calculate_informed_ratio_with_type_signal_dists(
                instance=instance,
                voting_rule=func,
                use_cost_proportional=use_cost_proportional,
                num_samples=num_samples,
            )
        else:
            row[name] = calculate_informed_ratio(
                instance=instance,
                voting_rule=func,
                use_cost_proportional=use_cost_proportional,
                num_samples=num_samples,
            )
    return row


def _calculate_informed_ratio_with_type_signal_dists(
    instance: dict,
    voting_rule,
    use_cost_proportional: bool = False,
    num_samples: int = 100,
) -> float:
    """
    Calculate informed ratio with type-dependent signal distributions.

    All types share the same feasible set (all projects), but each type receives
    systematically different signal probabilities for the same project-quality
    pair.
    """
    n = instance["n"]
    m = instance["m"]
    costs = instance["costs"]
    budget = instance["budget"]
    qual_priors = instance["qual_priors"]
    signal_dists = instance["signal_dists"]
    quality_range = instance["quality_range"]
    agent_types = instance["agent_types"]
    type_signal_dists = instance["type_signal_dists"]
    min_qual, max_qual = quality_range

    total_voting_utility = 0.0
    total_optimal_utility = 0.0

    for _ in range(num_samples):
        qualities = np.zeros(m, dtype=int)
        for j in range(m):
            qualities[j] = max_qual if np.random.random() < qual_priors[j] else min_qual

        votes = np.zeros((n, m), dtype=int)
        for i in range(n):
            t = int(agent_types[i])
            for j in range(m):
                signal_prob = type_signal_dists[(t, j, qualities[j])]
                if (j, qualities[j]) not in signal_dists:
                    signal_prob = signal_dists[(j, qualities[j])]
                votes[i, j] = 1 if np.random.random() < signal_prob else 0

        winning_set = voting_rule(votes, costs, budget)

        if use_cost_proportional:
            voting_utility = sum(costs[j] * qualities[j] for j in winning_set)
        else:
            voting_utility = sum(qualities[j] for j in winning_set)

        _, optimal_utility = knapsack_optimal(qualities, costs, budget, use_cost_proportional)
        total_voting_utility += voting_utility
        total_optimal_utility += max(optimal_utility, 1e-10)

    return total_voting_utility / total_optimal_utility if total_optimal_utility > 0 else 0.0


def _build_type_signal_dists(
    signal_dists: dict,
    m: int,
    quality_range: Tuple[int, int],
    num_types: int,
    rng: np.random.Generator,
) -> dict:
    """
    Build type-specific signal distributions over a shared feasible set.

    For each type and each project-quality pair, perturb the baseline signal
    probability and clamp to [0.05, 0.95] to keep signals informative.
    """
    min_q, max_q = quality_range
    type_signal_dists = {}
    for t in range(num_types):
        type_shift = rng.uniform(-0.15, 0.15)
        for j in range(m):
            project_shift = rng.uniform(-0.10, 0.10)
            for q in range(min_q, max_q + 1):
                base = signal_dists[(j, q)]
                p = np.clip(base + type_shift + project_shift, 0.05, 0.95)
                type_signal_dists[(t, j, q)] = float(p)
    return type_signal_dists


def _generate_instance_with_signal_types(
    n: int,
    m: int,
    alpha: float,
    budget: float,
    quality_range: Tuple[int, int],
    num_types: int,
    seed: int,
) -> dict:
    """
    Generate an instance with type-specific signal distributions.

    All types have the same feasible set (all projects). The only heterogeneity
    is in how informative/noisy their signals are.
    """
    rng = np.random.default_rng(seed)
    instance = generate_instance(
        n=n,
        m=m,
        alpha=alpha,
        budget=budget,
        quality_range=quality_range,
        seed=seed,
    )
    agent_types = rng.integers(0, num_types, size=n)
    type_signal_dists = _build_type_signal_dists(
        signal_dists=instance["signal_dists"],
        m=m,
        quality_range=quality_range,
        num_types=num_types,
        rng=rng,
    )
    instance["agent_types"] = agent_types
    instance["type_signal_dists"] = type_signal_dists
    instance["num_types"] = num_types
    return instance


def run_signal_type_case(
    n: int = 100,
    m: int = 8,
    alpha: float = 5.0,
    budget: float = 8.0,
    num_types_values: List[int] | None = None,
    quality_range: Tuple[int, int] = (0, 2),
    utility_type: str = "normal",
    num_samples: int = 30,
    num_trials: int = 20,
) -> Dict[int, Dict[str, List[float]]]:
    """Run the signal-type experiment for num_types in {1,2,3,4,5}."""
    if num_types_values is None:
        num_types_values = [1, 2, 3, 4, 5]

    aggregated: Dict[int, Dict[str, List[float]]] = {}
    for num_types in num_types_values:
        rules = _voting_rules(m)
        per_rule = {name: [] for name in rules}

        for trial in range(num_trials):
            instance = _generate_instance_with_signal_types(
                n=n,
                m=m,
                alpha=alpha,
                budget=budget,
                quality_range=quality_range,
                num_types=num_types,
                seed=trial,
            )
            row = _run_single_instance(
                instance=instance,
                utility_type=utility_type,
                num_samples=num_samples,
                m=m,
            )
            for rule_name, value in row.items():
                per_rule[rule_name].append(value)

        aggregated[num_types] = per_rule
    return aggregated


def run_base_case_n_scaling(
    n_values: List[int],
    m: int,
    alpha: float | None,
    budget: float,
    fixed_costs: List[float] | None = None,
    fixed_cost_scale: float | None = None,
    quality_range: Tuple[int, int] = (0, 2),
    utility_type: str = "normal",
    num_samples: int = 30,
    num_trials: int = 20,
) -> Dict[int, Dict[str, List[float]]]:
    """Run one base-case setting while increasing n."""
    aggregated: Dict[int, Dict[str, List[float]]] = {}
    for n in n_values:
        m_effective = len(fixed_costs) if fixed_costs is not None else m
        rules = _voting_rules(m_effective)
        per_rule = {name: [] for name in rules}
        for trial in range(num_trials):
            scale = (
                float(fixed_cost_scale)
                if (fixed_costs is not None and fixed_cost_scale is not None and fixed_cost_scale > 0)
                else 1.0
            )
            alpha_for_generation = (
                alpha
                if alpha is not None
                else float(max(fixed_costs)) / scale
            )
            effective_budget = budget / scale
            instance = generate_instance(
                n, m_effective, alpha_for_generation, effective_budget, quality_range, seed=trial
            )
            if fixed_costs is not None:
                instance["costs"] = np.array(fixed_costs, dtype=float) / scale
                instance["m"] = m_effective
            row = _run_single_instance(instance, utility_type, num_samples, m_effective)
            for rule_name, value in row.items():
                per_rule[rule_name].append(value)
        aggregated[n] = per_rule
    return aggregated


def run_budget_increase_case(
    n: int,
    m: int,
    alpha: float,
    budget_values: List[float],
    quality_range: Tuple[int, int] = (0, 2),
    utility_type: str = "normal",
    num_samples: int = 30,
    num_trials: int = 20,
) -> Dict[float, Dict[str, List[float]]]:
    """Run fixed-alpha experiment while increasing budget."""
    aggregated: Dict[float, Dict[str, List[float]]] = {}
    for budget in budget_values:
        rules = _voting_rules(m)
        per_rule = {name: [] for name in rules}
        for trial in range(num_trials):
            instance = generate_instance(n, m, alpha, budget, quality_range, seed=trial)
            row = _run_single_instance(instance, utility_type, num_samples, m)
            for rule_name, value in row.items():
                per_rule[rule_name].append(value)
        aggregated[budget] = per_rule
    return aggregated


def run_alpha_increase_fixed_budget_case(
    n: int,
    m: int,
    budget: float,
    alpha_values: List[float],
    quality_range: Tuple[int, int] = (0, 2),
    utility_type: str = "normal",
    num_samples: int = 30,
    num_trials: int = 20,
) -> Dict[float, Dict[str, List[float]]]:
    """Run fixed-budget experiment while increasing alpha."""
    aggregated: Dict[float, Dict[str, List[float]]] = {}
    for alpha in alpha_values:
        rules = _voting_rules(m)
        per_rule = {name: [] for name in rules}
        for trial in range(num_trials):
            instance = generate_instance(n, m, alpha, budget, quality_range, seed=trial)
            row = _run_single_instance(instance, utility_type, num_samples, m)
            for rule_name, value in row.items():
                per_rule[rule_name].append(value)
        aggregated[alpha] = per_rule
    return aggregated


def run_alpha_increase_constant_ratio_case(
    n: int,
    m: int,
    alpha_values: List[float],
    ratios_budget_over_alpha_plus_one: List[float],
    quality_range: Tuple[int, int] = (0, 2),
    utility_type: str = "normal",
    num_samples: int = 30,
    num_trials: int = 20,
) -> Dict[float, Dict[float, Dict[str, List[float]]]]:
    """
    Run alpha-increase experiments while preserving budget/(alpha+1)=ratio.

    Returns a nested dictionary keyed by ratio then alpha.
    """
    aggregated: Dict[float, Dict[float, Dict[str, List[float]]]] = {}
    for ratio in ratios_budget_over_alpha_plus_one:
        aggregated[ratio] = {}
        for alpha in alpha_values:
            budget = ratio * (alpha + 1.0)
            rules = _voting_rules(m)
            per_rule = {name: [] for name in rules}
            for trial in range(num_trials):
                instance = generate_instance(n, m, alpha, budget, quality_range, seed=trial)
                row = _run_single_instance(instance, utility_type, num_samples, m)
                for rule_name, value in row.items():
                    per_rule[rule_name].append(value)
            aggregated[ratio][alpha] = per_rule
    return aggregated


def run_alpha_constant_m_budget_increase_case(
    n: int,
    alpha_values: List[float],
    m_values: List[int],
    budget_values: List[float],
    quality_range: Tuple[int, int] = (0, 2),
    utility_type: str = "normal",
    num_samples: int = 30,
    num_trials: int = 20,
) -> Dict[float, Dict[Tuple[int, float], Dict[str, List[float]]]]:
    """Run alpha-constant case while (m, budget) scale together."""
    if len(m_values) != len(budget_values):
        raise ValueError("m_values and budget_values must have the same length.")

    aggregated: Dict[float, Dict[Tuple[int, float], Dict[str, List[float]]]] = {}
    for alpha in alpha_values:
        aggregated[alpha] = {}
        for m, budget in zip(m_values, budget_values):
            rules = _voting_rules(m)
            per_rule = {name: [] for name in rules}
            for trial in range(num_trials):
                instance = generate_instance(n, m, alpha, budget, quality_range, seed=trial)
                row = _run_single_instance(instance, utility_type, num_samples, m)
                for rule_name, value in row.items():
                    per_rule[rule_name].append(value)
            aggregated[alpha][(m, budget)] = per_rule
    return aggregated


if __name__ == "__main__":
    experiments = define_experiments()
    print("Experiment suite loaded with the following cases:")
    for key in experiments:
        print(f" - {key}")
