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
    gc_plus_av,
    mes_plus_av,
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
        "GC+AV": gc_plus_av,
        "MES+AV": mes_plus_av,
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
            "budget_values": [10.0, 20.0, 30.0, 40.0, 50.0],
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
            "alpha": 10.0,
            "budget": 14.0,
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
        if "reject_masks" in instance:
            row[name] = _calculate_informed_ratio_with_reject_masks(
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


def _calculate_informed_ratio_with_reject_masks(
    instance: dict,
    voting_rule,
    use_cost_proportional: bool = False,
    num_samples: int = 100,
) -> float:
    """
    Calculate informed ratio with type-based reject masks.

    Signal generation remains shared across agents via common signal distributions.
    If reject_masks[i, j] is True, agent i rejects project j regardless of signal.
    """
    n = instance["n"]
    m = instance["m"]
    costs = instance["costs"]
    budget = instance["budget"]
    qual_priors = instance["qual_priors"]
    signal_dists = instance["signal_dists"]
    quality_range = instance["quality_range"]
    reject_masks = instance["reject_masks"]
    min_qual, max_qual = quality_range

    total_voting_utility = 0.0
    total_optimal_utility = 0.0

    for _ in range(num_samples):
        qualities = np.zeros(m, dtype=int)
        for j in range(m):
            qualities[j] = max_qual if np.random.random() < qual_priors[j] else min_qual

        votes = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                signal_prob = signal_dists[(j, qualities[j])]
                signal_value = 1 if np.random.random() < signal_prob else 0
                votes[i, j] = 0 if reject_masks[i, j] else signal_value

        winning_set = voting_rule(votes, costs, budget)

        if use_cost_proportional:
            voting_utility = sum(costs[j] * qualities[j] for j in winning_set)
        else:
            voting_utility = sum(qualities[j] for j in winning_set)

        _, optimal_utility = knapsack_optimal(qualities, costs, budget, use_cost_proportional)
        total_voting_utility += voting_utility
        total_optimal_utility += max(optimal_utility, 1e-10)

    return total_voting_utility / total_optimal_utility if total_optimal_utility > 0 else 0.0


def _build_signal_type_masks(n: int, m: int, num_types: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build reject masks for each agent type.

    Each type receives a reject set R_t and feasible set F_t where:
    - R_t union F_t = all projects
    - R_t intersection F_t = empty
    """
    type_masks = np.zeros((num_types, m), dtype=bool)
    for t in range(num_types):
        reject_prob = rng.uniform(0.2, 0.6)
        type_masks[t] = rng.random(m) < reject_prob
        # Keep non-degenerate partitions.
        if type_masks[t].all():
            type_masks[t, rng.integers(0, m)] = False
        if (~type_masks[t]).all():
            type_masks[t, rng.integers(0, m)] = True
    agent_types = rng.integers(0, num_types, size=n)
    return type_masks[agent_types]


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
    Generate an instance and apply signal-type reject behavior.

    All agents share the same signal model. Agents of a given type still receive
    positive signals, but approvals are forced to 0 for projects in their reject
    set R.
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
    reject_masks = _build_signal_type_masks(n=n, m=m, num_types=num_types, rng=rng)
    adjusted_signals = instance["signals"].copy()
    adjusted_signals[reject_masks] = 0
    instance["signals"] = adjusted_signals
    instance["reject_masks"] = reject_masks
    instance["num_types"] = num_types
    return instance


def run_signal_type_case(
    n: int = 100,
    m: int = 8,
    alpha: float = 10.0,
    budget: float = 14.0,
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
            alpha_for_generation = alpha if alpha is not None else float(max(fixed_costs))
            instance = generate_instance(n, m_effective, alpha_for_generation, budget, quality_range, seed=trial)
            if fixed_costs is not None:
                instance["costs"] = np.array(fixed_costs, dtype=float)
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
