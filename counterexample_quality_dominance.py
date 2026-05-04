"""
Counterexample showing quality dominance is necessary.

Setup:
- m projects, unit costs, budget k (so exactly k projects can be funded).
- True qualities are sorted: first k projects have quality 2, remaining have quality 1.
- Let A be project k (0-index: k-1), B be project k+1 (0-index: k).

We violate quality dominance by assigning signal distributions such that B's signals
are strictly more positive than A's, despite A having higher true quality.

This can cause B to be selected while A is excluded.
"""

from __future__ import annotations

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from simulation import (
    approval_voting,
    approval_voting_per_cost,
    cost_bucketed_randomized_rule,
    greedy_or_breakpoint_rule,
    greedy_cover,
    knapsack_optimal,
    method_of_equal_shares,
    mes_plus_av,
    mes_plus_phragmen,
    phragmen,
    proportional_approval_voting,
)


def build_counterexample_votes(
    n: int,
    m: int,
    qualities: np.ndarray,
    signal_distribution: dict[int, dict[int, float]],
    seed: int = 0,
) -> np.ndarray:
    """
    Sample approvals from project-specific signal distributions at true qualities.
    """
    rng = np.random.default_rng(seed)
    votes = np.zeros((n, m), dtype=int)
    for j in range(m):
        q = int(qualities[j])
        p = float(signal_distribution[j][q])
        votes[:, j] = (rng.random(n) < p).astype(int)
    return votes


def run_counterexample(m: int = 8, k: int = 4, n: int = 2000, seed: int = 0) -> None:
    if not (2 <= k < m):
        raise ValueError("Require 2 <= k < m.")

    costs = np.ones(m, dtype=float)
    budget = float(k)

    # True quality order: top k all quality 2, rest quality 1.
    qualities = np.array([2] * k + [1] * (m - k), dtype=int)
    a_idx = k - 1
    b_idx = k

    # Explicit signal distributions used by this witness construction.
    # Format: P(signal=1 | quality=q) for q in {1, 2}.
    signal_distribution: dict[int, dict[int, float]] = {}
    for j in range(m):
        if j <= k - 2:
            # Strong high-quality projects before A.
            signal_distribution[j] = {1: 0.30, 2: 0.62}
        elif j == a_idx:
            # A has high true quality but weak signals.
            signal_distribution[j] = {1: 0.18, 2: 0.22}
        elif j == b_idx:
            # B has lower true quality but stronger signals than A.
            signal_distribution[j] = {1: 0.52, 2: 0.72}
        else:
            # Remaining projects have weak signals.
            signal_distribution[j] = {1: 0.10, 2: 0.20}

    votes = build_counterexample_votes(
        n=n,
        m=m,
        qualities=qualities,
        signal_distribution=signal_distribution,
        seed=seed,
    )

    rules = {
        "AV": approval_voting,
        "AV/Cost": approval_voting_per_cost,
        "Bucket": cost_bucketed_randomized_rule,
        "GoB": greedy_or_breakpoint_rule,
        "GC": greedy_cover,
        "MES": method_of_equal_shares,
        "MES+AV": mes_plus_av,
        "MES+Phragmen": mes_plus_phragmen,
        "Phragmen": phragmen,
    }
    if m <= 12:
        rules["seq-PAV"] = proportional_approval_voting
        rules["ls-PAV"] = local_search_pav

    optimal_set, optimal_utility = knapsack_optimal(
        qualities=qualities,
        costs=costs,
        budget=budget,
        use_cost_proportional=False,
    )

    print("=" * 72)
    print("Counterexample: quality dominance violation")
    print("=" * 72)
    print(f"m={m}, k={k}, n={n}, seed={seed}, unit costs, budget={budget}")
    print(f"A index={a_idx} (quality={qualities[a_idx]})")
    print(f"B index={b_idx} (quality={qualities[b_idx]})")
    print("Signal order violates QD: B has strictly more positive signals than A.")
    print(
        "Specifically: P_B(1) > P_A(1) and P_B(2) > P_A(2), "
        "despite quality(A)=2 > quality(B)=1."
    )
    print()
    print("Signal distributions P(signal=1 | quality=q):")
    print(f"  A (project {a_idx}): q=1 -> {signal_distribution[a_idx][1]:.1f}, q=2 -> {signal_distribution[a_idx][2]:.1f}")
    print(f"  B (project {b_idx}): q=1 -> {signal_distribution[b_idx][1]:.1f}, q=2 -> {signal_distribution[b_idx][2]:.1f}")
    if k - 2 >= 0:
        print(
            f"  Projects 0..{k-2} (quality-2 group before A): "
            f"q=1 -> {signal_distribution[0][1]:.1f}, q=2 -> {signal_distribution[0][2]:.1f}"
        )
    if b_idx + 1 <= m - 1:
        print(
            f"  Projects {b_idx+1}..{m-1} (quality-1 group after B): "
            f"q=1 -> {signal_distribution[m-1][1]:.1f}, q=2 -> {signal_distribution[m-1][2]:.1f}"
        )
    print()
    approval_counts = votes.sum(axis=0)
    print("Realized approval rates from sampled votes:")
    print(f"  A (project {a_idx}): approvals={approval_counts[a_idx]}/{n} ({approval_counts[a_idx] / n:.3f})")
    print(f"  B (project {b_idx}): approvals={approval_counts[b_idx]}/{n} ({approval_counts[b_idx] / n:.3f})")
    if k - 2 >= 0:
        print(f"  Mean over projects 0..{k-2}: {np.mean(approval_counts[:k-1]) / n:.3f}")
    if b_idx + 1 <= m - 1:
        print(f"  Mean over projects {b_idx+1}..{m-1}: {np.mean(approval_counts[b_idx+1:]) / n:.3f}")
    print()
    print(f"Optimal (true-quality) set: {sorted(optimal_set)}")
    print(f"Optimal utility: {optimal_utility}")
    print("-" * 72)

    witness_found = False
    for rule_name, rule_func in rules.items():
        chosen = rule_func(votes, costs, budget)
        chosen_sorted = sorted(chosen)
        chosen_utility = sum(qualities[j] for j in chosen)
        includes_b_not_a = (b_idx in chosen) and (a_idx not in chosen)
        witness_found = witness_found or includes_b_not_a

        marker = " <-- witness (B in, A out)" if includes_b_not_a else ""
        performance = chosen_utility / max(optimal_utility, 1e-10)
        print(f"{rule_name:14s} chosen={chosen_sorted} performance={performance:.3f}{marker}")

    print("-" * 72)
    if witness_found:
        print("Conclusion: At least one aggregation rule selects B but not A.")
        print("This demonstrates QD is necessary for quality-ranked selection guarantees.")
    else:
        print("No witness found under current parameters; try increasing n or changing m,k.")


def run_counterexample_aggregate(
    m: int = 8,
    k: int = 4,
    n: int = 2000,
    num_runs: int = 50,
    seed_start: int = 0,
    save_plot: bool = True,
    plot_filename: str = "plots/requested_experiments/counterexample_performance.png",
    save_data: bool = True,
    data_filename: str = "data/requested_experiments/counterexample_aggregate.json",
) -> None:
    """
    Aggregate the counterexample across multiple seeds and report cumulative stats.
    """
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")
    if not (2 <= k < m):
        raise ValueError("Require 2 <= k < m.")

    costs = np.ones(m, dtype=float)
    budget = float(k)
    qualities = np.array([2] * k + [1] * (m - k), dtype=int)
    a_idx = k - 1
    b_idx = k

    signal_distribution: dict[int, dict[int, float]] = {}
    for j in range(m):
        if j <= k - 2:
            signal_distribution[j] = {1: 0.30, 2: 0.62}
        elif j == a_idx:
            signal_distribution[j] = {1: 0.18, 2: 0.22}
        elif j == b_idx:
            signal_distribution[j] = {1: 0.52, 2: 0.72}
        else:
            signal_distribution[j] = {1: 0.10, 2: 0.20}

    rules = {
        "AV": approval_voting,
        "AV/Cost": approval_voting_per_cost,
        "Bucket": cost_bucketed_randomized_rule,
        "GoB": greedy_or_breakpoint_rule,
        "GC": greedy_cover,
        "MES": method_of_equal_shares,
        "MES+AV": mes_plus_av,
        "MES+Phragmen": mes_plus_phragmen,
        "Phragmen": phragmen,
    }
    if m <= 12:
        rules["seq-PAV"] = proportional_approval_voting
        rules["ls-PAV"] = local_search_pav

    optimal_set, optimal_utility = knapsack_optimal(
        qualities=qualities,
        costs=costs,
        budget=budget,
        use_cost_proportional=False,
    )

    perf_by_rule = {name: [] for name in rules}
    counts_by_rule = {
        name: {
            "opt_chose_a": 0,
            "non_opt_chose_b": 0,
            "non_opt_else": 0,
        }
        for name in rules
    }

    for run_idx in range(num_runs):
        seed = seed_start + run_idx
        votes = build_counterexample_votes(
            n=n,
            m=m,
            qualities=qualities,
            signal_distribution=signal_distribution,
            seed=seed,
        )
        for rule_name, rule_func in rules.items():
            chosen = rule_func(votes, costs, budget)
            chosen_sorted = tuple(sorted(chosen))
            chosen_utility = sum(qualities[j] for j in chosen)
            performance = chosen_utility / max(optimal_utility, 1e-10)
            perf_by_rule[rule_name].append(float(performance))

            is_opt = chosen_sorted == tuple(sorted(optimal_set))
            if is_opt and (a_idx in chosen):
                counts_by_rule[rule_name]["opt_chose_a"] += 1
            elif (not is_opt) and (b_idx in chosen):
                counts_by_rule[rule_name]["non_opt_chose_b"] += 1
            else:
                counts_by_rule[rule_name]["non_opt_else"] += 1

    print("=" * 72)
    print("Counterexample aggregate runner")
    print("=" * 72)
    print(f"m={m}, k={k}, n={n}, num_runs={num_runs}, seed_start={seed_start}")
    print("-" * 72)
    for rule_name in rules:
        mean_perf = float(np.mean(perf_by_rule[rule_name]))
        std_perf = float(np.std(perf_by_rule[rule_name]))
        opt_chose_a = counts_by_rule[rule_name]["opt_chose_a"]
        non_opt_chose_b = counts_by_rule[rule_name]["non_opt_chose_b"]
        non_opt_else = counts_by_rule[rule_name]["non_opt_else"]
        print(
            f"{rule_name:14s} mean_performance={mean_perf:.4f} "
            f"std={std_perf:.4f} "
            f"OPT_chose_A={opt_chose_a} "
            f"NonOPT_chose_B={non_opt_chose_b} "
            f"NonOPT_else={non_opt_else}"
        )

    if save_plot:
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        labels = list(rules.keys())
        opt_chose_a_vals = [counts_by_rule[name]["opt_chose_a"] for name in labels]
        non_opt_chose_b_vals = [counts_by_rule[name]["non_opt_chose_b"] for name in labels]
        non_opt_else_vals = [counts_by_rule[name]["non_opt_else"] for name in labels]

        bar_width = 0.26
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(
            x - bar_width,
            opt_chose_a_vals,
            width=bar_width,
            label="OPT, chose A",
            color="#4C72B0",
        )
        ax.bar(
            x,
            non_opt_chose_b_vals,
            width=bar_width,
            label="Non-OPT, chose B",
            color="#DD8452",
        )
        ax.bar(
            x + bar_width,
            non_opt_else_vals,
            width=bar_width,
            label="Non-OPT, else",
            color="#55A868",
        )
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.0, float(num_runs) * 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
        plt.title(f"Counterexample aggregate outcome categories ({num_runs} runs)")
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved aggregate plot: {plot_filename}")

    if save_data:
        os.makedirs(os.path.dirname(data_filename), exist_ok=True)
        payload = {
            "case_id": "counterexample_aggregate",
            "params": {
                "m": int(m),
                "k": int(k),
                "n": int(n),
                "num_runs": int(num_runs),
                "seed_start": int(seed_start),
            },
            "optimal": {
                "set": sorted(int(j) for j in optimal_set),
                "utility": float(optimal_utility),
                "a_idx": int(a_idx),
                "b_idx": int(b_idx),
            },
            "results": {
                rule_name: {
                    "mean_performance": float(np.mean(perf_by_rule[rule_name])),
                    "std_performance": float(np.std(perf_by_rule[rule_name])),
                    "performances": [float(x) for x in perf_by_rule[rule_name]],
                    "counts": {
                        "opt_chose_a": int(counts_by_rule[rule_name]["opt_chose_a"]),
                        "non_opt_chose_b": int(counts_by_rule[rule_name]["non_opt_chose_b"]),
                        "non_opt_else": int(counts_by_rule[rule_name]["non_opt_else"]),
                    },
                }
                for rule_name in rules
            },
        }
        with open(data_filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved aggregate data: {data_filename}")


if __name__ == "__main__":
    run_counterexample(m=8, k=4, n=2000, seed=0)
