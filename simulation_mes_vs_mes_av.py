"""
MES vs MES + AV Comparison Simulation

This script compares the informed ratio performance between standard MES
and a hybrid approach: MES followed by AV for budget exhaustion.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set

# Import functions from the main simulation module
from simulation import (
    approval_voting, method_of_equal_shares,
    calculate_informed_ratio, generate_instance
)


def mes_plus_av(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    Hybrid MES + AV: First run MES, then use AV to exhaust remaining budget.
    
    Args:
        votes: (n, m) binary array, votes[i, j] = 1 if agent i approves alternative j
        costs: (m,) array of costs
        budget: budget constraint
    
    Returns:
        Set of winning alternative indices
    """
    # First, run MES
    mes_winning_set = method_of_equal_shares(votes, costs, budget)
    
    # Calculate remaining budget
    used_budget = sum(costs[j] for j in mes_winning_set)
    remaining_budget = budget - used_budget
    
    # If no remaining budget, return MES result
    if remaining_budget <= 1e-6:  # Small threshold for floating point
        return mes_winning_set
    
    # Use AV to fill remaining budget
    m = len(costs)
    approval_counts = votes.sum(axis=0)
    
    # Get alternatives not yet selected
    remaining_alternatives = [j for j in range(m) if j not in mes_winning_set]
    
    if not remaining_alternatives:
        return mes_winning_set
    
    # Sort by approval count (descending), break ties by index
    remaining_alternatives.sort(key=lambda j: (approval_counts[j], -j), reverse=True)
    
    # Add alternatives until budget is exhausted
    for j in remaining_alternatives:
        if costs[j] <= remaining_budget:
            mes_winning_set.add(j)
            remaining_budget -= costs[j]
        else:
            break  # Can't afford any more alternatives
    
    return mes_winning_set


def run_mes_vs_mes_av_comparison(n_values: List[int], m: int, alpha: float,
                                 budget: float, quality_range: Tuple[int, int],
                                 utility_type: str = 'normal',
                                 num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation comparing MES vs MES + AV.
    
    Args:
        n_values: list of n (number of agents) to test
        m: number of alternatives
        alpha: cost ratio
        budget: budget constraint
        quality_range: (min_quality, max_quality)
        utility_type: 'normal' or 'cost_proportional'
        num_samples: number of samples per trial
        num_trials: number of independent trials
    
    Returns:
        Dictionary with results: {rule: [ratios]}
    """
    use_cost_proportional = (utility_type == 'cost_proportional')
    
    voting_rules = {
        'MES': method_of_equal_shares,
        'MES + AV': mes_plus_av
    }
    
    results = {
        rule: {'mean': [], 'std': []}
        for rule in voting_rules.keys()
    }
    
    for n in n_values:
        print(f"Running simulation for n={n}...", end=' ', flush=True)
        rule_ratios = {rule: [] for rule in voting_rules.keys()}
        
        for trial in range(num_trials):
            # Generate instance
            instance = generate_instance(n, m, alpha, budget, quality_range, seed=trial)
            
            # Calculate informed ratio for each rule
            for rule_name, rule_func in voting_rules.items():
                try:
                    ratio = calculate_informed_ratio(instance, rule_func, use_cost_proportional, num_samples)
                    rule_ratios[rule_name].append(ratio)
                except Exception as e:
                    print(f"\n    Error in {rule_name}: {e}")
                    rule_ratios[rule_name].append(0.0)
        
        # Calculate mean and std over trials
        for rule_name in voting_rules.keys():
            ratios = rule_ratios[rule_name]
            results[rule_name]['mean'].append(np.mean(ratios))
            results[rule_name]['std'].append(np.std(ratios))
            print(f"{rule_name}: {np.mean(ratios):.4f}±{np.std(ratios):.4f}  ", end='', flush=True)
        print("done")
    
    return results, voting_rules.keys()


def plot_mes_vs_mes_av(n_values: List[int], results: dict, rule_names: List[str],
                      m: int, alpha: float, budget: float, utility_type: str,
                      filename: str = None):
    """Plot comparison between MES and MES + AV with error bars."""
    import os
    plt.figure(figsize=(12, 8))
    
    # Define styles for each rule
    rule_styles = {
        'MES': {'marker': 'o', 'linestyle': '-', 'color': '#2ca02c'},
        'MES + AV': {'marker': 's', 'linestyle': '--', 'color': '#d62728'},
        'MES+AV': {'marker': 's', 'linestyle': '--', 'color': '#d62728'}
    }
    
    # Plot for each rule
    for rule_name in rule_names:
        means = results[rule_name]['mean']
        stds = results[rule_name]['std']
        style = rule_styles.get(rule_name, {'marker': 'o', 'linestyle': '-', 'color': 'black'})
        plt.plot(n_values, means,
                    marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], label=rule_name,
                    linewidth=3, markersize=8)
    
    # Add horizontal line at y=1 (perfect performance)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (Performance=1)')
    
    plt.xlabel('Number of Agents (n)', fontsize=24)
    plt.ylabel('Performance', fontsize=24)
    plt.title(f'MES vs MES + AV Comparison\n'
              f'(m={m}, α={alpha}, B={budget}, utility={utility_type})',
              fontsize=28, fontweight='bold')
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to use most of the visual space
    all_means = [m for r in results.values() for m in r['mean']]
    y_min = max(0, min(all_means) - 0.05)
    y_max = min(1.05, max(all_means) + 0.05)
    plt.ylim([y_min, y_max])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'mes_vs_mes_av_m{m}_alpha{alpha}_B{budget}_{utility_type}_{timestamp}.png'
    # Save to plots/simulation_mes_vs_mes_av folder
    os.makedirs('plots/simulation_mes_vs_mes_av', exist_ok=True)
    filepath = os.path.join('plots/simulation_mes_vs_mes_av', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filepath}'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Simulation parameters
    n_values = list(range(10, 201, 10))  # n=10 to n=200 in steps of 10
    m = 8  # number of alternatives
    alpha = 5.0  # cost ratio (max/min) - non-unit cost simulation
    budget = 8.0
    quality_range = (0, 1)  # binary qualities
    utility_type = 'normal'  # or 'cost_proportional'
    
    print("=" * 60)
    print("MES vs MES + AV Comparison Simulation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n values: {n_values}")
    print(f"  m (alternatives): {m}")
    print(f"  alpha (cost ratio): {alpha}")
    print(f"  budget: {budget}")
    print(f"  quality range: {quality_range}")
    print(f"  utility type: {utility_type}")
    print("=" * 60)
    
    # Run simulation for original n values
    results, rule_names = run_mes_vs_mes_av_comparison(
        n_values, m, alpha, budget, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot results
    plot_mes_vs_mes_av(n_values, results, rule_names, m, alpha, budget, utility_type)
    
    print("\nSimulation complete!")
