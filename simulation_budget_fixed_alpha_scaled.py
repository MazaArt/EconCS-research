"""
Fixed Budget with Scaled Alpha (Cost Ratio) Simulation

This script analyzes how informed ratios change as cost ratio (alpha) scales,
with a fixed budget.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import functions from the main simulation module
from simulation import (
    approval_voting, greedy_cover, method_of_equal_shares, mes_plus_av, phragmen,
    proportional_approval_voting,
    calculate_informed_ratio, generate_instance
)


def run_alpha_scaled_analysis(n: int, m: int, base_budget: float,
                              alpha_values: List[float], quality_range: Tuple[int, int],
                              utility_type: str = 'normal',
                              num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation with budget proportional to expected cost and scaled alpha values.
    
    The budget is scaled proportionally to the expected cost per alternative.
    Expected cost per alternative = (1 + alpha) / 2, so budget scales as (1 + alpha) / 2.
    This keeps the budget constraint at a consistent level relative to total costs.
    
    Args:
        n: number of agents (fixed)
        m: number of alternatives
        base_budget: base budget constraint (for alpha=1.0, budget = base_budget)
        alpha_values: list of alpha (cost ratio) values to test
        quality_range: (min_quality, max_quality)
        utility_type: 'normal' or 'cost_proportional'
        num_samples: number of samples per trial
        num_trials: number of independent trials
    
    Returns:
        Dictionary with results: {rule: {'mean': [...], 'std': [...]}}
    """
    use_cost_proportional = (utility_type == 'cost_proportional')
    
    voting_rules = {
        'AV': approval_voting,
        'GC': greedy_cover,
        'MES': method_of_equal_shares,
        'MES+AV': mes_plus_av,
        'Phragmen': phragmen
    }
    
    # Try PAV for small instances
    if m <= 12:
        voting_rules['PAV'] = proportional_approval_voting
    
    results = {
        rule: {'mean': [], 'std': []}
        for rule in voting_rules.keys()
    }
    
    for alpha in alpha_values:
        # Scale budget proportionally to expected cost
        # Expected cost per alternative = (1 + alpha) / 2
        # Scale budget by this ratio to keep constraint level consistent
        scaled_budget = base_budget * (1 + alpha) / 2.0
        
        print(f"Running simulation for alpha={alpha} (budget={scaled_budget:.2f})...", end=' ', flush=True)
        rule_ratios = {rule: [] for rule in voting_rules.keys()}
        
        for trial in range(num_trials):
            # Generate instance with scaled budget
            instance = generate_instance(n, m, alpha, scaled_budget, quality_range, seed=trial)
            
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


def plot_alpha_scaled(alpha_values: List[float], results: dict, rule_names: List[str],
                     n: int, m: int, base_budget: float, utility_type: str,
                     filename: str = None):
    """Plot performance with error bars vs alpha values."""
    import os
    plt.figure(figsize=(12, 8))
    
    # Define colors, markers, and linestyles for each rule
    rule_styles = {
        'AV': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'markersize': 8},
        'GC': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'markersize': 8},
        'MES': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'markersize': 8},
        'MES+AV': {'color': '#d62728', 'marker': 'v', 'linestyle': ':', 'markersize': 8},
        'Phragmen': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-', 'markersize': 8},
        'PAV': {'color': '#8c564b', 'marker': 'p', 'linestyle': '--', 'markersize': 8}
    }
    
    for rule_name in rule_names:
        means = results[rule_name]['mean']
        stds = results[rule_name]['std']
        style = rule_styles.get(rule_name, {'color': 'black', 'marker': 'o', 'linestyle': '-', 'markersize': 8})
        
        plt.plot(alpha_values, means,
                    marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], label=rule_name,
                    linewidth=3, markersize=style['markersize'])
    
    # Add horizontal line at y=1 (perfect performance)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (Performance=1)')
    
    plt.xlabel('Cost Ratio (α)', fontsize=24)
    plt.ylabel('Performance', fontsize=24)
    plt.title(f'Performance vs Cost Ratio\n'
              f'(n={n}, m={m}, base_B={base_budget}, budget∝(1+α)/2, utility={utility_type})',
              fontsize=28, fontweight='bold')
    plt.legend(fontsize=20, loc='best')
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
        filename = f'alpha_scaled_n{n}_m{m}_baseB{base_budget}_{utility_type}_{timestamp}.png'
    # Save to plots/simulation_budget_fixed_alpha_scaled folder
    os.makedirs('plots/simulation_budget_fixed_alpha_scaled', exist_ok=True)
    filepath = os.path.join('plots/simulation_budget_fixed_alpha_scaled', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filepath}'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Simulation parameters - Scaled budget with alpha
    n = 200  # Fixed number of agents (upwards of 200)
    m = 8  # number of alternatives
    base_budget = 5.0  # Base budget (budget scales as base_B * (1+α)/2)
    alpha_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Alpha from 1 to 10
    quality_range = (0, 1)  # binary qualities
    utility_type = 'normal'  # or 'cost_proportional'
    
    print("=" * 60)
    print("Scaled Budget with Scaled Alpha Simulation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n (agents): {n} (fixed)")
    print(f"  m (alternatives): {m}")
    print(f"  base budget: {base_budget} (budget scales as base_B * (1+α)/2)")
    print(f"  alpha values: {alpha_values}")
    print(f"  quality range: {quality_range}")
    print(f"  utility type: {utility_type}")
    print("=" * 60)
    
    # Run simulation
    results, rule_names = run_alpha_scaled_analysis(
        n, m, base_budget, alpha_values, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot results
    plot_alpha_scaled(alpha_values, results, rule_names, n, m, base_budget, utility_type)
    
    print("\nSimulation complete!")
