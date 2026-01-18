"""
Fixed Alpha (Cost Ratio) with Scaled Budget Simulation

This script analyzes how performance changes as budget scales,
with a fixed cost ratio (alpha).
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


def run_budget_scaled_analysis(n: int, m: int, alpha: float,
                               budget_values: List[float], quality_range: Tuple[int, int],
                               utility_type: str = 'normal',
                               num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation with fixed alpha and scaled budget values.
    
    Args:
        n: number of agents (fixed)
        m: number of alternatives
        alpha: cost ratio (fixed)
        budget_values: list of budget values to test
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
    
    for budget in budget_values:
        print(f"Running simulation for budget={budget}...", end=' ', flush=True)
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


def plot_budget_scaled(budget_values: List[float], results: dict, rule_names: List[str],
                       n: int, m: int, alpha: float, utility_type: str,
                       filename: str = None):
    """Plot performance with error bars vs budget values."""
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
        
        plt.errorbar(budget_values, means, yerr=stds,
                    marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], label=rule_name,
                    linewidth=3, markersize=style['markersize'],
                    capsize=5, capthick=2, elinewidth=2)
    
    # Add horizontal line at y=1 (perfect performance)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (Performance=1)')
    
    plt.xlabel('Budget (B)', fontsize=24)
    plt.ylabel('Performance', fontsize=24)
    plt.title(f'Performance vs Budget (n={n}, m={m}, α={alpha}, utility={utility_type})',
              fontsize=28, fontweight='bold')
    plt.legend(fontsize=20, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to use most of the visual space
    all_means = [m for r in results.values() for m in r['mean']]
    all_stds = [s for r in results.values() for s in r['std']]
    y_min = max(0, min(all_means) - 2 * max(all_stds) if all_stds else 0)
    y_max = min(1.05, max(all_means) + 2 * max(all_stds) if all_stds else 1.05)
    plt.ylim([y_min, y_max])
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'budget_scaled_n{n}_m{m}_alpha{alpha}_{utility_type}_{timestamp}.png'
    # Save to plots/simulation_alpha_fixed_budget_scaled folder
    os.makedirs('plots/simulation_alpha_fixed_budget_scaled', exist_ok=True)
    filepath = os.path.join('plots/simulation_alpha_fixed_budget_scaled', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filepath}'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Simulation parameters
    n = 200  # Fixed number of agents (upwards of 200)
    m = 8  # number of alternatives
    alpha = 5.0  # Fixed cost ratio - non-unit cost simulation
    budget_values = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # Budget values
    quality_range = (0, 1)  # binary qualities
    utility_type = 'normal'  # or 'cost_proportional'
    
    print("=" * 60)
    print("Fixed Alpha with Scaled Budget Simulation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n (agents): {n} (fixed)")
    print(f"  m (alternatives): {m}")
    print(f"  alpha (cost ratio): {alpha} (fixed)")
    print(f"  budget values: {budget_values}")
    print(f"  quality range: {quality_range}")
    print(f"  utility type: {utility_type}")
    print("=" * 60)
    
    # Run simulation
    results, rule_names = run_budget_scaled_analysis(
        n, m, alpha, budget_values, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot results
    plot_budget_scaled(budget_values, results, rule_names, n, m, alpha, utility_type)
    
    print("\nSimulation complete!")
