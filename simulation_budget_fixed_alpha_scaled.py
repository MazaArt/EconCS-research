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
    approval_voting, greedy_cover, method_of_equal_shares, phragmen,
    proportional_approval_voting,
    calculate_informed_ratio, generate_instance
)


def run_alpha_scaled_analysis(n: int, m: int, base_budget: float,
                              alpha_values: List[float], quality_range: Tuple[int, int],
                              utility_type: str = 'normal',
                              num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation with budget proportional to expected total cost and scaled alpha values.
    
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
        Dictionary with results: {rule: [ratios]}
    """
    use_cost_proportional = (utility_type == 'cost_proportional')
    
    voting_rules = {
        'AV': approval_voting,
        'GC': greedy_cover,
        'MES': method_of_equal_shares,
        'Phragmen': phragmen
    }
    
    # Try PAV for small instances
    if m <= 12:
        voting_rules['PAV'] = proportional_approval_voting
    
    results = {rule: [] for rule in voting_rules.keys()}
    
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
        
        # Average over trials
        for rule_name in voting_rules.keys():
            avg_ratio = np.mean(rule_ratios[rule_name])
            results[rule_name].append(avg_ratio)
            print(f"{rule_name}: {avg_ratio:.4f}  ", end='', flush=True)
        print("done")
    
    return results, voting_rules.keys()


def plot_alpha_scaled(alpha_values: List[float], results: dict, rule_names: List[str],
                     n: int, m: int, base_budget: float, utility_type: str,
                     filename: str = None):
    """Plot informed ratios vs alpha values."""
    import os
    plt.figure(figsize=(10, 6))
    
    # Plot for each rule
    for rule_name in rule_names:
        ratios = results[rule_name]
        plt.plot(alpha_values, ratios, marker='o', label=rule_name, linewidth=2, markersize=7)
    
    plt.xlabel('Cost Ratio (α)', fontsize=12)
    plt.ylabel('Informed Ratio', fontsize=12)
    plt.title(f'Informed Ratio vs Cost Ratio\n'
              f'(n={n}, m={m}, base_B={base_budget}, budget∝(1+α)/2, utility={utility_type})',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.tight_layout()
    
    if filename is None:
        filename = f'alpha_scaled_n{n}_m{m}_baseB{base_budget}_{utility_type}.png'
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
    # Simulation parameters
    n = 100  # Fixed number of agents
    m = 8  # number of alternatives
    base_budget = 6.0  # Base budget (for alpha=1.0, budget = base_budget)
    # Budget scales as: budget(alpha) = base_budget * (1 + alpha) / 2
    # This keeps budget at ~60% of expected total cost (expected = m * (1+alpha)/2 = 4*(1+alpha))
    alpha_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # Scaled alpha values
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
    
    # Run simulation for original alpha values
    results, rule_names = run_alpha_scaled_analysis(
        n, m, base_budget, alpha_values, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot results
    plot_alpha_scaled(alpha_values, results, rule_names, n, m, base_budget, utility_type)
    
    # Run finer-grained simulation with more alpha values
    print("\n" + "=" * 60)
    print("Running finer-grained alpha analysis")
    print("=" * 60)
    alpha_values_fine = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0]
    results_fine, rule_names_fine = run_alpha_scaled_analysis(
        n, m, base_budget, alpha_values_fine, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot finer-grained results
    filename_fine = f'alpha_scaled_n{n}_m{m}_baseB{base_budget}_{utility_type}_fine.png'
    plot_alpha_scaled(alpha_values_fine, results_fine, rule_names_fine, n, m, base_budget, utility_type, filename=filename_fine)
    
    print("\nSimulation complete!")
