"""
Fixed Alpha (Cost Ratio) with Scaled Budget Simulation

This script analyzes how informed ratios change as budget scales,
with a fixed cost ratio (alpha).
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
        
        # Average over trials
        for rule_name in voting_rules.keys():
            avg_ratio = np.mean(rule_ratios[rule_name])
            results[rule_name].append(avg_ratio)
            print(f"{rule_name}: {avg_ratio:.4f}  ", end='', flush=True)
        print("done")
    
    return results, voting_rules.keys()


def plot_budget_scaled(budget_values: List[float], results: dict, rule_names: List[str],
                       n: int, m: int, alpha: float, utility_type: str,
                       filename: str = None):
    """Plot informed ratios vs budget values."""
    import os
    plt.figure(figsize=(10, 6))
    
    # Plot for each rule
    for rule_name in rule_names:
        ratios = results[rule_name]
        plt.plot(budget_values, ratios, marker='o', label=rule_name, linewidth=2, markersize=7)
    
    plt.xlabel('Budget (B)', fontsize=12)
    plt.ylabel('Informed Ratio', fontsize=12)
    plt.title(f'Informed Ratio vs Budget (n={n}, m={m}, α={alpha}, utility={utility_type})',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.tight_layout()
    
    if filename is None:
        filename = f'budget_scaled_n{n}_m{m}_alpha{alpha}_{utility_type}.png'
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
    n = 100  # Fixed number of agents
    m = 8  # number of alternatives
    alpha = 3.0  # Fixed cost ratio
    budget_values = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # Scaled budget values
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
    
    # Run simulation for original budget values
    results, rule_names = run_budget_scaled_analysis(
        n, m, alpha, budget_values, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot results
    plot_budget_scaled(budget_values, results, rule_names, n, m, alpha, utility_type)
    
    # Run finer-grained simulation with more budget values
    print("\n" + "=" * 60)
    print("Running finer-grained budget analysis")
    print("=" * 60)
    budget_values_fine = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0]
    results_fine, rule_names_fine = run_budget_scaled_analysis(
        n, m, alpha, budget_values_fine, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot finer-grained results
    filename_fine = f'budget_scaled_n{n}_m{m}_alpha{alpha}_{utility_type}_fine.png'
    plot_budget_scaled(budget_values_fine, results_fine, rule_names_fine, n, m, alpha, utility_type, filename=filename_fine)
    
    print("\nSimulation complete!")
