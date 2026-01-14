"""
Unit Cost vs General Cost Comparison Simulation

This script compares the informed ratio performance between unit cost (alpha=1)
and general cost (alpha>1) settings.
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


def run_unit_vs_general_comparison(n_values: List[int], m: int, alpha_values: List[float],
                                   budget: float, quality_range: Tuple[int, int],
                                   utility_type: str = 'normal',
                                   num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation comparing unit cost (alpha=1) vs general cost (alpha>1).
    
    Args:
        n_values: list of n (number of agents) to test
        m: number of alternatives
        alpha_values: list of alpha (cost ratio) values to test
        budget: budget constraint
        quality_range: (min_quality, max_quality)
        utility_type: 'normal' or 'cost_proportional'
        num_samples: number of samples per trial
        num_trials: number of independent trials
    
    Returns:
        Dictionary with results: {alpha: {rule: [ratios]}}
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
    
    results = {alpha: {rule: [] for rule in voting_rules.keys()} for alpha in alpha_values}
    
    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"Running simulation for alpha={alpha}")
        print(f"{'='*60}")
        
        for n in n_values:
            print(f"  n={n}...", end=' ', flush=True)
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
                results[alpha][rule_name].append(avg_ratio)
            print("done")
    
    return results, voting_rules.keys()


def plot_unit_vs_general(n_values: List[int], results: dict, rule_names: List[str],
                         alpha_values: List[float], m: int, budget: float,
                         utility_type: str, filename: str = None):
    """Plot unit cost vs general cost comparison."""
    import os
    n_plots = len(rule_names)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, rule_name in enumerate(rule_names):
        ax = axes[idx]
        
        # Plot for each alpha value
        for alpha in alpha_values:
            ratios = results[alpha][rule_name]
            label = f"α={alpha}" + (" (unit cost)" if alpha == 1.0 else " (general cost)")
            ax.plot(n_values, ratios, marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Number of Agents (n)', fontsize=11)
        ax.set_ylabel('Informed Ratio', fontsize=11)
        ax.set_title(f'{rule_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Unit Cost (α=1) vs General Cost (α>1) Comparison\n'
                 f'(m={m}, B={budget}, utility={utility_type})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if filename is None:
        filename = f'unit_vs_general_cost_m{m}_B{budget}_{utility_type}.png'
    # Save to plots/simulation_unit_vs_general folder
    os.makedirs('plots/simulation_unit_vs_general', exist_ok=True)
    filepath = os.path.join('plots/simulation_unit_vs_general', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filepath}'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Simulation parameters
    n_values = [50, 100, 200, 500, 1000]
    m = 8  # number of alternatives
    alpha_values = [1.0, 2.0, 3.0, 5.0]  # 1.0 = unit cost, >1.0 = general cost
    budget = 15.0
    quality_range = (0, 1)  # binary qualities
    utility_type = 'normal'  # or 'cost_proportional'
    
    print("=" * 60)
    print("Unit Cost vs General Cost Comparison Simulation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n values: {n_values}")
    print(f"  m (alternatives): {m}")
    print(f"  alpha values: {alpha_values}")
    print(f"  budget: {budget}")
    print(f"  quality range: {quality_range}")
    print(f"  utility type: {utility_type}")
    print("=" * 60)
    
    # Run simulation for original n values
    results, rule_names = run_unit_vs_general_comparison(
        n_values, m, alpha_values, budget, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot results
    plot_unit_vs_general(n_values, results, rule_names, alpha_values, m, budget, utility_type)
    
    # Run finer-grained simulation for small n values
    print("\n" + "=" * 60)
    print("Running finer-grained unit vs general comparison (n = 5, 10, 20, 50, 100, 250)")
    print("=" * 60)
    n_values_fine = [5, 10, 20, 50, 100, 250]
    results_fine, rule_names_fine = run_unit_vs_general_comparison(
        n_values_fine, m, alpha_values, budget, quality_range,
        utility_type, num_samples=30, num_trials=5
    )
    
    # Plot finer-grained results
    filename_fine = f'unit_vs_general_cost_m{m}_B{budget}_{utility_type}_fine.png'
    plot_unit_vs_general(n_values_fine, results_fine, rule_names_fine, alpha_values, m, budget, utility_type, filename=filename_fine)
    
    print("\nSimulation complete!")

