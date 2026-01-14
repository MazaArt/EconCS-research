"""
Utility Function Comparison Simulation

This script compares the informed ratio performance between normal utility
and cost-proportional utility functions.
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


def run_utility_comparison(n_values: List[int], m: int, alpha: float,
                          budget: float, quality_range: Tuple[int, int],
                          num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation comparing normal vs cost-proportional utility functions.
    
    Args:
        n_values: list of n (number of agents) to test
        m: number of alternatives
        alpha: cost ratio
        budget: budget constraint
        quality_range: (min_quality, max_quality)
        num_samples: number of samples per trial
        num_trials: number of independent trials
    
    Returns:
        Dictionary with results: {utility_type: {rule: [ratios]}}
    """
    utility_types = ['normal', 'cost_proportional']
    
    voting_rules = {
        'AV': approval_voting,
        'GC': greedy_cover,
        'MES': method_of_equal_shares,
        'Phragmen': phragmen
    }
    
    # Try PAV for small instances
    if m <= 12:
        voting_rules['PAV'] = proportional_approval_voting
    
    results = {
        util_type: {rule: [] for rule in voting_rules.keys()}
        for util_type in utility_types
    }
    
    for utility_type in utility_types:
        print(f"\n{'='*60}")
        print(f"Running simulation for {utility_type} utility")
        print(f"{'='*60}")
        use_cost_proportional = (utility_type == 'cost_proportional')
        
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
                results[utility_type][rule_name].append(avg_ratio)
            print("done")
    
    return results, voting_rules.keys()


def plot_utility_comparison(n_values: List[int], results: dict, rule_names: List[str],
                           m: int, alpha: float, budget: float):
    """Plot comparison between normal and cost-proportional utility functions."""
    n_rules = len(rule_names)
    n_cols = 2
    n_rows = (n_rules + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rules == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, rule_name in enumerate(rule_names):
        ax = axes[idx]
        
        # Plot for each utility type
        for utility_type in ['normal', 'cost_proportional']:
            ratios = results[utility_type][rule_name]
            label = 'Normal' if utility_type == 'normal' else 'Cost-Proportional'
            marker = 'o' if utility_type == 'normal' else 's'
            linestyle = '-' if utility_type == 'normal' else '--'
            ax.plot(n_values, ratios, marker=marker, linestyle=linestyle,
                   label=label, linewidth=2, markersize=7)
        
        ax.set_xlabel('Number of Agents (n)', fontsize=11)
        ax.set_ylabel('Informed Ratio', fontsize=11)
        ax.set_title(f'{rule_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # Hide unused subplots
    for idx in range(n_rules, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Normal Utility vs Cost-Proportional Utility Comparison\n'
                 f'(m={m}, α={alpha}, B={budget})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    filename = f'utility_comparison_m{m}_alpha{alpha}_B{budget}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filename}'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Simulation parameters
    n_values = [50, 100, 200, 500, 1000]
    m = 8  # number of alternatives
    alpha = 3.0  # cost ratio (max/min)
    budget = 15.0
    quality_range = (0, 1)  # binary qualities
    
    print("=" * 60)
    print("Utility Function Comparison Simulation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n values: {n_values}")
    print(f"  m (alternatives): {m}")
    print(f"  alpha (cost ratio): {alpha}")
    print(f"  budget: {budget}")
    print(f"  quality range: {quality_range}")
    print("=" * 60)
    
    # Run simulation
    results, rule_names = run_utility_comparison(
        n_values, m, alpha, budget, quality_range,
        num_samples=30, num_trials=5
    )
    
    # Plot results
    plot_utility_comparison(n_values, results, rule_names, m, alpha, budget)
    
    print("\nSimulation complete!")

