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
    approval_voting, greedy_cover, method_of_equal_shares, mes_plus_av, phragmen,
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
        'MES+AV': mes_plus_av,
        'Phragmen': phragmen
    }
    
    # Try PAV for small instances
    if m <= 12:
        voting_rules['PAV'] = proportional_approval_voting
    
    results = {
        util_type: {rule: {'mean': [], 'std': []} for rule in voting_rules.keys()}
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
            
            # Calculate mean and std over trials
            for rule_name in voting_rules.keys():
                ratios = rule_ratios[rule_name]
                results[utility_type][rule_name]['mean'].append(np.mean(ratios))
                results[utility_type][rule_name]['std'].append(np.std(ratios))
            print("done")
    
    return results, voting_rules.keys()


def plot_utility_comparison(n_values: List[int], results: dict, rule_names: List[str],
                           m: int, alpha: float, budget: float, filename: str = None):
    """Plot comparison between normal and cost-proportional utility functions."""
    import os
    
    # Control flag: Set to True to show STD bars, False to show only means
    # SHOW_STD_BARS = True   # Uncomment this line to enable STD bars
    SHOW_STD_BARS = False    # Comment out this line to disable STD bars
    
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
            means = results[utility_type][rule_name]['mean']
            stds = results[utility_type][rule_name]['std']
            label = 'Normal' if utility_type == 'normal' else 'Cost-Proportional'
            marker = 'o' if utility_type == 'normal' else 's'
            linestyle = '-' if utility_type == 'normal' else '--'
            color = '#1f77b4' if utility_type == 'normal' else '#ff7f0e'
            if SHOW_STD_BARS:
                ax.errorbar(n_values, means, yerr=stds,
                           marker=marker, linestyle=linestyle, color=color,
                           label=label, linewidth=3, markersize=8,
                           capsize=5, capthick=2, elinewidth=2)
            else:
                ax.plot(n_values, means,
                           marker=marker, linestyle=linestyle, color=color,
                           label=label, linewidth=3, markersize=8)
        
        # Add horizontal line at y=1 (perfect performance)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (Performance=1)')
        
        ax.set_xlabel('Number of Agents (n)', fontsize=22)
        ax.set_ylabel('Performance', fontsize=22)
        ax.set_title(f'{rule_name}', fontsize=24, fontweight='bold')
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)
        
        # Adjust y-axis to use most of the visual space
        all_means = [m for ut in ['normal', 'cost_proportional'] for m in results[ut][rule_name]['mean']]
        if SHOW_STD_BARS:
            all_stds = [s for ut in ['normal', 'cost_proportional'] for s in results[ut][rule_name]['std']]
            y_min = max(0, min(all_means) - 2 * max(all_stds) if all_stds else 0)
            y_max = min(1.05, max(all_means) + 2 * max(all_stds) if all_stds else 1.05)
        else:
            y_min = max(0, min(all_means) - 0.05)
            y_max = min(1.05, max(all_means) + 0.05)
        ax.set_ylim([y_min, y_max])
        ax.tick_params(labelsize=18)
    
    # Hide unused subplots
    for idx in range(n_rules, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Normal Utility vs Cost-Proportional Utility Comparison\n'
                 f'(m={m}, α={alpha}, B={budget})',
                 fontsize=28, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'utility_comparison_m{m}_alpha{alpha}_B{budget}_{timestamp}.png'
    # Save to plots/simulation_utility_comparison folder
    os.makedirs('plots/simulation_utility_comparison', exist_ok=True)
    filepath = os.path.join('plots/simulation_utility_comparison', filename)
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
    
    # Run simulation for original n values
    results, rule_names = run_utility_comparison(
        n_values, m, alpha, budget, quality_range,
        num_samples=30, num_trials=5
    )
    
    # Plot results
    plot_utility_comparison(n_values, results, rule_names, m, alpha, budget)
    
    print("\nSimulation complete!")

