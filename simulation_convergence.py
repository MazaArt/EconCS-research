"""
Convergence Analysis Simulation

This script analyzes the asymptotic convergence of performance (ratio to optimal knapsack) as n increases,
with confidence intervals and error bars.
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


def run_convergence_analysis(n_values: List[int], m: int, alpha: float,
                             budget: float, quality_range: Tuple[int, int],
                             utility_type: str = 'normal',
                             num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation for convergence analysis with error bars.
    
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
        Dictionary with results: {rule: {'mean': [...], 'std': [...], 'all': [[...]]}}
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
        rule: {
            'mean': [],
            'std': [],
            'all': []  # Store all trials for each n
        }
        for rule in voting_rules.keys()
    }
    
    for n in n_values:
        print(f"Running simulation for n={n}...")
        rule_ratios = {rule: [] for rule in voting_rules.keys()}
        
        for trial in range(num_trials):
            # Generate instance
            instance = generate_instance(n, m, alpha, budget, quality_range, seed=trial)
            
            # Calculate performance (ratio to optimal) for each rule
            for rule_name, rule_func in voting_rules.items():
                try:
                    ratio = calculate_informed_ratio(instance, rule_func, use_cost_proportional, num_samples)
                    rule_ratios[rule_name].append(ratio)
                except Exception as e:
                    print(f"  Error in {rule_name}: {e}")
                    rule_ratios[rule_name].append(0.0)
        
        # Calculate statistics
        for rule_name in voting_rules.keys():
            ratios = rule_ratios[rule_name]
            results[rule_name]['mean'].append(np.mean(ratios))
            results[rule_name]['std'].append(np.std(ratios))
            results[rule_name]['all'].append(ratios)
            print(f"  {rule_name}: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    
    return results, voting_rules.keys()


def plot_convergence(n_values: List[int], results: dict, rule_names: List[str],
                     m: int, alpha: float, budget: float, utility_type: str,
                     show_confidence: bool = True, confidence_level: float = 0.95,
                     filename: str = None):
    """Plot convergence analysis with error bars and confidence intervals."""
    import os
    n_rules = len(rule_names)
    n_cols = 2
    n_rows = (n_rules + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rules == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Try to import scipy for confidence intervals
    try:
        from scipy import stats
        has_scipy = True
    except ImportError:
        has_scipy = False
        show_confidence = False
    
    for idx, rule_name in enumerate(rule_names):
        ax = axes[idx]
        
        means = results[rule_name]['mean']
        stds = results[rule_name]['std']
        
        # Plot with error bars
        ax.errorbar(n_values, means, yerr=stds,
                   marker='o', linestyle='-', label='Mean ± std',
                   linewidth=3, markersize=8,
                   capsize=5, capthick=2, elinewidth=2)
        
        if show_confidence and has_scipy and len(results[rule_name]['all'][0]) > 1:
            # Calculate confidence intervals using t-distribution
            lower_bound = []
            upper_bound = []
            for i, all_ratios in enumerate(results[rule_name]['all']):
                if len(all_ratios) > 1:
                    # Use t-distribution for small samples
                    t_critical = stats.t.ppf((1 + confidence_level) / 2, len(all_ratios) - 1)
                    sem = np.std(all_ratios) / np.sqrt(len(all_ratios))
                    ci = t_critical * sem
                    # Clip lower bound to be at least 0 (performance can't be negative)
                    lower_bound.append(max(0.0, means[i] - ci))
                    upper_bound.append(means[i] + ci)
                else:
                    lower_bound.append(means[i])
                    upper_bound.append(means[i])
            
            # Fill confidence interval
            ax.fill_between(n_values, lower_bound, upper_bound, alpha=0.2, 
                           label=f'{int(confidence_level*100)}% CI')
        
        # Add horizontal line at y=1 (perfect performance)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect (Performance=1)')
        
        ax.set_xlabel('Number of Agents (n)', fontsize=22)
        ax.set_ylabel('Performance', fontsize=22)
        ax.set_title(f'{rule_name} Convergence', fontsize=24, fontweight='bold')
        ax.legend(fontsize=18, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Adjust y-axis to use most of the visual space - ensure it goes from 0 to positive values
        means = results[rule_name]['mean']
        stds = results[rule_name]['std']
        if means and stds:
            y_min = max(0.0, min(means) - 2 * max(stds))
            y_max = min(1.05, max(means) + 2 * max(stds))
        elif means:
            y_min = max(0.0, min(means) - 0.05)
            y_max = min(1.05, max(means) + 0.05)
        else:
            y_min = 0.0
            y_max = 1.05
        # Ensure y_min is never negative and y_min < y_max (critical to prevent inversion)
        y_min = max(0.0, y_min)
        if y_min >= y_max:
            y_min = max(0.0, y_max - 0.1)
        # Set y-axis limits (bottom < top, both >= 0)
        ax.set_ylim([y_min, y_max])
        ax.tick_params(labelsize=18)
    
    # Hide unused subplots
    for idx in range(n_rules, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Asymptotic Convergence Analysis\n'
                 f'(m={m}, α={alpha}, B={budget}, utility={utility_type})',
                 fontsize=28, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'convergence_m{m}_alpha{alpha}_B{budget}_{utility_type}_{timestamp}.png'
    # Save to plots/simulation_convergence folder
    os.makedirs('plots/simulation_convergence', exist_ok=True)
    filepath = os.path.join('plots/simulation_convergence', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filepath}'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Simulation parameters
    # Convergence to 1.0 should only happen under alpha = 1 (unit cost)
    # Budget should be less than number of alternatives to observe convergence
    n_values = list(range(10, 201, 10))  # n=10 to n=200 in steps of 10
    m = 8  # number of alternatives
    alpha = 1.0  # Unit cost - required for convergence to 1.0
    budget = 5.0  # Budget < m (8) - can't afford all alternatives
    quality_range = (0, 1)  # binary qualities
    utility_type = 'normal'  # or 'cost_proportional'
    
    print("=" * 60)
    print("Convergence Analysis Simulation")
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
    results, rule_names = run_convergence_analysis(
        n_values, m, alpha, budget, quality_range,
        utility_type, num_samples=30, num_trials=5  
    )
    
    # Plot results
    try:
        from scipy import stats
        plot_convergence(n_values, results, rule_names, m, alpha, budget, utility_type,
                        show_confidence=True, confidence_level=0.95)
    except ImportError:
        print("Warning: scipy not available, plotting without confidence intervals")
        plot_convergence(n_values, results, rule_names, m, alpha, budget, utility_type,
                        show_confidence=False, confidence_level=0.95)
    
    print("\nSimulation complete!")

