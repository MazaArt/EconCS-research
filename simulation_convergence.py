"""
Convergence Analysis Simulation

This script analyzes the asymptotic convergence of informed ratios as n increases,
with confidence intervals and error bars.
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
            
            # Calculate informed ratio for each rule
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
                     show_confidence: bool = True, confidence_level: float = 0.95):
    """Plot convergence analysis with error bars and confidence intervals."""
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
        
        # Plot mean line
        ax.plot(n_values, means, marker='o', label='Mean', linewidth=2, markersize=8)
        
        # Plot error bars (standard deviation)
        ax.errorbar(n_values, means, yerr=stds, fmt='none', 
                   capsize=5, capthick=1.5, alpha=0.6, label=f'±1 std')
        
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
                    lower_bound.append(means[i] - ci)
                    upper_bound.append(means[i] + ci)
                else:
                    lower_bound.append(means[i])
                    upper_bound.append(means[i])
            
            # Fill confidence interval
            ax.fill_between(n_values, lower_bound, upper_bound, alpha=0.2, 
                           label=f'{int(confidence_level*100)}% CI')
        
        # Add horizontal line at y=1 (perfect convergence)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect (IR=1)')
        
        ax.set_xlabel('Number of Agents (n)', fontsize=11)
        ax.set_ylabel('Informed Ratio', fontsize=11)
        ax.set_title(f'{rule_name} Convergence', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.15])
    
    # Hide unused subplots
    for idx in range(n_rules, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Asymptotic Convergence Analysis\n'
                 f'(m={m}, α={alpha}, B={budget}, utility={utility_type})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    filename = f'convergence_m{m}_alpha{alpha}_B{budget}_{utility_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filename}'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Simulation parameters
    n_values = [20, 50, 100, 200, 500, 1000, 2000]  # Extended range for convergence
    m = 8  # number of alternatives
    alpha = 1.0  # Unit cost for convergence analysis
    budget = 15.0
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
    
    # Run simulation
    results, rule_names = run_convergence_analysis(
        n_values, m, alpha, budget, quality_range,
        utility_type, num_samples=30, num_trials=10  # More trials for better statistics
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

