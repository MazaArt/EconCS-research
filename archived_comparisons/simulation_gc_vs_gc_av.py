"""
GC vs GC + AV Comparison Simulation

This script compares the informed ratio performance between standard Greedy Cover (GC)
and a hybrid approach: GC followed by AV for budget exhaustion.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set

# Import functions from the main simulation module
from simulation import (
    approval_voting, greedy_cover, gc_plus_av,
    calculate_informed_ratio, generate_instance
)


def run_gc_vs_gc_av_comparison(n_values: List[int], m: int, alpha: float,
                               budget: float, quality_range: Tuple[int, int],
                               utility_type: str = 'normal',
                               num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation comparing GC vs GC + AV.
    
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
        'GC': greedy_cover,
        'GC + AV': gc_plus_av
    }
    
    results = {
        rule: {'mean': [], 'std': [], 'all': []}  # Added 'all' to store trial data
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
            results[rule_name]['all'].append(ratios.copy())  # Store all trial data
            print(f"{rule_name}: {np.mean(ratios):.4f}±{np.std(ratios):.4f}  ", end='', flush=True)
        print("done")
    
    return results, voting_rules.keys()


def plot_gc_vs_gc_av(n_values: List[int], results: dict, rule_names: List[str],
                     m: int, alpha: float, budget: float, utility_type: str,
                     filename: str = None):
    """Plot comparison between GC and GC + AV with error bars."""
    import os
    
    # Control flag: Set to True to show STD bars, False to show only means
    SHOW_STD_BARS = True   # Uncomment this line to enable STD bars
    # SHOW_STD_BARS = False    # Comment out this line to disable STD bars
    
    plt.figure(figsize=(12, 8))
    
    # Define styles for each rule
    rule_styles = {
        'GC': {'marker': 'o', 'linestyle': '-', 'color': '#ff7f0e'},
        'GC + AV': {'marker': 's', 'linestyle': '--', 'color': '#e377c2'},
        'GC+AV': {'marker': 's', 'linestyle': '--', 'color': '#e377c2'}
    }
    
    # Plot for each rule
    for rule_name in rule_names:
        means = results[rule_name]['mean']
        stds = results[rule_name]['std']
        style = rule_styles.get(rule_name, {'marker': 'o', 'linestyle': '-', 'color': 'black'})
        if SHOW_STD_BARS:
            # Plot error bars first (transparent)
            plt.errorbar(n_values, means, yerr=stds,
                        linestyle='none', color=style['color'],
                        capsize=5, capthick=1.5, elinewidth=1.5,
                        alpha=0.4)
            # Plot main line and markers on top (fully opaque)
            plt.plot(n_values, means,
                    marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], label=rule_name,
                    linewidth=3, markersize=8)
        else:
            plt.plot(n_values, means,
                        marker=style['marker'], linestyle=style['linestyle'],
                        color=style['color'], label=rule_name,
                        linewidth=3, markersize=8)
    
    # Add horizontal line at y=1 (perfect performance)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (Performance=1)')
    
    plt.xlabel('Number of Agents (n)', fontsize=24)
    plt.ylabel('Performance', fontsize=24)
    plt.title(f'GC vs GC + AV Comparison\n'
              f'(m={m}, α={alpha}, B={budget}, utility={utility_type})',
              fontsize=28, fontweight='bold')
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to use most of the visual space
    all_means = [m for r in results.values() for m in r['mean']]
    if SHOW_STD_BARS:
        all_stds = [s for r in results.values() for s in r['std']]
        y_min = max(0, min(all_means) - 2 * max(all_stds) if all_stds else 0)
        y_max = min(1.05, max(all_means) + 2 * max(all_stds) if all_stds else 1.05)
    else:
        y_min = max(0, min(all_means) - 0.05)
        y_max = min(1.05, max(all_means) + 0.05)
    plt.ylim([y_min, y_max])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'gc_vs_gc_av_m{m}_alpha{alpha}_B{budget}_{utility_type}_{timestamp}.png'
    # Save to plots/simulation_gc_vs_gc_av folder
    os.makedirs('plots/simulation_gc_vs_gc_av', exist_ok=True)
    filepath = os.path.join('plots/simulation_gc_vs_gc_av', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filepath}'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Import statistical analysis module
    from statistical_analysis import (
        run_pairwise_tests, print_statistical_results, 
        print_win_matrix, print_effect_size_interpretation
    )
    
    # Simulation parameters
    n_values = list(range(10, 201, 10))  # n=10 to n=200 in steps of 10
    m = 8  # number of alternatives
    alpha = 5.0  # cost ratio (max/min) - non-unit cost simulation
    budget = 8.0
    quality_range = (0, 2)  # binary qualities
    utility_type = 'normal'  # or 'cost_proportional'
    
    print("=" * 60)
    print("GC vs GC + AV Comparison Simulation")
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
    results, rule_names = run_gc_vs_gc_av_comparison(
        n_values, m, alpha, budget, quality_range,
        utility_type, num_samples=30, num_trials=100
    )
    
    # Plot results
    plot_gc_vs_gc_av(n_values, results, rule_names, m, alpha, budget, utility_type)
    
    # Statistical Analysis
    test_results, win_counts = run_pairwise_tests(results, rule_names, n_values, x_label='n')
    print_statistical_results(test_results, win_counts, rule_names, n_values, x_label='n')
    print_win_matrix(win_counts, rule_names, n_values)
    print_effect_size_interpretation()
    
    print("\nSimulation complete!")
