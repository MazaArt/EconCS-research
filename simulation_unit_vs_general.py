"""
Unit Cost vs General Cost Comparison Simulation

This script compares the informed ratio performance between unit cost (alpha=1)
and general cost (alpha>1) settings.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import functions from the main simulation module
from simulation import (
    approval_voting, approval_voting_per_cost, greedy_cover, gc_plus_av,
    method_of_equal_shares, mes_plus_av, phragmen, proportional_approval_voting,
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
        'AV/Cost': approval_voting_per_cost,
        # 'GC': greedy_cover,  # Commented out - use GC+AV instead
        'GC+AV': gc_plus_av,
        # 'MES': method_of_equal_shares,  # Commented out - use MES+AV instead
        'MES+AV': mes_plus_av,
        'Phragmen': phragmen
    }
    
    # Try PAV for small instances
    if m <= 12:
        voting_rules['PAV'] = proportional_approval_voting
    
    results = {
        alpha: {rule: {'mean': [], 'std': [], 'all': []} for rule in voting_rules.keys()}
        for alpha in alpha_values
    }
    
    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"Running simulation for α={alpha}")
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
            
            # Calculate mean and std over trials
            for rule_name in voting_rules.keys():
                ratios = rule_ratios[rule_name]
                results[alpha][rule_name]['mean'].append(np.mean(ratios))
                results[alpha][rule_name]['std'].append(np.std(ratios))
                results[alpha][rule_name]['all'].append(ratios.copy())  # Store all trial data
            print("done")
    
    return results, voting_rules.keys()


def plot_unit_vs_general(n_values: List[int], results: dict, rule_names: List[str],
                         alpha_values: List[float], m: int, budget: float,
                         utility_type: str, filename: str = None):
    """Plot unit cost vs general cost comparison with error bars."""
    import os
    
    # Control flag: Set to True to show STD bars, False to show only means
    SHOW_STD_BARS = True   # Uncomment this line to enable STD bars
    # SHOW_STD_BARS = False    # Comment out this line to disable STD bars
    
    n_plots = len(rule_names)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Define markers and linestyles for alpha values
    alpha_styles = {
        1.0: {'marker': 'o', 'linestyle': '-', 'color': '#1f77b4'},
        2.0: {'marker': 's', 'linestyle': '--', 'color': '#ff7f0e'},
        3.0: {'marker': '^', 'linestyle': '-.', 'color': '#2ca02c'},
        5.0: {'marker': 'v', 'linestyle': ':', 'color': '#d62728'}
    }
    
    for idx, rule_name in enumerate(rule_names):
        ax = axes[idx]
        
        # Plot for each alpha value
        for alpha in alpha_values:
            means = results[alpha][rule_name]['mean']
            stds = results[alpha][rule_name]['std']
            style = alpha_styles.get(alpha, {'marker': 'o', 'linestyle': '-', 'color': 'black'})
            label = f"α={alpha}" + (" (unit cost)" if alpha == 1.0 else " (general cost)")
            if SHOW_STD_BARS:
                container = ax.errorbar(n_values, means, yerr=stds,
                           marker=style['marker'], linestyle=style['linestyle'],
                           color=style['color'], label=label, linewidth=3, markersize=8,
                           capsize=5, capthick=1.5, elinewidth=1.5,
                           alpha=0.4)  # Alpha for error bars
                container[0].set_alpha(1.0)  # Make the main line fully opaque
            else:
                ax.plot(n_values, means,
                           marker=style['marker'], linestyle=style['linestyle'],
                           color=style['color'], label=label, linewidth=3, markersize=8)
        
        # Add horizontal line at y=1 (perfect performance)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (Performance=1)')
        
        ax.set_xlabel('Number of Agents (n)', fontsize=22)
        ax.set_ylabel('Performance', fontsize=22)
        ax.set_title(f'{rule_name}', fontsize=24, fontweight='bold')
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)
        
        # Adjust y-axis to use most of the visual space
        all_means = [m for alpha in alpha_values for m in results[alpha][rule_name]['mean']]
        if SHOW_STD_BARS:
            all_stds = [s for alpha in alpha_values for s in results[alpha][rule_name]['std']]
            y_min = max(0, min(all_means) - 2 * max(all_stds) if all_stds else 0)
            y_max = min(1.05, max(all_means) + 2 * max(all_stds) if all_stds else 1.05)
        else:
            y_min = max(0, min(all_means) - 0.05)
            y_max = min(1.05, max(all_means) + 0.05)
        ax.set_ylim([y_min, y_max])
        ax.tick_params(labelsize=18)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Unit Cost (alpha=1) vs General Cost (alpha>1) Comparison\n'
                 f'(m={m}, B={budget}, utility={utility_type})',
                 fontsize=28, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'unit_vs_general_cost_m{m}_B{budget}_{utility_type}_{timestamp}.png'
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
    # Import statistical analysis module
    from statistical_analysis import (
        run_pairwise_tests, print_statistical_results, 
        print_win_matrix, print_effect_size_interpretation,
        perform_paired_ttest
    )
    from itertools import combinations
    
    # Simulation parameters
    n_values = list(range(10, 101, 10))  # n=10 to n=200 in steps of 10
    m = 8  # number of alternatives
    alpha_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # 1.0 = unit cost, >1.0 = general cost
    budget = 7.0
    quality_range = (0, 2)  # binary qualities
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
        utility_type, num_samples=30, num_trials=100
    )
    
    # Plot results
    plot_unit_vs_general(n_values, results, rule_names, alpha_values, m, budget, utility_type)
    
    # =========================================================================
    # Statistical Analysis Part 1: For each alpha, compare different rules
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Comparing RULES within each alpha")
    print("=" * 80)
    
    for alpha in alpha_values:
        print(f"\n{'#' * 80}")
        print(f"# STATISTICAL ANALYSIS FOR α = {alpha}")
        print(f"{'#' * 80}")
        test_results, win_counts = run_pairwise_tests(results[alpha], rule_names, n_values, x_label='n')
        print_statistical_results(test_results, win_counts, rule_names, n_values, x_label='n')
        print_win_matrix(win_counts, rule_names, n_values)
    
    # =========================================================================
    # Statistical Analysis Part 2: For each rule, compare different alphas
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Comparing ALPHAS within each rule")
    print("=" * 80)
    
    alpha_pairs = list(combinations(alpha_values, 2))
    rule_names_list = list(rule_names)
    
    for rule_name in rule_names_list:
        print(f"\n{'#' * 80}")
        print(f"# STATISTICAL ANALYSIS FOR RULE: {rule_name}")
        print(f"{'#' * 80}")
        
        # Storage for this rule's cross-alpha comparisons
        cross_alpha_results = {pair: [] for pair in alpha_pairs}
        cross_alpha_wins = {pair: {'A_wins': 0, 'B_wins': 0, 'no_diff': 0} for pair in alpha_pairs}
        
        for alpha_A, alpha_B in alpha_pairs:
            print(f"\n{'─' * 80}")
            print(f"Comparison: α={alpha_A} vs α={alpha_B}")
            print(f"{'─' * 80}")
            
            cohens_d_label = "Cohen's d"
            print(f"{'':>8} | {'Mean Diff':>10} | {'Std Diff':>10} | {'t-stat':>10} | "
                  f"{'p-value':>12} | {cohens_d_label:>10} | {'Result':>12}")
            print("-" * 85)
            
            for idx, n in enumerate(n_values):
                data_A = results[alpha_A][rule_name]['all'][idx]
                data_B = results[alpha_B][rule_name]['all'][idx]
                
                test_result = perform_paired_ttest(data_A, data_B, alpha=0.05)
                test_result['n'] = n
                cross_alpha_results[(alpha_A, alpha_B)].append(test_result)
                
                if not test_result['valid']:
                    print(f"n={n:>4} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | "
                          f"{'N/A':>12} | {'N/A':>10} | {'Invalid':>12}")
                    continue
                
                mean_diff = test_result['mean_diff']
                std_diff = test_result['std_diff']
                t_stat = test_result['t_stat']
                p_value = test_result['p_value']
                cohens_d = test_result['cohens_d']
                
                if test_result['significant']:
                    if test_result['winner'] == 'A':
                        winner_label = f"a={alpha_A} wins*"
                        cross_alpha_wins[(alpha_A, alpha_B)]['A_wins'] += 1
                    else:
                        winner_label = f"a={alpha_B} wins*"
                        cross_alpha_wins[(alpha_A, alpha_B)]['B_wins'] += 1
                else:
                    winner_label = "No sig. diff"
                    cross_alpha_wins[(alpha_A, alpha_B)]['no_diff'] += 1
                
                print(f"n={n:>4} | {mean_diff:>10.4f} | {std_diff:>10.4f} | {t_stat:>10.2f} | "
                      f"{p_value:>12.2e} | {cohens_d:>10.3f} | {winner_label:>12}")
            
            # Summary for this alpha pair
            wins_A = cross_alpha_wins[(alpha_A, alpha_B)]['A_wins']
            wins_B = cross_alpha_wins[(alpha_A, alpha_B)]['B_wins']
            no_diff = cross_alpha_wins[(alpha_A, alpha_B)]['no_diff']
            print(f"\nSummary: α={alpha_A} wins {wins_A}/{len(n_values)}, "
                  f"α={alpha_B} wins {wins_B}/{len(n_values)}, "
                  f"No significant difference {no_diff}/{len(n_values)}")
        
        # Win matrix for this rule across alphas
        print(f"\n{'=' * 60}")
        print(f"WIN MATRIX FOR {rule_name}: alpha comparisons across {len(n_values)} n values")
        print(f"(Row alpha beats Column alpha)")
        print(f"{'=' * 60}")
        
        col_width = 10
        header = " " * col_width + "|"
        for alpha in alpha_values:
            header += f" a={alpha:<{col_width-3}} |"
        print(header)
        print("-" * len(header))
        
        for alpha_A in alpha_values:
            row = f"a={alpha_A:<{col_width-3}}|"
            for alpha_B in alpha_values:
                if alpha_A == alpha_B:
                    row += f" {'--':^{col_width}} |"
                else:
                    if (alpha_A, alpha_B) in cross_alpha_wins:
                        wins = cross_alpha_wins[(alpha_A, alpha_B)]['A_wins']
                    elif (alpha_B, alpha_A) in cross_alpha_wins:
                        wins = cross_alpha_wins[(alpha_B, alpha_A)]['B_wins']
                    else:
                        wins = 0
                    row += f" {wins:^{col_width}} |"
            print(row)
    
    print_effect_size_interpretation()
    
    print("\nSimulation complete!")

