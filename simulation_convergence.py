"""
Convergence Analysis Simulation

This script analyzes the asymptotic convergence of performance (ratio to optimal knapsack) as n increases,
with confidence intervals and error bars.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import functions from the main simulation module
from simulation import (
    approval_voting, approval_voting_per_cost, greedy_cover,
    gc_plus_av, method_of_equal_shares, mes_plus_av, mes_plus_phragmen, phragmen,
    proportional_approval_voting, calculate_informed_ratio, generate_instance
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
        'AV/Cost': approval_voting_per_cost,
        'GC': greedy_cover,
        'GC+AV': gc_plus_av,
        'MES': method_of_equal_shares,
        'MES+AV': mes_plus_av,
        'MES+Phragmen': mes_plus_phragmen,
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


def test_convergence_to_one(results: dict, rule_names: List[str], n_values: List[int],
                            epsilon: float = 0.02, alpha_level: float = 0.05) -> dict:
    """
    Statistical hypothesis testing for convergence to 1.
    
    Tests performed for each rule at each n:
    1. One-sample t-test: H0: μ = 1 vs H1: μ < 1
       - If p > alpha, cannot reject H0, suggests mean could be 1
    2. Equivalence test (TOST): H0: |μ - 1| >= ε vs H1: |μ - 1| < ε
       - If p < alpha, reject H0, conclude mean is within ε of 1
    3. Confidence interval check: Does 95% CI include 1?
    
    "Sufficiently close to 1" definition:
    - Primary: TOST p-value < alpha (equivalence established within ε)
    - Secondary: 95% CI lower bound >= 1 - ε (practical equivalence)
    
    Args:
        results: Dictionary with 'all' containing trial data
        rule_names: List of rule names
        n_values: List of n values tested
        epsilon: Equivalence margin (default 0.02, i.e., within 2% of 1)
        alpha_level: Significance level (default 0.05)
    
    Returns:
        Dictionary with test results for each rule and n
    """
    from scipy import stats
    
    test_results = {rule: {} for rule in rule_names}
    
    print("\n" + "=" * 80)
    print("CONVERGENCE HYPOTHESIS TESTING")
    print("=" * 80)
    print(f"Epsilon (equivalence margin): {epsilon}")
    print(f"Alpha level: {alpha_level}")
    print(f"'Sufficiently close to 1' means: performance within [{1-epsilon:.3f}, 1.000]")
    print("=" * 80)
    
    for rule_name in rule_names:
        print(f"\n{'─' * 40}")
        print(f"Rule: {rule_name}")
        print(f"{'─' * 40}")
        
        test_results[rule_name] = {
            'n_values': n_values,
            't_test_p': [],
            'tost_p': [],
            'ci_lower': [],
            'ci_upper': [],
            'is_converged': [],
            'mean': [],
            'std': []
        }
        
        for i, n in enumerate(n_values):
            data = np.array(results[rule_name]['all'][i])
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            n_samples = len(data)
            se = std / np.sqrt(n_samples) if n_samples > 0 else 1e-10
            
            # 1. One-sample t-test (H0: μ = 1, H1: μ < 1)
            if n_samples > 1 and se > 1e-10:
                t_stat, t_pvalue_two = stats.ttest_1samp(data, 1.0)
                t_pvalue = t_pvalue_two / 2 if t_stat < 0 else 1 - t_pvalue_two / 2
            else:
                t_pvalue = 1.0 if mean >= 1.0 else 0.0
            
            # 2. TOST for equivalence
            if n_samples > 1 and se > 1e-10:
                t_lower = (mean - (1 - epsilon)) / se
                p_lower = 1 - stats.t.cdf(t_lower, n_samples - 1)
                t_upper = (mean - (1 + epsilon)) / se
                p_upper = stats.t.cdf(t_upper, n_samples - 1)
                tost_p = max(p_lower, p_upper)
            else:
                tost_p = 0.0 if abs(mean - 1.0) < epsilon else 1.0
            
            # 3. Confidence interval
            if n_samples > 1:
                t_critical = stats.t.ppf(1 - alpha_level / 2, n_samples - 1)
                ci_lower = mean - t_critical * se
                ci_upper = mean + t_critical * se
            else:
                ci_lower = ci_upper = mean
            
            # Determine convergence
            is_converged = (tost_p < alpha_level) or (ci_lower >= 1 - epsilon)
            
            test_results[rule_name]['t_test_p'].append(t_pvalue)
            test_results[rule_name]['tost_p'].append(tost_p)
            test_results[rule_name]['ci_lower'].append(ci_lower)
            test_results[rule_name]['ci_upper'].append(ci_upper)
            test_results[rule_name]['is_converged'].append(is_converged)
            test_results[rule_name]['mean'].append(mean)
            test_results[rule_name]['std'].append(std)
        
        # Print summary for last few n values
        print(f"\n  {'n':>6} | {'Mean':>7} | {'95% CI':>17} | {'TOST p':>8} | {'Converged?':>10}")
        print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*17}-+-{'-'*8}-+-{'-'*10}")
        for i in range(max(0, len(n_values)-5), len(n_values)):
            n = n_values[i]
            mean = test_results[rule_name]['mean'][i]
            ci_l = test_results[rule_name]['ci_lower'][i]
            ci_u = test_results[rule_name]['ci_upper'][i]
            tost_p = test_results[rule_name]['tost_p'][i]
            converged = test_results[rule_name]['is_converged'][i]
            conv_str = "YES ✓" if converged else "NO"
            print(f"  {n:>6} | {mean:>7.4f} | [{ci_l:.4f}, {ci_u:.4f}] | {tost_p:>8.4f} | {conv_str:>10}")
        
        # Final verdict
        last_converged = test_results[rule_name]['is_converged'][-1]
        last_mean = test_results[rule_name]['mean'][-1]
        if last_converged:
            print(f"\n  ✓ {rule_name} appears to CONVERGE to 1 (mean={last_mean:.4f} is within ε={epsilon} of 1)")
        else:
            print(f"\n  ✗ {rule_name} has NOT YET converged (mean={last_mean:.4f}, need larger n)")
    
    return test_results


def print_convergence_summary(test_results: dict, rule_names: List[str], 
                              n_values: List[int], epsilon: float):
    """Print a summary of convergence test results."""
    print("\n" + "=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)
    print(f"\nDefinition of 'sufficiently close to 1':")
    print(f"  - Performance ratio is within ε = {epsilon} of 1.0")
    print(f"  - i.e., performance ∈ [{1-epsilon:.3f}, 1.000]")
    print(f"  - Verified via TOST (Two One-Sided Tests) at α = 0.05")
    print()
    
    print(f"{'Rule':<12} | {'Final Mean':>10} | {'Converged at n':>15} | {'Status':>15}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*15}-+-{'-'*15}")
    
    for rule_name in rule_names:
        final_mean = test_results[rule_name]['mean'][-1]
        
        # Find first n where convergence is achieved
        converged_n = None
        for i, n in enumerate(n_values):
            if test_results[rule_name]['is_converged'][i]:
                converged_n = n
                break
        
        if converged_n is not None:
            status = "✓ CONVERGED"
            conv_str = f"n ≥ {converged_n}"
        else:
            status = "✗ NOT YET"
            conv_str = "n > " + str(n_values[-1])
        
        print(f"{rule_name:<12} | {final_mean:>10.4f} | {conv_str:>15} | {status:>15}")
    
    print("\n" + "=" * 80)


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
        container = ax.errorbar(n_values, means, yerr=stds,
                   marker='o', linestyle='-', label='Mean ± std',
                   linewidth=3, markersize=8,
                   capsize=5, capthick=1.5, elinewidth=1.5,
                   alpha=0.4)  # Alpha for error bars
        container[0].set_alpha(1.0)  # Make the main line fully opaque
        
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
    # Import statistical analysis module
    from statistical_analysis import (
        run_pairwise_tests, print_statistical_results, 
        print_win_matrix, print_effect_size_interpretation
    )
    
    # Simulation parameters
    # Convergence to 1.0 should only happen under alpha = 1 (unit cost)
    # Budget should be less than number of alternatives to observe convergence
    n_values = list(range(10, 101, 10))  # n=10 to n=200 in steps of 10
    m = 8  # number of alternatives
    alpha = 1.0  # Unit cost - required for convergence to 1.0
    budget = 4.0  # Budget < m (8) - can't afford all alternatives
    quality_range = (0, 2)  # binary qualities
    utility_type = 'cost_proportional'  # or 'normal'
    epsilon = 0.01  # Equivalence margin for convergence test
    
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
    print(f"  epsilon (convergence threshold): {epsilon}")
    print("=" * 60)
    
    # Run simulation for original n values
    results, rule_names = run_convergence_analysis(
        n_values, m, alpha, budget, quality_range,
        utility_type, num_samples=100, num_trials=100 
    )
    
    # Hypothesis testing for convergence to 1
    try:
        from scipy import stats
        test_results = test_convergence_to_one(results, rule_names, n_values, epsilon=epsilon)
        print_convergence_summary(test_results, rule_names, n_values, epsilon)
    except ImportError:
        print("Warning: scipy not available, skipping convergence hypothesis testing")
        test_results = None
    
    # Plot results
    try:
        from scipy import stats
        plot_convergence(n_values, results, rule_names, m, alpha, budget, utility_type,
                        show_confidence=True, confidence_level=0.95)
    except ImportError:
        print("Warning: scipy not available, plotting without confidence intervals")
        plot_convergence(n_values, results, rule_names, m, alpha, budget, utility_type,
                        show_confidence=False, confidence_level=0.95)
    
    # Statistical Analysis (pairwise comparisons)
    test_results_pairwise, win_counts = run_pairwise_tests(results, rule_names, n_values, x_label='n')
    print_statistical_results(test_results_pairwise, win_counts, rule_names, n_values, x_label='n')
    print_win_matrix(win_counts, rule_names, n_values)
    print_effect_size_interpretation()
    
    print("\nSimulation complete!")

