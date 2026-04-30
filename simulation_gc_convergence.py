"""
GC (Greedy Cover) Convergence Analysis Simulation

This script specifically tests whether GC converges to optimal performance (ratio = 1)
as the number of agents n increases, with statistical hypothesis testing.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy import stats

# Import functions from the main simulation module
from simulation import (
    approval_voting, approval_voting_per_cost, greedy_cover,
    method_of_equal_shares, mes_plus_av, mes_plus_phragmen, phragmen,
    proportional_approval_voting,
    calculate_informed_ratio, generate_instance
)


def run_gc_convergence_analysis(n_values: List[int], m: int, alpha: float,
                                   budget: float, quality_range: Tuple[int, int],
                                   utility_type: str = 'normal',
                                   num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run convergence analysis across PB aggregation methods.
    
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
        'MES': method_of_equal_shares,
        'MES+AV': mes_plus_av,
        'MES+Phragmen': mes_plus_phragmen,
        'Phragmen': phragmen
    }
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
        print(f"Running simulation for n={n}...", end=' ', flush=True)
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
                    print(f"\n  Error in {rule_name}: {e}")
                    rule_ratios[rule_name].append(0.0)
        
        # Calculate statistics
        for rule_name in voting_rules.keys():
            ratios = rule_ratios[rule_name]
            results[rule_name]['mean'].append(np.mean(ratios))
            results[rule_name]['std'].append(np.std(ratios))
            results[rule_name]['all'].append(ratios)
        
        summary = ", ".join(f"{rule}={results[rule]['mean'][-1]:.4f}" for rule in voting_rules.keys())
        print(summary)
    
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
    - Secondary: 95% CI upper bound >= 1 - ε (practical equivalence)
    
    Args:
        results: Dictionary with 'all' containing trial data
        rule_names: List of rule names
        n_values: List of n values tested
        epsilon: Equivalence margin (default 0.02, i.e., within 2% of 1)
        alpha_level: Significance level (default 0.05)
    
    Returns:
        Dictionary with test results for each rule and n
    """
    test_results = {rule: {} for rule in rule_names}
    
    print("\n" + "=" * 80)
    print("CONVERGENCE HYPOTHESIS TESTING")
    print("=" * 80)
    print(f"Epsilon (equivalence margin): {epsilon}")
    print(f"Alpha level: {alpha_level}")
    print(f"'Sufficiently close to 1' means: performance within [{1-epsilon}, 1] = [{1-epsilon:.3f}, 1.000]")
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
            se = std / np.sqrt(n_samples)
            
            # 1. One-sample t-test (H0: μ = 1, H1: μ < 1)
            t_stat, t_pvalue_two = stats.ttest_1samp(data, 1.0)
            t_pvalue = t_pvalue_two / 2 if t_stat < 0 else 1 - t_pvalue_two / 2  # One-sided
            
            # 2. TOST (Two One-Sided Tests) for equivalence
            # H0: μ <= 1-ε OR μ >= 1+ε (not equivalent)
            # H1: 1-ε < μ < 1+ε (equivalent)
            # Test 1: H0: μ <= 1-ε, H1: μ > 1-ε
            t_lower = (mean - (1 - epsilon)) / se
            p_lower = 1 - stats.t.cdf(t_lower, n_samples - 1)
            # Test 2: H0: μ >= 1+ε, H1: μ < 1+ε (always true since mean <= 1 typically)
            t_upper = (mean - (1 + epsilon)) / se
            p_upper = stats.t.cdf(t_upper, n_samples - 1)
            # TOST p-value is max of the two
            tost_p = max(p_lower, p_upper)
            
            # 3. Confidence interval
            t_critical = stats.t.ppf(1 - alpha_level / 2, n_samples - 1)
            ci_lower = mean - t_critical * se
            ci_upper = mean + t_critical * se
            
            # Determine convergence
            # Converged if: TOST rejects H0 (tost_p < alpha) OR CI_lower >= 1 - epsilon
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


def plot_gc_convergence(n_values: List[int], results: dict, rule_names: List[str],
                           test_results: dict, m: int, alpha: float, budget: float, 
                           utility_type: str, epsilon: float = 0.02, filename: str = None):
    """Plot convergence analysis for GC and AV with hypothesis test results."""
    import os
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    rule_styles = {
        'GC': {'color': '#ff7f0e', 'marker': 's'},
        'AV': {'color': '#1f77b4', 'marker': 'o'}
    }
    
    for idx, rule_name in enumerate(rule_names):
        ax = axes[idx]
        
        means = results[rule_name]['mean']
        stds = results[rule_name]['std']
        style = rule_styles.get(rule_name, {'color': 'black', 'marker': 'o'})
        
        # Plot with error bars
        ax.errorbar(n_values, means, yerr=stds,
                   marker=style['marker'], linestyle='-', color=style['color'],
                   linewidth=2, markersize=8,
                   capsize=4, capthick=1.5, elinewidth=1.5,
                   alpha=0.7, label='Mean ± std')
        
        # Plot confidence intervals
        ci_lower = test_results[rule_name]['ci_lower']
        ci_upper = test_results[rule_name]['ci_upper']
        ax.fill_between(n_values, ci_lower, ci_upper, alpha=0.2, color=style['color'],
                       label='95% CI')
        
        # Add horizontal lines
        ax.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Perfect (1.0)')
        ax.axhline(y=1.0 - epsilon, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
                  label=f'Threshold (1-ε={1-epsilon:.2f})')
        
        # Mark converged points
        for i, n in enumerate(n_values):
            if test_results[rule_name]['is_converged'][i]:
                ax.scatter([n], [means[i]], color='green', s=100, zorder=5, marker='o', 
                          edgecolors='darkgreen', linewidths=2)
        
        ax.set_xlabel('Number of Agents (n)', fontsize=14)
        ax.set_ylabel('Performance', fontsize=14)
        ax.set_title(f'{rule_name} Convergence', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis
        y_min = max(0, min(ci_lower) - 0.05)
        y_max = min(1.1, max(ci_upper) + 0.05)
        ax.set_ylim([y_min, y_max])
        ax.tick_params(labelsize=12)
        
        # Add convergence status
        last_converged = test_results[rule_name]['is_converged'][-1]
        status = "✓ Converged" if last_converged else "✗ Not yet"
        ax.text(0.05, 0.95, status, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontweight='bold',
               color='green' if last_converged else 'red')
    
    plt.suptitle(f'GC Convergence Analysis (m={m}, α={alpha}, B={budget}, ε={epsilon})',
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'gc_convergence_m{m}_alpha{alpha}_B{budget}_{timestamp}.png'
    
    os.makedirs('plots/simulation_gc_convergence', exist_ok=True)
    filepath = os.path.join('plots/simulation_gc_convergence', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filepath}'")
    plt.show()


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
    
    print(f"{'Rule':<10} | {'Final Mean':>10} | {'Converged at n':>15} | {'Status':>15}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*15}-+-{'-'*15}")
    
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
        
        print(f"{rule_name:<10} | {final_mean:>10.4f} | {conv_str:>15} | {status:>15}")
    
    print("\n" + "=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Simulation parameters
    n_values = list(range(1000, 10000, 1000))  # n=50 to n=1000 in steps of 50
    m = 8  # number of alternatives
    alpha = 1.0  # Unit cost - required for theoretical convergence to 1.0
    budget = 4.0  # Budget < m
    quality_range = (0, 2)  # qualities in {0, 1, 2}
    utility_type = 'cost_proportional'  # or 'normal'
    epsilon = 0.01  # Equivalence margin: within 1% of optimal
    
    print("=" * 80)
    print("GC (GREEDY COVER) CONVERGENCE ANALYSIS")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  n values: {n_values[0]} to {n_values[-1]} (step={n_values[1]-n_values[0]})")
    print(f"  m (alternatives): {m}")
    print(f"  alpha (cost ratio): {alpha}")
    print(f"  budget: {budget}")
    print(f"  quality range: {quality_range}")
    print(f"  utility type: {utility_type}")
    print(f"  epsilon (convergence threshold): {epsilon}")
    print("=" * 80)
    
    # Run simulation
    results, rule_names = run_gc_convergence_analysis(
        n_values, m, alpha, budget, quality_range,
        utility_type, num_samples=100, num_trials=100
    )
    
    # Statistical hypothesis testing
    test_results = test_convergence_to_one(results, rule_names, n_values, epsilon=epsilon)
    
    # Plot results
    plot_gc_convergence(n_values, results, rule_names, test_results, 
                          m, alpha, budget, utility_type, epsilon)
    
    # Print summary
    print_convergence_summary(test_results, rule_names, n_values, epsilon)
    
    print("\nSimulation complete!")
