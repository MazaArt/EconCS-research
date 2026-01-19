"""
Statistical Analysis Utilities for PB Simulations

This module provides:
1. Paired t-test for comparing voting rules
2. Formatted output of statistical results
3. Summary matrix of significant comparisons
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy import stats
from itertools import combinations
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def perform_paired_ttest(data_A: List[float], data_B: List[float], 
                         alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform paired t-test between two voting rules.
    
    Args:
        data_A: Performance data for rule A (list of trial results)
        data_B: Performance data for rule B (list of trial results)
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary with test results
    """
    if len(data_A) != len(data_B) or len(data_A) < 2:
        return {
            'valid': False,
            't_stat': np.nan,
            'p_value': np.nan,
            'significant': False,
            'mean_diff': np.nan,
            'std_diff': np.nan,
            'cohens_d': np.nan,
            'winner': None
        }
    
    data_A = np.array(data_A)
    data_B = np.array(data_B)
    D = data_A - data_B
    
    # Paired t-test (two-sided)
    t_stat, p_value = stats.ttest_rel(data_A, data_B)
    
    # Effect size: Cohen's d for paired data
    mean_diff = np.mean(D)
    std_diff = np.std(D, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    # Determine winner (if significant)
    significant = p_value < alpha
    if significant:
        winner = 'A' if mean_diff > 0 else 'B'
    else:
        winner = None
    
    return {
        'valid': True,
        't_stat': t_stat,
        'p_value': p_value,
        'significant': significant,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'cohens_d': cohens_d,
        'winner': winner
    }


def run_pairwise_tests(results: Dict, rule_names: List[str], 
                       x_values: List, x_label: str = 'n',
                       alpha: float = 0.05) -> Tuple[Dict, Dict]:
    """
    Run paired t-tests for all pairs of voting rules at each x value.
    
    Args:
        results: Dictionary with structure {rule: {'all': [[trial_results], ...]}}
        rule_names: List of rule names to compare
        x_values: List of x-axis values (e.g., n values)
        x_label: Label for x-axis (for output)
        alpha: Significance level
    
    Returns:
        - test_results: Detailed results for each pair and x value
        - win_matrix: Count of significant wins for each pair across x values
    """
    rule_names = list(rule_names)
    pairs = list(combinations(rule_names, 2))
    
    # Initialize results storage
    test_results = {pair: [] for pair in pairs}
    win_counts = {pair: {'A_wins': 0, 'B_wins': 0, 'no_diff': 0} for pair in pairs}
    
    for idx, x in enumerate(x_values):
        for rule_A, rule_B in pairs:
            # Get trial data for this x value
            data_A = results[rule_A]['all'][idx]
            data_B = results[rule_B]['all'][idx]
            
            # Perform paired t-test
            test_result = perform_paired_ttest(data_A, data_B, alpha)
            test_result[x_label] = x
            test_results[(rule_A, rule_B)].append(test_result)
            
            # Update win counts
            if test_result['significant']:
                if test_result['winner'] == 'A':
                    win_counts[(rule_A, rule_B)]['A_wins'] += 1
                else:
                    win_counts[(rule_A, rule_B)]['B_wins'] += 1
            else:
                win_counts[(rule_A, rule_B)]['no_diff'] += 1
    
    return test_results, win_counts


def print_statistical_results(test_results: Dict, win_counts: Dict,
                              rule_names: List[str], x_values: List,
                              x_label: str = 'n', alpha: float = 0.05):
    """
    Print formatted statistical results.
    
    Args:
        test_results: Detailed test results from run_pairwise_tests
        win_counts: Win count summary from run_pairwise_tests
        rule_names: List of rule names
        x_values: List of x-axis values
        x_label: Label for x-axis
        alpha: Significance level used
    """
    rule_names = list(rule_names)
    pairs = list(combinations(rule_names, 2))
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS: Paired t-tests (α = {:.2f})".format(alpha))
    print("=" * 80)
    
    # Print summary for each pair
    for rule_A, rule_B in pairs:
        print(f"\n{'─' * 80}")
        print(f"Comparison: {rule_A} vs {rule_B}")
        print(f"{'─' * 80}")
        
        # Header
        cohens_d_label = "Cohen's d"
        print(f"{'':>8} | {'Mean Diff':>10} | {'Std Diff':>10} | {'t-stat':>10} | "
              f"{'p-value':>12} | {cohens_d_label:>10} | {'Result':>12}")
        print("-" * 80)
        
        for result in test_results[(rule_A, rule_B)]:
            x_val = result[x_label]
            if not result['valid']:
                print(f"{x_label}={x_val:>4} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | "
                      f"{'N/A':>12} | {'N/A':>10} | {'Invalid':>12}")
                continue
            
            mean_diff = result['mean_diff']
            std_diff = result['std_diff']
            t_stat = result['t_stat']
            p_value = result['p_value']
            cohens_d = result['cohens_d']
            
            if result['significant']:
                winner = rule_A if result['winner'] == 'A' else rule_B
                sig_marker = f"{winner} wins*"
            else:
                sig_marker = "No sig. diff"
            
            print(f"{x_label}={x_val:>4} | {mean_diff:>10.4f} | {std_diff:>10.4f} | {t_stat:>10.2f} | "
                  f"{p_value:>12.2e} | {cohens_d:>10.3f} | {sig_marker:>12}")
        
        # Summary for this pair
        wins_A = win_counts[(rule_A, rule_B)]['A_wins']
        wins_B = win_counts[(rule_A, rule_B)]['B_wins']
        no_diff = win_counts[(rule_A, rule_B)]['no_diff']
        print(f"\nSummary: {rule_A} wins {wins_A}/{len(x_values)}, "
              f"{rule_B} wins {wins_B}/{len(x_values)}, "
              f"No significant difference {no_diff}/{len(x_values)}")


def print_win_matrix(win_counts: Dict, rule_names: List[str], x_values: List):
    """
    Print a matrix showing win counts for all rule pairs.
    
    Args:
        win_counts: Win count summary from run_pairwise_tests
        rule_names: List of rule names
        x_values: List of x-axis values
    """
    rule_names = list(rule_names)
    n_total = len(x_values)
    
    print("\n" + "=" * 80)
    print("WIN MATRIX: Number of significant wins across all {} conditions".format(n_total))
    print("(Row rule beats Column rule)")
    print("=" * 80)
    
    # Calculate column width based on rule name length
    max_name_len = max(len(name) for name in rule_names)
    col_width = max(max_name_len + 2, 10)
    
    # Header row
    header = " " * col_width + "|"
    for rule in rule_names:
        header += f" {rule:^{col_width}} |"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for rule_A in rule_names:
        row = f"{rule_A:<{col_width}}|"
        for rule_B in rule_names:
            if rule_A == rule_B:
                row += f" {'--':^{col_width}} |"
            else:
                # Find the pair (order matters for who wins)
                if (rule_A, rule_B) in win_counts:
                    wins = win_counts[(rule_A, rule_B)]['A_wins']
                elif (rule_B, rule_A) in win_counts:
                    wins = win_counts[(rule_B, rule_A)]['B_wins']
                else:
                    wins = 0
                row += f" {wins:^{col_width}} |"
        print(row)
    
    print("\n* Note: Cell (i,j) shows how many times rule i significantly outperformed rule j")


def print_effect_size_interpretation():
    """Print Cohen's d interpretation guide."""
    print("\n" + "-" * 40)
    print("Cohen's d interpretation:")
    print("  |d| ≈ 0.2 : Small effect")
    print("  |d| ≈ 0.5 : Medium effect")
    print("  |d| ≈ 0.8 : Large effect")
    print("-" * 40)


# ============================================================================
# Helper function to modify results structure
# ============================================================================

def ensure_all_trials_stored(results: Dict, rule_names: List[str]) -> Dict:
    """
    Ensure results dictionary has 'all' field with trial data.
    If not present, creates empty structure (won't work for t-test).
    
    Args:
        results: Original results dictionary
        rule_names: List of rule names
    
    Returns:
        Modified results dictionary with 'all' field
    """
    for rule in rule_names:
        if 'all' not in results[rule]:
            print(f"Warning: No trial data found for {rule}. Statistical tests will be invalid.")
            # Create placeholder with single values (won't work for t-test)
            results[rule]['all'] = [[m] for m in results[rule]['mean']]
    return results
