"""
Simulation for Informed Participatory Budgeting

This module implements voting rules (AV, PAV, Greedy Cover, MES, Phragmen),
simulates PB instances, and calculates performance (ratio to optimal knapsack).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Tuple, Set, Callable
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Utility Functions
# ============================================================================

def normal_utility(winning_set: Set[int], qualities: np.ndarray) -> float:
    """Normal utility: sum of qualities."""
    return sum(qualities[j] for j in winning_set)


def cost_proportional_utility(winning_set: Set[int], qualities: np.ndarray, costs: np.ndarray) -> float:
    """Cost-proportional utility: sum of cost * quality."""
    return sum(costs[j] * qualities[j] for j in winning_set)


# ============================================================================
# Voting Rules
# ============================================================================

def approval_voting(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    AV: Select alternatives in descending order of approval counts.
    
    Args:
        votes: (n, m) binary array, votes[i, j] = 1 if agent i approves alternative j
        costs: (m,) array of costs
        budget: budget constraint
    
    Returns:
        Set of winning alternative indices
    """
    m = len(costs)
    approval_counts = votes.sum(axis=0)
    
    # Sort by approval count (descending), break ties by index
    alternatives = sorted(range(m), key=lambda j: (approval_counts[j], -j), reverse=True)
    
    winning_set = set()
    remaining_budget = budget
    
    for j in alternatives:
        if costs[j] <= remaining_budget:
            winning_set.add(j)
            remaining_budget -= costs[j]
    
    return winning_set


def approval_voting_per_cost(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    AV/Cost: Greedily select alternatives by approval_count/cost ratio (descending).
    
    Args:
        votes: (n, m) binary array, votes[i, j] = 1 if agent i approves alternative j
        costs: (m,) array of costs
        budget: budget constraint
    
    Returns:
        Set of winning alternative indices
    """
    m = len(costs)
    approval_counts = votes.sum(axis=0)
    
    # Calculate ratios (approval count / cost)
    ratios = np.zeros(m)
    for j in range(m):
        if costs[j] > 0:
            ratios[j] = approval_counts[j] / costs[j]
        else:
            ratios[j] = np.inf
    
    # Sort by ratio (descending), break ties by approval count, then by index
    alternatives = sorted(range(m), key=lambda j: (ratios[j], approval_counts[j], -j), reverse=True)
    
    winning_set = set()
    remaining_budget = budget
    
    for j in alternatives:
        if costs[j] <= remaining_budget:
            winning_set.add(j)
            remaining_budget -= costs[j]
    
    return winning_set


def proportional_approval_voting(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    PAV (Sequential Greedy): Iteratively select alternative with highest marginal PAV gain.
    
    PAV score = sum over agents of (1 + 1/2 + ... + 1/k) for k approved winners.
    
    Args:
        votes: (n, m) binary array
        costs: (m,) array of costs
        budget: budget constraint
    
    Returns:
        Set of winning alternative indices
    """
    n, m = votes.shape
    winning_set = set()
    remaining_budget = budget
    
    # Track how many approved winners each agent has
    approved_winners = np.zeros(n)
    
    while True:
        best_j = None
        best_gain = -np.inf
        
        for j in range(m):
            if j in winning_set or costs[j] > remaining_budget:
                continue
            
            # Calculate marginal PAV gain: sum of 1/(k+1) for each approver
            gain = 0.0
            for i in range(n):
                if votes[i, j] == 1:
                    gain += 1.0 / (approved_winners[i] + 1)
            
            if gain > best_gain:
                best_gain = gain
                best_j = j
        
        if best_j is None:
            break
        
        winning_set.add(best_j)
        remaining_budget -= costs[best_j]
        for i in range(n):
            if votes[i, best_j] == 1:
                approved_winners[i] += 1
    
    return winning_set


def greedy_cover(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    Greedy Cover: Iteratively select the alternative that covers the most uncovered agents.
    
    An agent is "covered" once at least one alternative they approve has been selected.
    
    Args:
        votes: (n, m) binary array
        costs: (m,) array of costs
        budget: budget constraint
    
    Returns:
        Set of winning alternative indices
    """
    n, m = votes.shape
    winning_set = set()
    remaining_budget = budget
    covered = np.zeros(n, dtype=bool)  # Track covered agents
    remaining = set(range(m))
    
    while remaining:
        best_j = None
        best_new_covered = -1
        
        for j in remaining:
            if costs[j] > remaining_budget:
                continue
            
            # Count how many NEW agents this alternative would cover
            approvers = votes[:, j] == 1
            new_covered = np.sum(approvers & ~covered)
            
            if new_covered > best_new_covered:
                best_new_covered = new_covered
                best_j = j
        
        if best_j is None or best_new_covered == 0:
            break
        
        # Update covered agents
        covered |= (votes[:, best_j] == 1)
        winning_set.add(best_j)
        remaining_budget -= costs[best_j]
        remaining.remove(best_j)
    
    return winning_set


def gc_plus_av(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    Hybrid GC + AV: First run Greedy Cover, then use AV to exhaust remaining budget.
    
    Args:
        votes: (n, m) binary array, votes[i, j] = 1 if agent i approves alternative j
        costs: (m,) array of costs
        budget: budget constraint
    
    Returns:
        Set of winning alternative indices
    """
    # First, run Greedy Cover
    gc_winning_set = greedy_cover(votes, costs, budget)
    
    # Calculate remaining budget
    used_budget = sum(costs[j] for j in gc_winning_set)
    remaining_budget = budget - used_budget
    
    # If no remaining budget, return GC result
    if remaining_budget <= 1e-9:
        return gc_winning_set
    
    # Use AV to fill remaining budget
    m = len(costs)
    approval_counts = votes.sum(axis=0)
    
    # Get alternatives not yet selected
    remaining_alternatives = [j for j in range(m) if j not in gc_winning_set]
    
    if not remaining_alternatives:
        return gc_winning_set
    
    # Sort by approval count (descending), break ties by index
    remaining_alternatives.sort(key=lambda j: (approval_counts[j], -j), reverse=True)
    
    # Add alternatives until budget is exhausted
    for j in remaining_alternatives:
        if costs[j] <= remaining_budget:
            gc_winning_set.add(j)
            remaining_budget -= costs[j]
    
    return gc_winning_set


def method_of_equal_shares(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    MES: Each agent gets budget share B/n. In each round, select the alternative 
    that can be afforded at the lowest maximum contribution (rho) per approver.
    
    Args:
        votes: (n, m) binary array
        costs: (m,) array of costs
        budget: budget constraint
    
    Returns:
        Set of winning alternative indices
    """
    n, m = votes.shape
    budget_shares = np.full(n, budget / n)
    winning_set = set()
    remaining = set(range(m))
    
    while remaining:
        best_j = None
        best_rho = np.inf
        
        for j in remaining:
            approvers = np.where(votes[:, j] == 1)[0]
            if len(approvers) == 0:
                continue
            
            # Check if total available budget from approvers can cover cost
            total_available = sum(budget_shares[i] for i in approvers)
            if total_available < costs[j] - 1e-9:
                continue
            
            # Find minimum rho such that sum of min(budget_shares[i], rho) >= costs[j]
            approver_budgets = sorted([budget_shares[i] for i in approvers])
            
            rho = None
            cumsum = 0.0
            for k, b in enumerate(approver_budgets):
                # Agents 0..k-1 pay their full budget (cumsum)
                # Agents k..end pay rho each
                # Need: cumsum + (len(approvers) - k) * rho >= costs[j]
                needed_rho = (costs[j] - cumsum) / (len(approvers) - k)
                if needed_rho <= b + 1e-9:
                    rho = needed_rho
                    break
                cumsum += b
            
            if rho is not None and rho < best_rho:
                best_rho = rho
                best_j = j
        
        if best_j is None:
            break
        
        # Deduct payments from approvers
        approvers = np.where(votes[:, best_j] == 1)[0]
        for i in approvers:
            payment = min(budget_shares[i], best_rho)
            budget_shares[i] -= payment
        
        winning_set.add(best_j)
        remaining.remove(best_j)
    
    return winning_set


def phragmen(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    Fixed Phragmen implementation.
    Logic: Find x such that sum_{i in approvers} max(x - loads[i], 0) = cost[j]
    """
    n, m = votes.shape
    loads = np.zeros(n)
    winning_set = set()
    remaining = set(range(m))
    spent = 0.0
    
    while remaining:
        best_j = None
        best_x = np.inf
        
        for j in remaining:
            if spent + costs[j] > budget + 1e-9:
                continue
            
            approvers = np.where(votes[:, j] == 1)[0]
            if len(approvers) == 0:
                continue
            
            # 1. Sort loads of approvers ascending
            approver_loads = sorted([loads[i] for i in approvers])
            num_approvers = len(approvers)
            
            # 2. We want to find x. The equation is:
            #    sum_{i where l_i < x} (x - l_i) = cost
            #    k * x - sum(l_i for first k) = cost
            #    x = (cost + sum(l_i for first k)) / k
            
            x = None
            current_sum_loads = 0.0
            
            # Iterate through the number of agents (k) that are "paying" (load < x)
            # k goes from 1 to num_approvers
            for k in range(1, num_approvers + 1):
                # Add the k-th smallest load (index k-1) to the sum
                current_sum_loads += approver_loads[k-1]
                
                # Calculate candidate x assuming exactly k agents pay
                proposed_x = (costs[j] + current_sum_loads) / k
                
                # Check consistency:
                # The proposed x must be larger than the (k-1)-th load (the largest of the paying group)
                # And smaller than or equal to the k-th load (the smallest of the non-paying group, if any)
                
                lower_bound = approver_loads[k-1]
                upper_bound = approver_loads[k] if k < num_approvers else np.inf
                
                if proposed_x >= lower_bound - 1e-9 and proposed_x <= upper_bound + 1e-9:
                    x = proposed_x
                    break
            
            # Fallback (should typically be caught by k=num_approvers case, but for safety)
            if x is None:
                 x = (costs[j] + current_sum_loads) / num_approvers
            if x < best_x:
                best_x = x
                best_j = j
        
        if best_j is None:
            break
            
        # Update loads
        approvers = np.where(votes[:, best_j] == 1)[0]
        for i in approvers:
            loads[i] = max(loads[i], best_x)
            
        winning_set.add(best_j)
        spent += costs[best_j]
        remaining.remove(best_j)
        
    return winning_set


def mes_plus_av(votes: np.ndarray, costs: np.ndarray, budget: float) -> Set[int]:
    """
    Hybrid MES + AV: First run MES, then use AV to exhaust remaining budget.
    
    Args:
        votes: (n, m) binary array, votes[i, j] = 1 if agent i approves alternative j
        costs: (m,) array of costs
        budget: budget constraint
    
    Returns:
        Set of winning alternative indices
    """
    # First, run MES
    mes_winning_set = method_of_equal_shares(votes, costs, budget)
    
    # Calculate remaining budget
    used_budget = sum(costs[j] for j in mes_winning_set)
    remaining_budget = budget - used_budget
    
    # If no remaining budget, return MES result
    if remaining_budget <= 1e-9:
        return mes_winning_set
    
    # Use AV to fill remaining budget
    m = len(costs)
    approval_counts = votes.sum(axis=0)
    
    # Get alternatives not yet selected
    remaining_alternatives = [j for j in range(m) if j not in mes_winning_set]
    
    if not remaining_alternatives:
        return mes_winning_set
    
    # Sort by approval count (descending), break ties by index
    remaining_alternatives.sort(key=lambda j: (approval_counts[j], -j), reverse=True)
    
    # Add alternatives until budget is exhausted
    for j in remaining_alternatives:
        if costs[j] <= remaining_budget:
            mes_winning_set.add(j)
            remaining_budget -= costs[j]
    
    return mes_winning_set


# ============================================================================
# Knapsack Solver (Optimal Set)
# ============================================================================

def knapsack_optimal(qualities: np.ndarray, costs: np.ndarray, budget: float, 
                     use_cost_proportional: bool = False) -> Tuple[Set[int], float]:
    """
    Find optimal winning set using exhaustive search (for small m) or greedy (for large m).
    
    Args:
        qualities: (m,) array of qualities
        costs: (m,) array of costs
        budget: budget constraint
        use_cost_proportional: if True, use cost-proportional utility
    
    Returns:
        (optimal_set, optimal_utility) tuple
    """
    m = len(costs)
    
    # For small instances, use exhaustive search
    if m <= 20:
        best_utility = -np.inf
        best_set = set()
        
        for r in range(1, 2**m):
            subset = set()
            total_cost = 0.0
            for j in range(m):
                if (r >> j) & 1:
                    subset.add(j)
                    total_cost += costs[j]
            
            if total_cost <= budget:
                if use_cost_proportional:
                    utility = sum(costs[j] * qualities[j] for j in subset)
                else:
                    utility = sum(qualities[j] for j in subset)
                
                if utility > best_utility:
                    best_utility = utility
                    best_set = subset.copy()
        
        return best_set if best_set else set(), max(best_utility, 0.0)
    else:
        # For larger instances, use greedy approximation
        if use_cost_proportional:
            # For cost-proportional utility, sort by total value (cost * quality)
            values = np.array([costs[j] * qualities[j] for j in range(m)])
            sorted_indices = np.argsort(values)[::-1]
        else:
            # For normal utility, sort by quality/cost ratio
            ratios = np.array([qualities[j] / costs[j] if costs[j] > 0 else qualities[j] 
                              for j in range(m)])
            sorted_indices = np.argsort(ratios)[::-1]
        
        winning_set = set()
        remaining_budget = budget
        
        for j in sorted_indices:
            if costs[j] <= remaining_budget:
                winning_set.add(j)
                remaining_budget -= costs[j]
        
        if use_cost_proportional:
            utility = sum(costs[j] * qualities[j] for j in winning_set)
        else:
            utility = sum(qualities[j] for j in winning_set)
        
        return winning_set, utility


# ============================================================================
# Instance Generation
# ============================================================================

def generate_instance(n: int, m: int, alpha: float, budget: float, 
                     quality_range: Tuple[int, int], seed: int = None) -> dict:
    """
    Generate a random PB instance.
    
    Args:
        n: number of agents
        m: number of alternatives
        alpha: cost ratio (max cost / min cost)
        budget: budget constraint
        quality_range: (min_quality, max_quality)
        seed: random seed
    
    Returns:
        Dictionary with instance parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate costs: min = 1, max = alpha
    if alpha == 1.0:
        costs = np.ones(m)
    else:
        costs = np.random.uniform(1.0, alpha, m)
    costs = np.round(costs, 2)
    
    # Ensure at least one cost = 1 and one = alpha
    costs[0] = 1.0
    if m > 1:
        costs[-1] = alpha
    
    # Generate qualities (binary for simplicity, but can extend)
    min_qual, max_qual = quality_range
    qualities = np.random.randint(min_qual, max_qual + 1, m)
    
    # Generate common priors (probability that quality = max_qual)
    qual_priors = np.random.uniform(0.1, 0.9, m)
    
    # Generate signal distributions
    # For each alternative j and quality q, prob of positive signal
    signal_dists = {}
    for j in range(m):
        for q in range(min_qual, max_qual + 1):
            # Higher quality -> higher signal probability
            base_prob = 0.2 + 0.6 * (q - min_qual) / (max_qual - min_qual) if max_qual > min_qual else 0.6
            signal_dists[(j, q)] = np.random.uniform(base_prob - (0.3 / max_qual), base_prob + (0.3 / max_qual))
            signal_dists[(j, q)] = np.clip(signal_dists[(j, q)], 0.1, 0.9)
    
    # Generate signals for each agent
    signals = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            # Sample quality from prior
            qual = max_qual if np.random.random() < qual_priors[j] else min_qual
            # Sample signal based on quality
            signal_prob = signal_dists[(j, qual)]
            signals[i, j] = 1 if np.random.random() < signal_prob else 0
    
    return {
        'n': n,
        'm': m,
        'costs': costs,
        'budget': budget,
        'qualities': qualities,
        'qual_priors': qual_priors,
        'signal_dists': signal_dists,
        'signals': signals,
        'quality_range': quality_range
    }


# ============================================================================
# Performance Calculation (Ratio to Optimal Knapsack)
# ============================================================================

def calculate_informed_ratio(instance: dict, voting_rule: Callable, 
                            use_cost_proportional: bool = False, 
                            num_samples: int = 100) -> float:
    """
    Calculate performance (ratio to optimal knapsack) for a voting rule.
    
    Performance is defined as:
    E[utility(voting_rule_output)] / E[max_utility(optimal_set)]
    
    We approximate this by averaging over multiple random signal realizations.
    
    Args:
        instance: instance dictionary
        voting_rule: function(votes, costs, budget) -> winning_set
        use_cost_proportional: if True, use cost-proportional utility
        num_samples: number of samples for Monte Carlo
    
    Returns:
        Performance (ratio to optimal knapsack)
    """
    n = instance['n']
    m = instance['m']
    costs = instance['costs']
    budget = instance['budget']
    qual_priors = instance['qual_priors']
    signal_dists = instance['signal_dists']
    quality_range = instance['quality_range']
    min_qual, max_qual = quality_range
    
    total_voting_utility = 0.0
    total_optimal_utility = 0.0
    
    # Sample over quality realizations and signal realizations
    for sample in range(num_samples):
        # Sample quality vector
        qualities = np.zeros(m, dtype=int)
        for j in range(m):
            qualities[j] = max_qual if np.random.random() < qual_priors[j] else min_qual
        
        # Sample signals for all agents (informative voting: votes = signals)
        votes = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                signal_prob = signal_dists[(j, qualities[j])]
                votes[i, j] = 1 if np.random.random() < signal_prob else 0
        
        # Get winning set from voting rule
        winning_set = voting_rule(votes, costs, budget)
        
        # Calculate utility
        if use_cost_proportional:
            voting_utility = sum(costs[j] * qualities[j] for j in winning_set)
        else:
            voting_utility = sum(qualities[j] for j in winning_set)
        
        # Get optimal set
        optimal_set, optimal_utility = knapsack_optimal(qualities, costs, budget, use_cost_proportional)
        
        total_voting_utility += voting_utility
        total_optimal_utility += max(optimal_utility, 1e-10)  # Avoid division by zero
    
    return total_voting_utility / total_optimal_utility if total_optimal_utility > 0 else 0.0


# ============================================================================
# Simulation and Plotting
# ============================================================================

def run_simulation(n_values: List[int], m: int, alpha: float, budget: float,
                   quality_range: Tuple[int, int], utility_type: str = 'normal',
                   num_samples: int = 50, num_trials: int = 10) -> dict:
    """
    Run simulation for different values of n.
    
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
        Dictionary with results for each voting rule
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
        rule: {'mean': [], 'std': [], 'all': []}  # Added 'all' to store trial data
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
        
        # Calculate mean and std over trials, and store all trial data
        for rule_name in voting_rules.keys():
            ratios = rule_ratios[rule_name]
            results[rule_name]['mean'].append(np.mean(ratios))
            results[rule_name]['std'].append(np.std(ratios))
            results[rule_name]['all'].append(ratios.copy())  # Store all trial data
            print(f"  {rule_name}: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    
    return results, voting_rules.keys()  # Also return rule names


def plot_results(n_values: List[int], results: dict, title: str = "Performance vs Number of Agents", filename: str = None, show_std_bars = False):
    """Plot performance with error bars for different voting rules."""
    import os
    

    SHOW_STD_BARS = show_std_bars
    
    plt.figure(figsize=(12, 8))
    
    # Define colors, markers, and linestyles for each rule
    rule_styles = {
        'AV': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'markersize': 8},
        'AV/Cost': {'color': '#17becf', 'marker': 'H', 'linestyle': '-', 'markersize': 9},
        'GC': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'markersize': 8},
        'GC+AV': {'color': '#e377c2', 'marker': '*', 'linestyle': '--', 'markersize': 10},
        'MES': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'markersize': 8},
        'MES+AV': {'color': '#d62728', 'marker': 'v', 'linestyle': ':', 'markersize': 8},
        'Phragmen': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-', 'markersize': 8},
        'PAV': {'color': '#8c564b', 'marker': 'p', 'linestyle': '--', 'markersize': 8}
    }
    
    for rule_name in results.keys():
        means = results[rule_name]['mean']
        stds = results[rule_name]['std']
        style = rule_styles.get(rule_name, {'color': 'black', 'marker': 'o', 'linestyle': '-', 'markersize': 8})
        
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
                    linewidth=3, markersize=style['markersize'])
        else:
            plt.plot(n_values, means, 
                        marker=style['marker'], linestyle=style['linestyle'],
                        color=style['color'], label=rule_name, 
                        linewidth=3, markersize=style['markersize'])
    
    # Add horizontal line at y=1 (perfect performance)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (Performance=1)')
    
    plt.xlabel('Number of Agents (n)', fontsize=24)
    plt.ylabel('Performance', fontsize=24)
    plt.title(title, fontsize=28, fontweight='bold')
    plt.legend(fontsize=20, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to use most of the visual space
    all_means = [m for r in results.values() for m in r['mean']]
    if all_means:
        if SHOW_STD_BARS:
            all_stds = [s for r in results.values() for s in r['std']]
            if all_stds:
                y_min = max(0.0, min(all_means) - 2 * max(all_stds))
                y_max = min(1.05, max(all_means) + 2 * max(all_stds))
            else:
                y_min = max(0.0, min(all_means) - 0.05)
                y_max = min(1.05, max(all_means) + 0.05)
        else:
            y_min = max(0.0, min(all_means) - 0.05)
            y_max = min(1.05, max(all_means) + 0.05)
        # Ensure y_min is never negative
        y_min = max(0.0, y_min)
    else:
        y_min = 0.0
        y_max = 1.05
    # Ensure y_min < y_max to prevent axis inversion
    if y_min >= y_max:
        y_min = max(0.0, y_max - 0.1)
    plt.ylim([y_min, y_max])
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'performance_vs_n_{timestamp}.png'
    # Save to plots/simulation folder
    os.makedirs('plots/simulation', exist_ok=True)
    filepath = os.path.join('plots/simulation', filename)
    plt.savefig(filepath, dpi=300)
    print(f"Plot saved as '{filepath}'")
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
    n_values = list(range(10, 101, 10))  # n=10 to n=200 in steps of 10
    m = 8  # number of alternatives
    alpha = 5.0  # cost ratio (max/min) - non-unit cost simulation
    budget = 8.0  # budget constraint
    quality_range = (0, 2)  # binary qualities
    utility_type = 'cost_proportional'  # or 'normal'
    
    print("=" * 60)
    print("Participatory Budgeting Simulation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n values: {n_values}")
    print(f"  m (alternatives): {m}")
    print(f"  alpha (cost ratio): {alpha}")
    print(f"  budget: {budget}")
    print(f"  quality range: {quality_range}")
    print(f"  utility type: {utility_type}")
    print("=" * 60)
    
    # Run simulation for default parameters
    results, rule_names = run_simulation(n_values, m, alpha, budget, quality_range, 
                           utility_type, num_samples=30, num_trials=100)
    
    # Plot results
    plot_results(n_values, results, 
                title=f"Performance vs n (m={m}, α={alpha}, B={budget})",show_std_bars = True)
    
    # Statistical Analysis
    test_results, win_counts = run_pairwise_tests(results, rule_names, n_values, x_label='n')
    print_statistical_results(test_results, win_counts, rule_names, n_values, x_label='n')
    print_win_matrix(win_counts, rule_names, n_values)
    print_effect_size_interpretation()
    
    print("\nSimulation complete!")

